"""CPU gates for R061 deployment inputs and shared structured contract."""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

from experiments.r061_qwen38_m1 import (
    MAX_OUTPUT_TOKENS,
    _call_case,
    _chat_token_count,
    _exact_config,
    _forced_truncation_gate,
    _run_probe_optimizer,
    build_dev64_subset,
    finalize_matched_arms,
    parser_negative_gate,
    production_schema,
    run_qualification_phase,
    smoke_cases,
)
from toolusepcg.config import load_config
from toolusepcg.optimizer.greedy import GreedyOptimizer


class TestR061FrozenInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subset = build_dev64_subset()

    def test_dev64_is_deterministic_and_stratified(self) -> None:
        self.assertEqual(len(self.subset["cases"]), 64)
        self.assertEqual(len({case["target_id"] for case in self.subset["cases"]}), 64)
        self.assertEqual(self.subset, build_dev64_subset())
        for token, minimum in self.subset["requirements"].items():
            self.assertGreaterEqual(self.subset["observed_counts"][token], minimum)
        self.assertTrue(all(case["seed"] == 61064 + index for index, case in enumerate(self.subset["cases"])))
        self.assertTrue(all(len(case["initialization_sha256"]) == 64 for case in self.subset["cases"]))

    def test_smoke_suite_is_exactly_32_fixed_cases(self) -> None:
        cases = smoke_cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({case["case_id"] for case in cases}), 32)
        categories = {case["category"] for case in cases}
        self.assertTrue({"boundary", "multi_call", "stop_pressure", "retry", "determinism"} <= categories)
        self.assertEqual(cases[0]["checks"]["coordinates"], [[0, 0]])
        self.assertEqual(cases[5]["checks"]["tool_call_count"], 3)
        self.assertEqual(cases[8]["checks"]["tool_names"], ["generate_zelda_maze"])
        self.assertEqual(cases[7]["checks"]["modes"], ["rect", "single"])
        self.assertEqual(cases[28]["checks"]["modes"], ["rect"])

    def test_prompt_probe_can_lower_native_reasoning_cap(self) -> None:
        config = _exact_config(17, 31, max_tokens=4096)
        self.assertEqual(config.llm.max_tokens, 4096)
        self.assertEqual(MAX_OUTPUT_TOKENS, 8192)

    def test_imported_module_names_are_not_shadowed(self) -> None:
        source = Path("experiments/r061_qwen38_m1.py")
        tree = ast.parse(source.read_text(), filename=str(source))
        module_names = {
            alias.asname or alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        shadows: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                names.extend(arg.arg for arg in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ))
                if node.args.vararg:
                    names.append(node.args.vararg.arg)
                if node.args.kwarg:
                    names.append(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(node.id)
            shadows.extend(
                (getattr(node, "lineno", 0), name)
                for name in names if name in module_names
            )
        self.assertEqual(shadows, [])

    def test_matched_pair_finalizer_fails_closed_on_contract_drift(self) -> None:
        base = {
            "contract_gate_pass": True,
            "subset_sha256": "a" * 64,
            "case_contract_sha256": "b" * 64,
            "budgets": {"semantic_actions": 8},
            "sampling": {"temperature": 0.0},
            "model_id": "incumbent",
            "revision": "c" * 40,
            "aggregate": {"exact": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incumbent = root / "incumbent.json"
            qwen38 = root / "qwen38.json"
            output = root / "summary.json"
            incumbent.write_text(json.dumps(base))
            qwen38.write_text(json.dumps({
                **base, "model_id": "qwen38", "revision": "d" * 40,
            }))
            result = finalize_matched_arms(incumbent, qwen38, output)
            self.assertTrue(result["matched_contract_gate_pass"])
            qwen38.write_text(json.dumps({
                **base,
                "model_id": "qwen38",
                "revision": "d" * 40,
                "budgets": {"semantic_actions": 7},
            }))
            with self.assertRaises(RuntimeError):
                finalize_matched_arms(incumbent, qwen38, output)


class TestR061OneStructuredContract(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = production_schema()
        self.valid = {
            "type": "STEP",
            "rationale": "one legal edit",
            "plan": "place one wall",
            "tool_calls": [{
                "tool_name": "place_tile",
                "parameters": {
                    "mode": "single", "tile_type": "wall", "y": 0, "x": 0,
                },
            }],
            "expected_delta": {"player_key": 0, "key_door": 0},
            "acceptance_hint": "accept only if hard-valid",
        }

    def _optimizer(self) -> GreedyOptimizer:
        cfg = load_config("configs/zelda.yaml", {
            "targets": {
                "player_key": {"mode": "target", "value": 17},
                "key_door": {"mode": "target", "value": 31},
            },
            "optimizer": {"init_strategy": "empty"},
            "llm": {
                "provider": "openai",
                "model": "test-model",
                "base_url": "http://127.0.0.1:1/v1",
                "grammar_domain": "zelda",
                "grammar_max_tool_calls": 20,
            },
        })
        optimizer = GreedyOptimizer(cfg, verbose=False)
        optimizer.initialize(seed=0)
        return optimizer

    def test_decoder_schema_is_the_parser_schema(self) -> None:
        optimizer = self._optimizer()
        with mock.patch.dict(os.environ, {"RPE_PCG_COORD_GRAMMAR": "on"}):
            served = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
        self.assertEqual(served, optimizer._openai_step_schema)
        self.assertEqual(served, self.schema)
        Draft202012Validator.check_schema(served)

    def test_strict_parser_rejects_repair_and_truncation(self) -> None:
        optimizer = self._optimizer()
        good = json.dumps(self.valid)
        message, errors = optimizer.parse_response(good)
        self.assertIsNotNone(message)
        self.assertFalse(errors)
        for bad, truncated in ((good[:-1], False), (good + " trailing", False), (good, True)):
            with self.subTest(bad=bad[-12:], truncated=truncated):
                message, errors = optimizer.parse_response(bad, truncated=truncated)
                self.assertIsNone(message)
                self.assertTrue(errors)

    def test_expected_delta_and_nonempty_calls_are_required(self) -> None:
        for mutation in (
            {key: value for key, value in self.valid.items() if key != "expected_delta"},
            {**self.valid, "tool_calls": []},
        ):
            with self.assertRaises(Exception):
                Draft202012Validator(self.schema).validate(mutation)

    def test_frozen_parser_negative_fixtures(self) -> None:
        result = parser_negative_gate()
        self.assertTrue(result["gate_pass"])
        self.assertTrue(all(result["cases"].values()))

    def test_truncation_recovery_switches_to_thinking_off_once(self) -> None:
        class NeverDone:
            @staticmethod
            def should_terminate(_result):
                return False

        class FakeOptimizer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    llm=types.SimpleNamespace(
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                    ),
                    optimizer=types.SimpleNamespace(extra_instruction=None),
                )
                self.termination = NeverDone()
                self.current_result = object()
                self.step_count = 0
                self.step_records = []

            def step(self):
                self.step_count += 1
                record = types.SimpleNamespace(
                    step_num=self.step_count,
                    truncated=self.step_count == 1,
                    message_type="PARSE_ERROR" if self.step_count == 1 else "STEP",
                    reasoning_content="finished maze plan" if self.step_count == 1 else "",
                )
                self.step_records.append(record)
                return record

        optimizer = FakeOptimizer()
        recovery = _run_probe_optimizer(
            optimizer,
            max_steps=2,
            truncation_recovery="thinking_off_once",
        )
        self.assertTrue(recovery["attempted"])
        self.assertEqual(recovery["trigger_step"], 1)
        self.assertEqual(recovery["fallback_calls"], 1)
        self.assertEqual(recovery["recovered_truncations"], 1)
        self.assertTrue(optimizer.step_records[0].thinking_enabled)
        self.assertFalse(optimizer.step_records[1].thinking_enabled)
        self.assertFalse(
            optimizer.config.llm.extra_body["chat_template_kwargs"]["enable_thinking"]
        )

    def test_truncation_recovery_hands_reasoning_to_one_fallback(self) -> None:
        class NeverDone:
            @staticmethod
            def should_terminate(_result):
                return False

        class FakeOptimizer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    llm=types.SimpleNamespace(
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                    ),
                    optimizer=types.SimpleNamespace(extra_instruction="base rule"),
                )
                self.termination = NeverDone()
                self.current_result = object()
                self.step_count = 0
                self.step_records = []
                self.seen_instructions = []

            def step(self):
                self.seen_instructions.append(self.config.optimizer.extra_instruction)
                self.step_count += 1
                record = types.SimpleNamespace(
                    step_num=self.step_count,
                    truncated=self.step_count == 1,
                    message_type="PARSE_ERROR" if self.step_count == 1 else "STEP",
                    reasoning_content="finished maze plan" if self.step_count == 1 else "",
                )
                self.step_records.append(record)
                return record

        optimizer = FakeOptimizer()
        recovery = _run_probe_optimizer(
            optimizer,
            max_steps=2,
            truncation_recovery="thinking_off_handoff",
        )
        self.assertEqual(recovery["reasoning_handoff_chars"], len("finished maze plan"))
        self.assertEqual(optimizer.seen_instructions[0], "base rule")
        self.assertIn("finished maze plan", optimizer.seen_instructions[1])
        self.assertIn("base rule", optimizer.seen_instructions[1])
        self.assertEqual(optimizer.config.optimizer.extra_instruction, "base rule")
        self.assertEqual(optimizer.step_records[1].reasoning_handoff_chars, 18)

    def test_split_cap_restores_full_budget_for_handoff(self) -> None:
        class NeverDone:
            @staticmethod
            def should_terminate(_result):
                return False

        class FakeOptimizer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    llm=types.SimpleNamespace(
                        max_tokens=8192,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                    ),
                    optimizer=types.SimpleNamespace(extra_instruction=None),
                )
                self.termination = NeverDone()
                self.current_result = object()
                self.step_count = 0
                self.step_records = []
                self.seen_max_tokens = []

            def step(self):
                self.seen_max_tokens.append(self.config.llm.max_tokens)
                self.step_count += 1
                record = types.SimpleNamespace(
                    step_num=self.step_count,
                    truncated=self.step_count == 1,
                    message_type="PARSE_ERROR" if self.step_count == 1 else "STEP",
                    reasoning_content="partial plan" if self.step_count == 1 else "",
                )
                self.step_records.append(record)
                return record

        optimizer = FakeOptimizer()
        recovery = _run_probe_optimizer(
            optimizer,
            max_steps=2,
            truncation_recovery="thinking_off_handoff",
            thinking_max_tokens=4096,
            fallback_max_tokens=8192,
        )
        self.assertEqual(recovery["recovered_truncations"], 1)
        self.assertEqual(optimizer.seen_max_tokens, [4096, 8192])

    def test_latest_correction_handoff_overrides_earlier_candidate(self) -> None:
        class NeverDone:
            @staticmethod
            def should_terminate(_result):
                return False

        class FakeOptimizer:
            def __init__(self):
                self.config = types.SimpleNamespace(
                    llm=types.SimpleNamespace(
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                    ),
                    optimizer=types.SimpleNamespace(extra_instruction=None),
                )
                self.termination = NeverDone()
                self.current_result = object()
                self.step_count = 0
                self.step_records = []
                self.seen_instructions = []

            def step(self):
                self.seen_instructions.append(self.config.optimizer.extra_instruction)
                self.step_count += 1
                record = types.SimpleNamespace(
                    step_num=self.step_count,
                    truncated=self.step_count == 1,
                    message_type="PARSE_ERROR" if self.step_count == 1 else "STEP",
                    reasoning_content=(
                        "complete but invalid plan. Wait! final corrected geometry"
                        if self.step_count == 1 else ""
                    ),
                )
                self.step_records.append(record)
                return record

        optimizer = FakeOptimizer()
        recovery = _run_probe_optimizer(
            optimizer,
            max_steps=2,
            truncation_recovery="thinking_off_handoff_latest",
        )
        self.assertEqual(recovery["recovered_truncations"], 1)
        self.assertIn("Later self-corrections override earlier candidates", optimizer.seen_instructions[1])
        self.assertIn("final corrected geometry", optimizer.seen_instructions[1])


class TestR061QualificationReporting(unittest.TestCase):
    def test_batch_encoding_counts_input_ids_not_mapping_fields(self) -> None:
        class BatchEncoding(dict):
            pass

        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return BatchEncoding({
                    "input_ids": list(range(37)),
                    "attention_mask": [1] * 37,
                })

        self.assertEqual(
            _chat_token_count(FakeTokenizer(), [{"role": "user", "content": "x"}]),
            37,
        )

    def test_response_shape_failures_are_records_not_suite_exceptions(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return [1, 2]

        case = smoke_cases()[0]
        malformed = (
            types.SimpleNamespace(
                choices=[],
                usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            ),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    finish_reason="stop",
                    message=types.SimpleNamespace(content="{}"),
                )],
                usage=None,
            ),
        )
        for response in malformed:
            with self.subTest(response=response):
                completions = types.SimpleNamespace(create=lambda **_kwargs: response)
                client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(completions=completions)
                )
                record = _call_case(
                    client, FakeTokenizer(), case, {}, service_mode="shape-negative"
                )
                self.assertFalse(record["parse_ok"])
                self.assertFalse(record["contract_ok"])
                self.assertFalse(record["input_token_parity"])
                self.assertTrue(record["parse_error"])

    def test_forced_truncation_transport_failure_is_not_a_parser_pass(self) -> None:
        completions = types.SimpleNamespace(
            create=lambda **_kwargs: (_ for _ in ()).throw(
                TimeoutError("forced truncation request timed out")
            )
        )
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        result = _forced_truncation_gate(client, smoke_cases()[0]["messages"], {})
        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["request_completed"])
        self.assertIn("TimeoutError", result["error"])

    def test_truncation_is_recorded_fail_closed_and_all_32_cases_run(self) -> None:
        valid = {
            "type": "STEP",
            "rationale": "ok",
            "plan": "one edit",
            "tool_calls": [{
                "tool_name": "place_tile",
                "parameters": {
                    "mode": "single", "tile_type": "wall", "y": 0, "x": 0,
                },
            }],
            "expected_delta": {"player_key": 0, "key_door": 0},
            "acceptance_hint": "accept if hard-valid",
        }

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def apply_chat_template(self, *_args, **_kwargs):
                return [1, 2]

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                truncated = len(self.calls) == 1
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        finish_reason="length" if truncated else "stop",
                        message=types.SimpleNamespace(
                            content="{\"type\":\"STEP\"" if truncated else json.dumps(valid)
                        ),
                    )],
                    usage=types.SimpleNamespace(
                        prompt_tokens=2,
                        completion_tokens=MAX_OUTPUT_TOKENS if truncated else 20,
                    ),
                )

        completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )

        def strict_parse_witness(text: str, *, truncated: bool = False):
            if truncated:
                raise ValueError("truncated structured response")
            return json.loads(text)

        fake_modules = {
            "openai": types.SimpleNamespace(OpenAI=lambda **_kwargs: fake_client),
            "transformers": types.SimpleNamespace(AutoTokenizer=FakeTokenizer),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = root / "identity.json"
            identity.write_text(json.dumps({"snapshot": str(root / "snapshot")}))
            output = root / "qualification_restart.json"
            (root / "qualification_coldwarm.json").write_text(json.dumps({"anchors": {}}))
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch("experiments.r061_qwen38_m1.production_schema", return_value={}),
                mock.patch("experiments.r061_qwen38_m1.strict_parse", side_effect=strict_parse_witness),
                mock.patch("experiments.r061_qwen38_m1._server_model_id", return_value="Qwen/Qwen3.8-27B"),
                mock.patch("experiments.r061_qwen38_m1._gpu_memory", return_value=[]),
                mock.patch("experiments.r061_qwen38_m1._gpu_runtime", return_value={}),
            ):
                with self.assertRaisesRegex(RuntimeError, "qualification phase restart failed"):
                    run_qualification_phase(
                        identity, "http://127.0.0.1:8101/v1", output, "restart"
                    )

            report = json.loads(output.read_text())
            self.assertEqual(len(completions.calls), 32)
            self.assertTrue(all(call["max_tokens"] == MAX_OUTPUT_TOKENS for call in completions.calls))
            self.assertEqual(len(report["records"]), 32)
            self.assertFalse(report["records"][0]["parse_ok"])
            self.assertEqual(report["records"][0]["finish_reason"], "length")
            self.assertIn("truncated structured response", report["records"][0]["parse_error"])
            self.assertFalse(report["gate_pass"])

    @unittest.skipUnless(__import__("importlib.util", fromlist=["find_spec"]).find_spec("transformers"), "optional transformers serving qualification")
    def test_full_qualification_dry_run_executes_every_panel(self) -> None:
        cases_by_seed = {case["seed"]: case for case in smoke_cases()}

        def place_call(
            *, mode: str = "single", tile_type: str = "wall",
            y: int = 0, x: int = 0,
        ) -> dict[str, object]:
            parameters: dict[str, object] = {
                "mode": mode, "tile_type": tile_type, "y": y, "x": x,
            }
            if mode == "line":
                parameters["end_x"] = min(15, x + 1)
            elif mode == "rect":
                parameters.update({"end_y": min(15, y + 1), "end_x": min(15, x + 1)})
            return {"tool_name": "place_tile", "parameters": parameters}

        def valid_content(seed: int) -> str:
            checks = cases_by_seed.get(seed, {}).get("checks", {})
            result: dict[str, object] = {
                "type": "STEP",
                "rationale": "deterministic dry-run rationale",
                "plan": "one legal edit",
                "tool_calls": [place_call()],
                "expected_delta": {"player_key": 0, "key_door": 0},
                "acceptance_hint": "accept if hard-valid",
            }
            if "coordinates" in checks:
                y, x = checks["coordinates"][0]
                result["tool_calls"] = [place_call(y=y, x=x)]
            elif "tool_call_count" in checks:
                result["tool_calls"] = [
                    place_call(x=index) for index in range(checks["tool_call_count"])
                ]
            elif "modes" in checks:
                result["tool_calls"] = [place_call(mode=mode) for mode in checks["modes"]]
            elif "tool_names" in checks:
                result["tool_calls"] = [
                    {"tool_name": name, "parameters": {}}
                    for name in checks["tool_names"]
                ]
            elif "tile_types" in checks:
                result["tool_calls"] = [place_call(tile_type=checks["tile_types"][0])]
            if "delta_sign" in checks:
                key, sign = checks["delta_sign"]
                result["expected_delta"][key] = sign
            if "delta_exact" in checks:
                result["expected_delta"] = checks["delta_exact"]
            if "large_abs_delta" in checks:
                result["expected_delta"]["player_key"] = checks["large_abs_delta"]
            if "rationale_contains" in checks:
                result["rationale"] = f"contains {checks['rationale_contains']} witness"
            return json.dumps(result, ensure_ascii=False, sort_keys=True)

        class FakeTokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                content = messages[-1]["content"]
                return range(max(1, content.count(" corridor")))

        state = {
            "active": 0,
            "max_active": 0,
            "requests": [],
            "urlopen_paths": [],
            "retry_attempts": 0,
        }
        state_lock = threading.Lock()

        class MainCompletions:
            def create(self, **request):
                seed = int(request.get("seed", 0))
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    state["requests"].append(request)
                try:
                    # The cold/warm concurrent panel uses seeds 61124..61131.
                    # A short delay makes overlap observable without mocking the executor.
                    if 61124 <= seed <= 61131:
                        time.sleep(0.03)
                    truncated = request.get("max_tokens") == 1
                    messages = request["messages"]
                    prompt_tokens = max(1, messages[-1]["content"].count(" corridor"))
                    content = "{\"type\":\"STEP\"" if truncated else valid_content(seed)
                    return types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            message=types.SimpleNamespace(content=content),
                            finish_reason="length" if truncated else "stop",
                        )],
                        usage=types.SimpleNamespace(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=1 if truncated else 20,
                        ),
                    )
                finally:
                    with state_lock:
                        state["active"] -= 1

        main_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=MainCompletions())
        )

        class FailingCompletions:
            def create(self, **_request):
                raise ConnectionError("dry-run unavailable endpoint")

        class RetryCompletions:
            def create(self, **_request):
                from experiments.r061_qwen38_m1 import _BrokenTransportHandler

                # Exercise the retry witness as two attempts: a synthetic 503,
                # followed by one strict-parseable response.
                _BrokenTransportHandler.retry_requests = 1
                state["retry_attempts"] += 1
                _BrokenTransportHandler.retry_requests = 2
                state["retry_attempts"] += 1
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content=valid_content(0)),
                        finish_reason="stop",
                    )],
                    usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=20),
                )

        class FakeTransportServer:
            def __init__(self, *_args, **_kwargs) -> None:
                self.server_address = ("127.0.0.1", 18061)

            def serve_forever(self) -> None:
                return

            def shutdown(self) -> None:
                return

            def server_close(self) -> None:
                return

        base_url = "http://qualification.stub/v1"

        def openai_factory(*, base_url: str, **_kwargs):
            if base_url == "http://127.0.0.1:1/v1":
                return types.SimpleNamespace(
                    chat=types.SimpleNamespace(completions=FailingCompletions())
                )
            if base_url == "http://127.0.0.1:18061/v1":
                return types.SimpleNamespace(
                    chat=types.SimpleNamespace(completions=RetryCompletions())
                )
            self.assertEqual(base_url, "http://qualification.stub/v1")
            return main_client

        def fake_urlopen(url: str, **_kwargs):
            state["urlopen_paths"].append(url)
            if url.endswith("/v1/models"):
                payload = {
                    "data": [{"id": "Qwen/Qwen3.8-27B"}],
                }
                return types.SimpleNamespace(read=lambda: json.dumps(payload).encode())
            if url.endswith("/metrics"):
                payload = (
                    "vllm:prompt_tokens_total{model_name=\"stub\"} 100\n"
                    "vllm:generation_tokens_total{model_name=\"stub\"} 20\n"
                    "vllm:gpu_cache_usage_perc{model_name=\"stub\"} 0.1\n"
                )
                return types.SimpleNamespace(read=lambda: payload.encode())
            if url.endswith("/partial"):
                raise ConnectionError("dry-run partial body")
            if url.endswith("/timeout"):
                raise TimeoutError("dry-run timeout")
            raise AssertionError(f"unexpected dry-run URL: {url}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = root / "identity.json"
            identity.write_text(json.dumps({"snapshot": str(root / "snapshot")}))
            coldwarm_path = root / "qualification_coldwarm.json"
            restart_path = root / "qualification_restart.json"
            with (
                mock.patch("openai.OpenAI", side_effect=openai_factory),
                mock.patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=FakeTokenizer(),
                ),
                mock.patch(
                    "experiments.r061_qwen38_m1.ThreadingHTTPServer",
                    FakeTransportServer,
                ),
                mock.patch(
                    "experiments.r061_qwen38_m1.urlopen",
                    side_effect=fake_urlopen,
                ),
                mock.patch(
                    "experiments.r061_qwen38_m1._gpu_memory",
                    return_value=[{
                        "index": 0,
                        "name": "dry-run-gpu",
                        "used_mib": 1,
                        "total_mib": 2,
                    }],
                ),
                mock.patch("experiments.r061_qwen38_m1._gpu_runtime", return_value={}),
            ):
                coldwarm = run_qualification_phase(
                    identity, base_url, coldwarm_path, "coldwarm"
                )
                restart = run_qualification_phase(
                    identity, base_url, restart_path, "restart"
                )

        self.assertTrue(coldwarm["gate_pass"])
        self.assertEqual(len(coldwarm["records"]), 32)
        modes = [record["service_mode"] for record in coldwarm["records"]]
        self.assertEqual(modes.count("cold"), 1)
        self.assertEqual(modes.count("warm"), 23)
        self.assertEqual(modes.count("concurrent"), 8)
        self.assertEqual(len(coldwarm["context_records"]), 4)
        self.assertTrue(coldwarm["negative_gates"]["forced_server_truncation_fail_closed"])
        self.assertTrue(coldwarm["negative_gates"]["explicit_retry"])
        self.assertTrue(coldwarm["negative_gates"]["parser"]["gate_pass"])
        self.assertEqual(
            coldwarm["negative_gates"]["transport"]["cases"],
            {"partial": True, "timeout": True, "server_503_retry": True},
        )
        self.assertTrue(restart["gate_pass"])
        self.assertEqual(len(restart["records"]), 32)
        self.assertTrue(restart["semantic_determinism_across_restart"])
        self.assertTrue(restart["byte_determinism_across_restart_diagnostic"])
        self.assertTrue(coldwarm["execution_complete"])
        self.assertGreater(coldwarm["throughput"]["output_tokens_per_panel_wall_second"], 0)
        self.assertTrue(coldwarm["server_metrics"]["before"]["available"])
        self.assertEqual(coldwarm["gpu_memory_peak_observed_mib"], {0: 1})
        self.assertGreaterEqual(state["max_active"], 2)
        self.assertEqual(state["retry_attempts"], 2)
        self.assertTrue(any(path.endswith("/partial") for path in state["urlopen_paths"]))
        self.assertTrue(any(path.endswith("/timeout") for path in state["urlopen_paths"]))
        normal_requests = [
            request for request in state["requests"] if request.get("max_tokens") != 1
        ]
        self.assertTrue(normal_requests)
        self.assertTrue(all(
            request["max_tokens"] == MAX_OUTPUT_TOKENS for request in normal_requests
        ))
        self.assertTrue(all(
            request["messages"][0]["role"] == "system"
            and "JSON_SCHEMA=" in request["messages"][0]["content"]
            for request in normal_requests
        ))




if __name__ == "__main__":
    unittest.main()

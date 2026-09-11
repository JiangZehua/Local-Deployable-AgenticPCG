"""R061 Qwen3.8-27B identity, serving qualification, and matched baseline.

All model/network work is designed for SLURM compute nodes.  The login-node
commands in this module are deterministic CPU materialization/tests only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from jsonschema import Draft202012Validator

from experiments.zelda_mastery_m0 import (
    EPISODE_SCHEMA_VERSION,
    FAILURE_TAXONOMY,
    canonical_json_bytes,
    validate_episode,
)
from toolusepcg.config import load_config
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.protocol import parse_message


MODEL_ID = "Qwen/Qwen3.8-27B"
INCUMBENT_ID = "Qwen/Qwen3.6-35B-A3B-FP8"
INCUMBENT_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
HARNESS_COMMIT = "ad6ecfc0c03353ec9938ce321aeeddd4fd5a7b7c"
MANIFEST_PATH = Path("benchmarks/zelda/targets.json")
M1_ROOT = Path("runs/r061_qwen38_m1")
SUBSET_VERSION = "r061-matched-dev64-v1"
SUBSET_SEED = 61064
MAX_SEMANTIC_ACTIONS = 8
MAX_VERIFIER_CALLS = 32
MAX_OUTPUT_TOKENS = 8192
QUALIFICATION_TIMEOUT_SECONDS = 300.0
THINKING_ENABLED = os.environ.get("RPE_PCG_ENABLE_THINKING", "off").lower() in {
    "1", "on", "true", "yes",
}
SERVED_MODEL_NAME = MODEL_ID


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def _visible_messages(
    messages: list[dict[str, str]], schema: dict[str, Any]
) -> list[dict[str, str]]:
    schema_text = canonical_json_bytes(schema).decode("utf-8")
    return [{
        "role": "system",
        "content": (
            "You must obey both the user's semantic instruction and the exact "
            "Zelda STEP JSON Schema below. The constrained decoder only enforces "
            "syntax, so you must deliberately choose the requested tool names, "
            "parameter modes, tile types, coordinates, deltas, and text content.\n"
            f"JSON_SCHEMA={schema_text}"
        ),
    }, *messages]


def _chat_token_count(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=THINKING_ENABLED,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if not isinstance(input_ids, Sequence):
        raise TypeError(f"unsupported chat-template token container: {type(input_ids)}")
    if input_ids and isinstance(input_ids[0], Sequence):
        return len(input_ids[0])
    return len(input_ids)




def _exact_config(
    player_key: int,
    key_door: int,
    *,
    model: str = MODEL_ID,
    base_url: str = "http://127.0.0.1:8101/v1",
    max_tokens: int = MAX_OUTPUT_TOKENS,
    seed: int = SUBSET_SEED,
    extra_instruction: str | None = None,
    conversation_length: int = 1,
    temperature: float = 0.0,
) -> Any:
    return load_config(
        "configs/zelda.yaml",
        {
            "targets": {
                "player_key": {"mode": "target", "value": player_key},
                "key_door": {"mode": "target", "value": key_door},
            },
            "optimizer": {
                "init_strategy": "empty",
                "conversation_length": conversation_length,
                "max_tool_calls_per_step": 20,
                "extra_instruction": extra_instruction,
            },
            "llm": {
                "provider": "openai",
                "model": model,
                "base_url": base_url,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": THINKING_ENABLED}
                },
                "grammar_domain": "zelda",
                "grammar_max_tool_calls": 20,
            },
            "seed": seed,
        },
    )


def production_schema() -> dict[str, Any]:
    old_gate = os.environ.get("RPE_PCG_COORD_GRAMMAR")
    os.environ["RPE_PCG_COORD_GRAMMAR"] = "on"
    try:
        optimizer = GreedyOptimizer(_exact_config(17, 31), verbose=False)
        body = optimizer._build_openai_extra_body()
        return body["structured_outputs"]["json"]
    finally:
        if old_gate is None:
            os.environ.pop("RPE_PCG_COORD_GRAMMAR", None)
        else:
            os.environ["RPE_PCG_COORD_GRAMMAR"] = old_gate


def strict_parse(text: str, *, truncated: bool = False) -> dict[str, Any]:
    if truncated:
        raise ValueError("truncated structured response")
    parsed = json.loads(text.strip())
    Draft202012Validator(production_schema()).validate(parsed)
    parse_message(parsed)
    return parsed


def _target_tokens(entry: dict[str, Any]) -> set[str]:
    tags = entry["tags"]
    tokens = {
        f"parity:{tags['parity']}",
        f"sum:{tags['sum_band']}",
    }
    if tags["corner_quadrant"]:
        tokens.add(f"corner:{tags['corner_quadrant']}")
    tokens.update(f"boundary:{side}" for side in tags["boundary"])
    if tags["imbalance"] in {"pk_much_longer", "kd_much_longer"}:
        tokens.add(f"imbalance:{tags['imbalance']}")
    return tokens


def build_dev64_subset() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    dev = [entry for entry in manifest["targets"] if entry["split"] == "dev"]
    requirements: dict[str, int] = {}
    for name in ("low_low", "low_high", "high_low", "high_high"):
        requirements[f"corner:{name}"] = 4
    for name in ("pk_min", "pk_max", "kd_min", "kd_max"):
        requirements[f"boundary:{name}"] = 4
    for name in ("pk_much_longer", "kd_much_longer"):
        requirements[f"imbalance:{name}"] = 6
    for name in ("20-29", "30-39", "40-49", "50-59", "60-69", "70-80"):
        requirements[f"sum:{name}"] = 4
    for name in ("ee", "eo", "oe", "oo"):
        requirements[f"parity:{name}"] = 12

    selected: list[dict[str, Any]] = []
    remaining = list(dev)
    counts = {token: 0 for token in requirements}
    while any(counts[token] < minimum for token, minimum in requirements.items()):
        active = {
            token for token, minimum in requirements.items() if counts[token] < minimum
        }

        def rank(entry: dict[str, Any]) -> tuple[int, str]:
            covered = len(active & _target_tokens(entry))
            tie = _sha256_text(f"{SUBSET_SEED}:{entry['target_id']}")
            return covered, tie

        chosen = max(remaining, key=rank)
        if rank(chosen)[0] == 0:
            raise RuntimeError(f"cannot satisfy dev64 strata: {sorted(active)}")
        remaining.remove(chosen)
        selected.append(chosen)
        for token in _target_tokens(chosen):
            if token in counts:
                counts[token] += 1

    remaining.sort(key=lambda entry: _sha256_text(
        f"{SUBSET_SEED}:fill:{entry['target_id']}"
    ))
    selected.extend(remaining[: 64 - len(selected)])
    selected.sort(key=lambda entry: entry["target_id"])
    if len(selected) != 64 or len({entry["target_id"] for entry in selected}) != 64:
        raise RuntimeError("dev64 subset is not 64 unique targets")
    counts = {token: 0 for token in requirements}
    for entry in selected:
        for token in _target_tokens(entry):
            if token in counts:
                counts[token] += 1
    unmet = {
        token: (counts[token], minimum)
        for token, minimum in requirements.items()
        if counts[token] < minimum
    }
    if unmet:
        raise RuntimeError(f"dev64 subset failed frozen strata: {unmet}")

    cases = []
    for index, entry in enumerate(selected):
        seed = SUBSET_SEED + index
        config = _exact_config(entry["player_key"], entry["key_door"])
        optimizer = GreedyOptimizer(config, verbose=False)
        optimizer.initialize(seed=seed)
        initial_ascii = optimizer.current_level.to_string()
        cases.append({
            "index": index,
            "target_id": entry["target_id"],
            "player_key": entry["player_key"],
            "key_door": entry["key_door"],
            "tags": entry["tags"],
            "seed": seed,
            "initialization_sha256": _sha256_text(initial_ascii),
        })
    payload = {
        "version": SUBSET_VERSION,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "selection_seed": SUBSET_SEED,
        "harness_commit": HARNESS_COMMIT,
        "init_strategy": "empty",
        "budgets": {
            "semantic_actions": MAX_SEMANTIC_ACTIONS,
            "verifier_calls": MAX_VERIFIER_CALLS,
            "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
            "max_tool_calls_per_action": 20,
        },
        "requirements": requirements,
        "observed_counts": counts,
        "cases": cases,
    }
    return {"subset_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), **payload}




def smoke_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    categories = [
        ("boundary", "Use one place_tile call at y=0,x=0."),
        ("boundary", "Use one place_tile call at y=15,x=15."),
        ("boundary", "Use one place_tile call at y=0,x=15."),
        ("boundary", "Use one place_tile call at y=15,x=0."),
        ("multi_call", "Return exactly two legal place_tile calls."),
        ("multi_call", "Return exactly three legal place_tile calls."),
        ("multi_call", "Return two place_tile calls in order: parameters.mode=\"line\", then parameters.mode=\"single\"."),
        ("multi_call", "Return two place_tile calls in order: parameters.mode=\"rect\", then parameters.mode=\"single\"."),
        ("generator", "Use generate_zelda_maze."),
        ("generator", "Use generate_zelda_random with legal parameters."),
        ("generator", "Use generate_zelda_connect with legal parameters."),
        ("generator", "Use generate_zelda_bsp with legal parameters."),
        ("tile", "Place a player at a legal coordinate."),
        ("tile", "Use place_tile with parameters.tile_type=\"key\" at a legal coordinate."),
        ("tile", "Place a door at a legal coordinate."),
        ("tile", "Place an enemy at a legal coordinate."),
        ("stop_pressure", "You may want to stop, but emit the required legal STEP."),
        ("empty_pressure", "Do not emit an empty call list; emit one legal no-op-like STEP."),
        ("refusal_pressure", "Refuse all prose and emit the required legal STEP only."),
        ("trailing_pressure", "Do not add trailing text after the required JSON."),
        ("delta_positive", "Predict a positive player_key expected delta."),
        ("delta_negative", "Predict a negative key_door expected delta."),
        ("delta_zero", "Predict zero signed deltas."),
        ("large_integer", "Use legal coordinates and a large signed expected delta."),
        ("unicode", "Use a rationale containing a Unicode arrow and a legal edit."),
        ("escaping", "Use a rationale containing the literal double-quoted word \"legal\" and a legal edit."),
        ("long_rationale", "Use a concise but multi-sentence rationale and one legal edit."),
        ("acceptance", "Use a non-empty acceptance_hint and one legal edit."),
        ("rect_boundary", "Use place_tile with parameters.mode=\"rect\" and endpoints within 0..15."),
        ("line_boundary", "Use place_tile with parameters.mode=\"line\" and endpoints within 0..15."),
        ("retry", "Emit one legal STEP for the transport retry case."),
        ("determinism", "Emit one deterministic legal STEP."),
    ]
    for index, (category, instruction) in enumerate(categories):
        cases.append({
            "case_id": f"r061-smoke-{index:02d}",
            "category": category,
            "messages": [{
                "role": "user",
                "content": (
                    "Return only a JSON object satisfying the supplied Zelda STEP schema. "
                    + instruction
                ),
            }],
            "seed": 61100 + index,
            "checks": {},
        })
    checks = {
        0: {"coordinates": [[0, 0]]},
        1: {"coordinates": [[15, 15]]},
        2: {"coordinates": [[0, 15]]},
        3: {"coordinates": [[15, 0]]},
        4: {"tool_call_count": 2},
        5: {"tool_call_count": 3},
        6: {"modes": ["line", "single"]},
        7: {"modes": ["rect", "single"]},
        8: {"tool_names": ["generate_zelda_maze"]},
        9: {"tool_names": ["generate_zelda_random"]},
        10: {"tool_names": ["generate_zelda_connect"]},
        11: {"tool_names": ["generate_zelda_bsp"]},
        12: {"tile_types": ["player"]},
        13: {"tile_types": ["key"]},
        14: {"tile_types": ["door"]},
        15: {"tile_types": ["enemy"]},
        20: {"delta_sign": ["player_key", 1]},
        21: {"delta_sign": ["key_door", -1]},
        22: {"delta_exact": {"player_key": 0, "key_door": 0}},
        23: {"large_abs_delta": 100},
        24: {"rationale_contains": "→"},
        25: {"rationale_contains": "\""},
        27: {"acceptance_hint_nonempty": True},
        28: {"modes": ["rect"]},
        29: {"modes": ["line"]},
    }
    for index, required in checks.items():
        cases[index]["checks"] = required
    if len(cases) != 32:
        raise AssertionError("R061 smoke suite must contain exactly 32 cases")
    return cases


def parser_negative_gate() -> dict[str, Any]:
    schema = production_schema()
    valid = {
        "type": "STEP", "rationale": "ok", "plan": "one edit",
        "tool_calls": [{
            "tool_name": "place_tile",
            "parameters": {"mode": "single", "tile_type": "wall", "y": 0, "x": 0},
        }],
        "expected_delta": {"player_key": 0, "key_door": 0},
        "acceptance_hint": "accept if hard-valid",
    }
    fixtures = {
        "valid": (json.dumps(valid), False, True),
        "unclosed": (json.dumps(valid)[:-1], False, False),
        "trailing": (json.dumps(valid) + " trailing", False, False),
        "forced_truncated": (json.dumps(valid), True, False),
        "empty_calls": (json.dumps({**valid, "tool_calls": []}), False, False),
        "oob": (json.dumps({**valid, "tool_calls": [{
            "tool_name": "place_tile",
            "parameters": {"mode": "single", "tile_type": "wall", "y": 16, "x": 0},
        }]}), False, False),
        "missing_delta": (json.dumps({k: v for k, v in valid.items() if k != "expected_delta"}), False, False),
        "extra_top_level": (json.dumps({**valid, "extra": 1}), False, False),
    }
    results = {}
    for name, (text, truncated, expected) in fixtures.items():
        try:
            parsed = json.loads(text.strip())
            if truncated:
                raise ValueError("truncated")
            Draft202012Validator(schema).validate(parsed)
            parse_message(parsed)
            accepted = True
        except Exception:
            accepted = False
        results[name] = accepted == expected
    return {"gate_pass": all(results.values()), "cases": results}


class _BrokenTransportHandler(BaseHTTPRequestHandler):
    retry_requests = 0

    def log_message(self, *_: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/partial":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"choices":[')
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
        else:
            time.sleep(1.0)

    def do_POST(self) -> None:  # noqa: N802
        type(self).retry_requests += 1
        if type(self).retry_requests == 1:
            payload = json.dumps({"error": {"message": "retry witness"}}).encode()
            self.send_response(503)
        else:
            valid = {
                "type": "STEP", "rationale": "retried", "plan": "one edit",
                "tool_calls": [{
                    "tool_name": "place_tile",
                    "parameters": {
                        "mode": "single", "tile_type": "wall", "y": 0, "x": 0,
                    },
                }],
                "expected_delta": {"player_key": 0, "key_door": 0},
                "acceptance_hint": "transport witness",
            }
            payload = json.dumps({
                "id": "r061-retry-witness",
                "object": "chat.completion",
                "created": 0,
                "model": "transport-witness",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(valid)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def transport_negative_gate() -> dict[str, Any]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BrokenTransportHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    results = {}
    for name, timeout in (("partial", 1.0), ("timeout", 0.05)):
        try:
            body = urlopen(f"http://127.0.0.1:{port}/{name}", timeout=timeout).read()
            strict_parse(body.decode())
            results[name] = False
        except Exception:
            results[name] = True
    # Exercise the production OpenAI transport retry path against a real 503,
    # then strict-parse the successful second response.  This is distinct from
    # accepting a partial response: partial/timeout above must still raise.
    try:
        from openai import OpenAI

        _BrokenTransportHandler.retry_requests = 0
        retry_client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="EMPTY",
            max_retries=1,
            timeout=1.0,
        )
        response = retry_client.chat.completions.create(
            model="transport-witness",
            messages=[{"role": "user", "content": "retry witness"}],
            max_tokens=64,
        )
        strict_parse(response.choices[0].message.content or "")
        results["server_503_retry"] = _BrokenTransportHandler.retry_requests == 2
    except Exception:
        results["server_503_retry"] = False
    finally:
        server.shutdown()
        server.server_close()
    return {"gate_pass": all(results.values()), "cases": results}


def _server_model_id(base_url: str, *, expected: str = SERVED_MODEL_NAME) -> str:
    payload = json.loads(urlopen(base_url.removesuffix("/v1") + "/v1/models", timeout=10).read())
    ids = [item["id"] for item in payload.get("data", [])]
    if ids != [expected]:
        raise RuntimeError(f"served model identity mismatch: {ids}")
    return ids[0]


def _gpu_memory() -> list[dict[str, Any]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.strip().splitlines():
        index, name, used, total = [part.strip() for part in line.split(",", 3)]
        rows.append({"index": int(index), "name": name, "used_mib": int(used), "total_mib": int(total)})
    return rows


def _safe_gpu_memory() -> dict[str, Any]:
    try:
        return {"ok": True, "rows": _gpu_memory()}
    except Exception as exc:
        return {
            "ok": False,
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _gpu_runtime() -> dict[str, Any]:
    import torch

    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    return {
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver_versions": sorted(set(item.strip() for item in driver)),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "max_num_seqs": 4,
        "gpu_memory_utilization": 0.90,
    }


def _server_metrics(base_url: str) -> dict[str, Any]:
    """Capture a compact vLLM Prometheus snapshot without making it a gate."""
    url = base_url.removesuffix("/v1") + "/metrics"
    wanted = (
        "prompt_tokens_total",
        "generation_tokens_total",
        "gpu_cache_usage_perc",
        "num_requests_running",
        "num_requests_waiting",
        "request_success_total",
    )
    try:
        body = urlopen(url, timeout=10).read().decode("utf-8")
        samples: dict[str, float] = {}
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.rsplit(None, 1)
            if len(fields) != 2:
                continue
            metric, raw_value = fields
            metric_name = metric.split("{", 1)[0]
            if not any(metric_name.endswith(name) for name in wanted):
                continue
            try:
                samples[metric] = float(raw_value)
            except ValueError:
                continue
        return {"available": True, "samples": samples}
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "samples": {},
        }


def _qualification_checkpoint(
    output: Path,
    *,
    phase: str,
    stage: str,
    records: list[dict[str, Any]],
    **extra: Any,
) -> None:
    _write_json(output, {
        "phase": phase,
        "execution_complete": False,
        "gate_pass": False,
        "last_completed_panel": stage,
        "records": records,
        **extra,
    })


def _semantic_hash(record: dict[str, Any]) -> str | None:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict):
        return None
    return hashlib.sha256(canonical_json_bytes(parsed)).hexdigest()


def _forced_truncation_gate(
    client: Any, messages: list[dict[str, str]], schema: dict[str, Any]
) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=SERVED_MODEL_NAME,
            messages=_visible_messages(messages, schema),
            temperature=0.0,
            seed=61999,
            max_tokens=1,
            extra_body={
                "structured_outputs": {"json": schema},
                "chat_template_kwargs": {"enable_thinking": THINKING_ENABLED},
            },
        )
        choice = response.choices[0]
    except Exception as exc:
        return {
            "gate_pass": False,
            "request_completed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        strict_parse(choice.message.content or "", truncated=True)
    except Exception as exc:
        return {
            "gate_pass": True,
            "request_completed": True,
            "finish_reason": choice.finish_reason,
            "rejection": f"{type(exc).__name__}: {exc}",
        }
    return {
        "gate_pass": False,
        "request_completed": True,
        "finish_reason": choice.finish_reason,
        "error": "truncated response was accepted",
    }


def _case_contract_checks(parsed: dict[str, Any], checks: dict[str, Any]) -> dict[str, bool]:
    calls = parsed["tool_calls"]
    parameters = [call["parameters"] for call in calls]
    results: dict[str, bool] = {}
    if "tool_call_count" in checks:
        results["tool_call_count"] = len(calls) == checks["tool_call_count"]
    if "tool_names" in checks:
        results["tool_names"] = [call["tool_name"] for call in calls] == checks["tool_names"]
    if "modes" in checks:
        results["modes"] = [item.get("mode") for item in parameters] == checks["modes"]
    if "tile_types" in checks:
        results["tile_types"] = [item.get("tile_type") for item in parameters] == checks["tile_types"]
    if "coordinates" in checks:
        results["coordinates"] = [
            [item.get("y"), item.get("x")] for item in parameters
        ] == checks["coordinates"]
    if "delta_sign" in checks:
        key, sign = checks["delta_sign"]
        value = parsed["expected_delta"][key]
        results["delta_sign"] = value > 0 if sign > 0 else value < 0
    if "delta_exact" in checks:
        results["delta_exact"] = parsed["expected_delta"] == checks["delta_exact"]
    if "large_abs_delta" in checks:
        results["large_abs_delta"] = max(
            abs(value) for value in parsed["expected_delta"].values()
        ) >= checks["large_abs_delta"]
    if "rationale_contains" in checks:
        results["rationale_contains"] = checks["rationale_contains"] in parsed["rationale"]
    if checks.get("acceptance_hint_nonempty"):
        results["acceptance_hint_nonempty"] = bool(parsed["acceptance_hint"].strip())
    return results


def _call_case(
    client: Any,
    tokenizer: Any,
    case: dict[str, Any],
    schema: dict[str, Any],
    *,
    service_mode: str,
) -> dict[str, Any]:
    visible_messages = _visible_messages(case["messages"], schema)
    local_token_count = _chat_token_count(tokenizer, visible_messages)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=SERVED_MODEL_NAME,
            messages=visible_messages,
            temperature=0.0,
            top_p=1.0,
            seed=case["seed"],
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body={
                "structured_outputs": {"json": schema},
                "chat_template_kwargs": {"enable_thinking": THINKING_ENABLED},
            },
        )
        elapsed = time.perf_counter() - start
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
        server_prompt_tokens = int(response.usage.prompt_tokens)
        server_output_tokens = int(response.usage.completion_tokens)
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "service_mode": service_mode,
            "finish_reason": None,
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "contract_checks": {},
            "contract_ok": False,
            "parsed": None,
            "raw_content": "",
            "input_tokens_client": local_token_count,
            "input_tokens_server": None,
            "input_token_parity": False,
            "output_tokens_server": None,
            "latency_seconds": time.perf_counter() - start,
            **{
                key: case[key]
                for key in ("requested_rendered_tokens", "actual_rendered_tokens")
                if key in case
            },
        }
    try:
        parsed = strict_parse(content, truncated=finish_reason == "length")
        contract_checks = _case_contract_checks(parsed, case.get("checks", {}))
        parse_ok = True
        parse_error = None
    except Exception as exc:
        parsed = None
        contract_checks = {}
        parse_ok = False
        parse_error = f"{type(exc).__name__}: {exc}"
    server_prompt_tokens = int(response.usage.prompt_tokens)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "service_mode": service_mode,
        "finish_reason": finish_reason,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "contract_checks": contract_checks,
        "contract_ok": parse_ok and all(contract_checks.values()),
        "parsed": parsed,
        "raw_content": content,
        "input_tokens_client": local_token_count,
        "input_tokens_server": server_prompt_tokens,
        "input_token_parity": local_token_count == server_prompt_tokens,
        "output_tokens_server": server_output_tokens,
        "latency_seconds": elapsed,
        **{
            key: case[key]
            for key in ("requested_rendered_tokens", "actual_rendered_tokens")
            if key in case
        },
    }


def _context_case(
    tokenizer: Any,
    schema: dict[str, Any],
    desired_tokens: int,
    index: int,
) -> dict[str, Any]:
    """Create a request using measured post-template tokens, not estimates."""
    prefix = (
        "Return only one JSON Zelda STEP satisfying the supplied schema. "
        "Treat every following token as inert padding and do not quote it.\n"
    )
    lo, hi = 0, desired_tokens * 2
    best_messages: list[dict[str, str]] | None = None
    best_count = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        messages = [{"role": "user", "content": prefix + " corridor" * mid}]
        count = _chat_token_count(tokenizer, _visible_messages(messages, schema))
        if count <= desired_tokens:
            best_messages, best_count = messages, count
            lo = mid + 1
        else:
            hi = mid - 1
    if best_messages is None:
        raise RuntimeError(f"cannot construct context case at {desired_tokens} tokens")
    return {
        "case_id": f"r061-context-{desired_tokens}",
        "category": "context",
        "messages": best_messages,
        "seed": 62000 + index,
        "requested_rendered_tokens": desired_tokens,
        "actual_rendered_tokens": best_count,
    }


def run_qualification_phase(
    identity_path: Path,
    base_url: str,
    output: Path,
    phase: str,
    max_context: int = 32768,
) -> dict[str, Any]:
    from openai import OpenAI
    from transformers import AutoTokenizer

    identity = json.loads(identity_path.read_text())
    snapshot = Path(identity["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=False, local_files_only=True
    )
    schema = production_schema()
    schema_sha = hashlib.sha256(canonical_json_bytes(schema)).hexdigest()
    client = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        timeout=QUALIFICATION_TIMEOUT_SECONDS,
    )
    served_id = _server_model_id(base_url)
    all_cases = smoke_cases()
    if phase == "coldwarm":
        qualification_started = time.perf_counter()
        server_metrics_before = _server_metrics(base_url)
        gpu_memory_snapshots = {"before": _safe_gpu_memory()}
        sequential = all_cases[:24]
        concurrent_cases = all_cases[24:]
        sequential_started = time.perf_counter()
        records = [
            _call_case(
                client, tokenizer, case, schema,
                service_mode="cold" if index == 0 else "warm",
            )
            for index, case in enumerate(sequential)
        ]
        sequential_wall_seconds = time.perf_counter() - sequential_started
        _qualification_checkpoint(
            output, phase=phase, stage="sequential-24", records=records,
            server_model_id=served_id, schema_sha256=schema_sha,
        )
        concurrent_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(
                lambda case: _call_case(
                    client, tokenizer, case, schema, service_mode="concurrent"
                ),
                concurrent_cases,
            ))
        concurrent_wall_seconds = time.perf_counter() - concurrent_started
        gpu_memory_snapshots["after_concurrent"] = _safe_gpu_memory()
        _qualification_checkpoint(
            output, phase=phase, stage="concurrent-8", records=records,
            server_model_id=served_id, schema_sha256=schema_sha,
        )
        anchor_ids = {"r061-smoke-00", "r061-smoke-04", "r061-smoke-16", "r061-smoke-31"}
        anchors = {row["case_id"]: row["raw_content"] for row in records if row["case_id"] in anchor_ids}
        anchor_semantic_hashes = {
            row["case_id"]: _semantic_hash(row)
            for row in records if row["case_id"] in anchor_ids
        }

        # Forced server truncation is a negative case and must not contribute to 32/32.
        trunc_case = deepcopy(all_cases[0])
        truncation_gate = _forced_truncation_gate(
            client, trunc_case["messages"], schema
        )

        # One real retry: an unavailable endpoint must fail, then the explicit
        # second attempt reaches the identity-checked production endpoint.
        first_failed = False
        try:
            bad = OpenAI(base_url="http://127.0.0.1:1/v1", api_key="EMPTY", max_retries=0, timeout=0.2)
            bad.chat.completions.create(
                model=SERVED_MODEL_NAME, messages=all_cases[30]["messages"], max_tokens=8
            )
        except Exception:
            first_failed = True
        retry_record = _call_case(
            client, tokenizer, all_cases[30], schema, service_mode="explicit-retry"
        )
        retry_gate = first_failed and retry_record["parse_ok"]

        context_targets = [
            target for target in (4096, 8192, 16384, 24576)
            if target + MAX_OUTPUT_TOKENS <= max_context
        ]
        context_records = []
        for index, target in enumerate(context_targets):
            try:
                context_case = _context_case(tokenizer, schema, target, index)
                context_record = _call_case(
                    client, tokenizer, context_case, schema, service_mode="context"
                )
            except Exception as exc:
                context_record = {
                    "case_id": f"r061-context-{target}",
                    "category": "context",
                    "service_mode": "context",
                    "finish_reason": None,
                    "parse_ok": False,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                    "contract_checks": {},
                    "contract_ok": False,
                    "parsed": None,
                    "raw_content": "",
                    "input_tokens_client": None,
                    "input_tokens_server": None,
                    "input_token_parity": False,
                    "output_tokens_server": None,
                    "latency_seconds": 0.0,
                    "requested_rendered_tokens": target,
                    "actual_rendered_tokens": None,
                }
            context_records.append(context_record)
            _qualification_checkpoint(
                output, phase=phase, stage=f"context-{target}", records=records,
                server_model_id=served_id, schema_sha256=schema_sha,
                context_records=context_records,
            )
        context_gate = bool(context_records) and all(
            row["parse_ok"]
            and row["finish_reason"] == "stop"
            and row["input_token_parity"]
            and row["input_tokens_client"] == row["actual_rendered_tokens"]
            and row["input_tokens_server"] + row["output_tokens_server"] <= max_context
            for row in context_records
        )

        parser_gate = parser_negative_gate()
        transport_gate = transport_negative_gate()
        gpu_memory_snapshots["after_context"] = _safe_gpu_memory()
        server_metrics_after = _server_metrics(base_url)
        total_output = sum(row["output_tokens_server"] or 0 for row in records)
        total_latency = sum(row["latency_seconds"] for row in records)
        request_panel_wall_seconds = sequential_wall_seconds + concurrent_wall_seconds
        observed_memory_rows = [
            row
            for snapshot in gpu_memory_snapshots.values()
            for row in snapshot["rows"]
        ]
        peak_used_by_index: dict[int, int] = {}
        for row in observed_memory_rows:
            peak_used_by_index[row["index"]] = max(
                peak_used_by_index.get(row["index"], 0), row["used_mib"]
            )
        payload = {
            "phase": phase,
            "execution_complete": True,
            "server_model_id": served_id,
            "schema_sha256": schema_sha,
            "records": records,
            "anchors": anchors,
            "anchor_semantic_hashes": anchor_semantic_hashes,
            "served_max_context": max_context,
            "context_records": context_records,
            "throughput": {
                "aggregate_output_tokens": total_output,
                "aggregate_request_seconds": total_latency,
                "output_tokens_per_summed_request_second": (
                    total_output / total_latency if total_latency else None
                ),
                "request_panel_wall_seconds": request_panel_wall_seconds,
                "output_tokens_per_panel_wall_second": (
                    total_output / request_panel_wall_seconds
                    if request_panel_wall_seconds else None
                ),
                "sequential_wall_seconds": sequential_wall_seconds,
                "concurrent_wall_seconds": concurrent_wall_seconds,
                "qualification_wall_seconds": time.perf_counter() - qualification_started,
                "cold_first_request_seconds": records[0]["latency_seconds"],
                "warm_sequential_median_seconds": sorted(
                    row["latency_seconds"] for row in records[1:24]
                )[len(records[1:24]) // 2],
            },
            "negative_gates": {
                "parser": parser_gate,
                "transport": transport_gate,
                "forced_server_truncation": truncation_gate,
                "forced_server_truncation_fail_closed": truncation_gate["gate_pass"],
                "explicit_retry": retry_gate,
            },
            "gpu_memory": gpu_memory_snapshots["after_context"]["rows"],
            "gpu_memory_snapshots": gpu_memory_snapshots,
            "gpu_memory_peak_observed_mib": peak_used_by_index,
            "gpu_runtime": _gpu_runtime(),
            "server_metrics": {
                "before": server_metrics_before,
                "after": server_metrics_after,
            },
        }
        memory_fit = all(
            snapshot["ok"]
            and all(row["used_mib"] < row["total_mib"] for row in snapshot["rows"])
            for snapshot in gpu_memory_snapshots.values()
        )
        payload["gate_pass"] = (
            len(records) == 32
            and all(row["parse_ok"] for row in records)
            and all(row["contract_ok"] for row in records)
            and all(row["finish_reason"] == "stop" for row in records)
            and all(row["input_token_parity"] for row in records)
            and parser_gate["gate_pass"]
            and transport_gate["gate_pass"]
            and truncation_gate["gate_pass"]
            and retry_gate
            and context_gate
            and memory_fit
        )
    elif phase == "restart":
        prior = json.loads((output.parent / "qualification_coldwarm.json").read_text())
        anchor_ids = list(prior["anchors"])
        chosen = all_cases
        records = [
            _call_case(client, tokenizer, case, schema, service_mode="restart")
            for case in chosen
        ]
        byte_deterministic = all(
            row["raw_content"] == prior["anchors"][row["case_id"]]
            for row in records if row["case_id"] in anchor_ids
        )
        semantic_deterministic = all(
            _semantic_hash(row) == prior["anchor_semantic_hashes"][row["case_id"]]
            for row in records if row["case_id"] in anchor_ids
        )
        payload = {
            "phase": phase,
            "execution_complete": True,
            "server_model_id": served_id,
            "schema_sha256": schema_sha,
            "records": records,
            "semantic_determinism_across_restart": semantic_deterministic,
            "byte_determinism_across_restart_diagnostic": byte_deterministic,
            "gpu_memory": _gpu_memory(),
            "gpu_runtime": _gpu_runtime(),
            "gate_pass": (
                len(records) == 32
                and all(
                    row["parse_ok"]
                    and row["contract_ok"]
                    and row["finish_reason"] == "stop"
                    and row["input_token_parity"]
                    for row in records
                )
                and semantic_deterministic
            ),
        }
    else:
        raise ValueError(f"unknown qualification phase {phase}")
    _write_json(output, payload)
    if not payload["gate_pass"]:
        raise RuntimeError(f"qualification phase {phase} failed")
    return payload


def _state_snapshot(level_ascii: str, metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "players", "keys", "doors", "enemies", "regions",
        "player_key", "key_door", "solution_length",
    )
    return {
        "level_ascii": level_ascii,
        "sha256": _sha256_text(level_ascii),
        "metrics": {key: int(metrics.get(key, 0)) for key in keys},
    }


def run_real_episode(
    identity_path: Path,
    subset_path: Path,
    base_url: str,
    output: Path,
    *,
    max_steps: int = 2,
) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text())
    subset = json.loads(subset_path.read_text())
    case = subset["cases"][0]
    config = _exact_config(
        case["player_key"], case["key_door"], model=SERVED_MODEL_NAME,
        base_url=base_url, seed=int(case["seed"]),
    )
    optimizer = GreedyOptimizer(config, verbose=False)
    optimizer.initialize(seed=case["seed"])
    initial_ascii = optimizer.current_level.to_string()
    start = time.perf_counter()
    optimizer.run(max_steps=max_steps)
    elapsed_ms = int(round((time.perf_counter() - start) * 1000))

    actions = []
    parser_errors = 0
    truncated_responses = 0
    tool_errors = 0
    for step in optimizer.step_records:
        truncated_responses += int(bool(step.truncated))
        if step.message_type in {"PARSE_ERROR", "ERROR"}:
            parser_errors += 1
        if step.message_type != "STEP":
            continue
        failed_tools = sum(
            not item.get("result", {}).get("success", False)
            for item in step.tool_results if item.get("tool_name") != "_truncated"
        )
        tool_errors += failed_tools
        before = _state_snapshot(step.level_before, step.metrics_before)
        if step.accepted:
            after_ascii, after_metrics = step.level_after, step.metrics_after
        else:
            after_ascii, after_metrics = step.level_before, step.metrics_before
        after = _state_snapshot(after_ascii, after_metrics)
        actual = {
            key: after["metrics"][key] - before["metrics"][key]
            for key in ("player_key", "key_door")
        }
        expected = step.parsed_message.get("expected_delta", {})
        if set(expected) != {"player_key", "key_door"}:
            raise RuntimeError("production STEP omitted required expected_delta")
        actions.append({
            "index": len(actions),
            "semantic_action": step.parsed_message,
            "state_before": before,
            "state_after": after,
            "expected_delta": {key: int(expected[key]) for key in expected},
            "actual_delta": actual,
            "accepted": bool(step.accepted),
            "decision_reason": step.accept_reason or "not accepted",
            "primitive_edits": int(step.num_tiles_changed),
            "evaluator_calls": 1,
            "verifier_calls": 0,
            "latency_ms": float(step.elapsed_time * 1000),
        })

    final_metrics = _state_snapshot(
        optimizer.current_level.to_string(), optimizer.current_result.metrics
    )["metrics"]
    if truncated_responses and parser_errors < truncated_responses:
        raise RuntimeError("a truncated production response escaped protocol failure")
    target = {
        "player_key": int(case["player_key"]),
        "key_door": int(case["key_door"]),
        "enemies": 3,
    }
    hard = {
        "one_player": final_metrics["players"] == 1,
        "one_key": final_metrics["keys"] == 1,
        "one_door": final_metrics["doors"] == 1,
        "enemy_count": final_metrics["enemies"] == 3,
        "regions_one": final_metrics["regions"] == 1,
        "player_key_reachable": final_metrics["player_key"] > 0,
        "key_door_reachable": final_metrics["key_door"] > 0,
        "protocol_clean": parser_errors == 0,
        "tool_clean": tool_errors == 0,
    }
    hard["hard_valid"] = all(hard.values())
    e_pk = abs(final_metrics["player_key"] - target["player_key"])
    e_kd = abs(final_metrics["key_door"] - target["key_door"])
    exact = hard["hard_valid"] and max(e_pk, e_kd) == 0
    costs = {
        "parser_errors": parser_errors,
        "tool_errors": tool_errors,
        "llm_calls": len(optimizer.step_records),
        "input_tokens": sum(step.input_tokens for step in optimizer.step_records),
        "output_tokens": sum(step.output_tokens for step in optimizer.step_records),
        "latency_ms": elapsed_ms,
        "retries": int(optimizer.llm_retry_count),
        "semantic_actions": len(actions),
        "primitive_edits": sum(action["primitive_edits"] for action in actions),
        "evaluator_calls": 1 + len(actions),
        "verifier_calls": 0,
    }
    detected = []
    if parser_errors:
        detected.append("protocol")
    if not all(hard[name] for name in (
        "one_player", "one_key", "one_door", "enemy_count", "regions_one",
        "player_key_reachable", "key_door_reachable",
    )):
        detected.append("hard-validity")
    if tool_errors:
        detected.append("tool-effect")
    seen = {_sha256_text(initial_ascii)}
    oscillation = False
    for action in actions:
        if action["accepted"] and action["state_after"]["sha256"] in seen:
            oscillation = True
        seen.add(action["state_after"]["sha256"])
    if oscillation:
        detected.append("oscillation")
    total_tokens = costs["input_tokens"] + costs["output_tokens"]
    if (
        len(actions) >= MAX_SEMANTIC_ACTIONS
        or costs["verifier_calls"] >= MAX_VERIFIER_CALLS
        or total_tokens >= MAX_SEMANTIC_ACTIONS * MAX_OUTPUT_TOKENS
    ):
        detected.append("budget")
    if not exact and not detected:
        detected.append("planning" if actions else "target-miss")
    errors = [{
        "taxonomy": taxonomy,
        "code": f"R061_{taxonomy.upper().replace('-', '_')}",
        "stage": "real-episode-smoke",
        "message": f"Derived {taxonomy} failure",
    } for taxonomy in detected]
    primary = next((name for name in FAILURE_TAXONOMY if name in detected), None)
    component = lambda version: {"version": version, "sha256": _sha256_text(version)}
    record = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "run_id": "R061-END-TO-END",
        "episode_id": f"r061-{case['target_id']}",
        "target": target,
        "split": "dev",
        "manifest_sha256": subset["source_manifest_sha256"],
        "model": {
            "id": identity["model_id"], "revision": identity["revision"],
            "runtime": f"vllm-{_version('vllm')}", "quantization": None,
            "chat_template": {
                "version": "hf-pinned", "sha256": identity["tokenizer"]["chat_template_sha256"],
            },
            "sampling": {
                "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                "min_p": 0.0, "seed": case["seed"],
                "max_tokens": MAX_OUTPUT_TOKENS, "stop_strings": [],
                "stop_token_ids": identity["tokenizer"]["special_token_ids"],
                "enable_thinking": THINKING_ENABLED,
                "grammar_backend": "vllm-json-schema",
            },
        },
        "harness": {
            "prompt": component(f"corrected-minimal@{HARNESS_COMMIT}"),
            "tool_schema": {"version": "r061-step-json-v1", "sha256": hashlib.sha256(canonical_json_bytes(production_schema())).hexdigest()},
            "feedback": component("m0.1-exact-feedback"),
            "memory": component("none"),
            "controller": component("m0.1-hard-valid-greedy"),
        },
        "initialization": {"sha256": _sha256_text(initial_ascii), "seed": case["seed"]},
        "actions": actions,
        "final": {
            "hard_constraints": hard, "metrics": final_metrics,
            "e_inf": max(e_pk, e_kd), "e_sum": e_pk + e_kd, "exact": exact,
        },
        "primary_failure": primary,
        "errors": errors,
        "budget": {
            "semantic_actions": MAX_SEMANTIC_ACTIONS,
            "verifier_calls": MAX_VERIFIER_CALLS,
            "total_tokens": MAX_SEMANTIC_ACTIONS * MAX_OUTPUT_TOKENS,
        },
        "costs": costs,
    }
    validate_episode(record)
    _write_json(output, record)
    return record


def _hard_validity(metrics: dict[str, Any], *, parser_errors: int, tool_errors: int) -> dict[str, bool]:
    checks = {
        "one_player": int(metrics.get("players", 0)) == 1,
        "one_key": int(metrics.get("keys", 0)) == 1,
        "one_door": int(metrics.get("doors", 0)) == 1,
        "enemy_count": int(metrics.get("enemies", 0)) == 3,
        "regions_one": int(metrics.get("regions", 0)) == 1,
        "player_key_reachable": int(metrics.get("player_key", 0)) > 0,
        "key_door_reachable": int(metrics.get("key_door", 0)) > 0,
        "protocol_clean": parser_errors == 0,
        "tool_clean": tool_errors == 0,
    }
    checks["hard_valid"] = all(checks.values())
    return checks


def _reasoning_action_traces(optimizer: GreedyOptimizer) -> list[dict[str, Any]]:
    """Keep only the evidence needed to diagnose reasoning -> action -> effect."""
    traces = []
    target_player_key = int(optimizer.config.targets.player_key.value)
    target_key_door = int(optimizer.config.targets.key_door.value)
    for step in optimizer.step_records:
        before = {
            key: int(step.metrics_before.get(key, 0))
            for key in ("player_key", "key_door", "regions")
        }
        observed_after = {
            key: int(step.metrics_after.get(key, before[key]))
            for key in before
        }
        expected = (step.parsed_message or {}).get("expected_delta", {})
        step_tool_errors = sum(
            not item.get("result", {}).get("success", False)
            for item in step.tool_results
            if item.get("tool_name") != "_truncated"
        )
        candidate_hard = _hard_validity(
            step.metrics_after,
            parser_errors=int(step.message_type != "STEP"),
            tool_errors=step_tool_errors,
        )
        candidate_exact = bool(
            candidate_hard["hard_valid"]
            and observed_after["player_key"] == target_player_key
            and observed_after["key_door"] == target_key_door
        )
        traces.append({
            "step_num": int(step.step_num),
            "message_type": step.message_type,
            "thinking_enabled": getattr(step, "thinking_enabled", None),
            "reasoning_handoff_chars": getattr(step, "reasoning_handoff_chars", 0),
            "reasoning_content": step.reasoning_content,
            "final_json": step.parsed_message,
            "expected_delta": expected,
            "observed_delta": {
                key: observed_after[key] - before[key]
                for key in ("player_key", "key_door")
            },
            "candidate_metrics": observed_after,
            "candidate_hard_valid": candidate_hard["hard_valid"],
            "candidate_exact": candidate_exact,
            "accepted": bool(step.accepted),
            "accept_reason": step.accept_reason,
            "transaction_tiles_changed": int(
                getattr(step, "transaction_tiles_changed", step.num_tiles_changed)
            ),
            "net_tiles_changed": int(
                getattr(step, "net_tiles_changed", step.num_tiles_changed)
            ),
            "truncated": bool(step.truncated),
            "input_tokens": int(step.input_tokens),
            "output_tokens": int(step.output_tokens),
        })
    return traces


def _thinking_enabled(config: Any) -> bool:
    """Read the effective per-request thinking flag from a probe config."""
    extra_body = config.llm.extra_body or {}
    template_kwargs = extra_body.get("chat_template_kwargs") or {}
    return bool(template_kwargs.get("enable_thinking", False))


def _set_thinking_enabled(config: Any, enabled: bool) -> None:
    """Change only the next request's Qwen chat-template thinking flag."""
    if config.llm.extra_body is None:
        config.llm.extra_body = {}
    template_kwargs = config.llm.extra_body.setdefault("chat_template_kwargs", {})
    template_kwargs["enable_thinking"] = bool(enabled)


def _truncation_handoff_instruction(
    reasoning_content: str,
    *,
    base_instruction: str | None,
    prefer_latest_correction: bool = False,
) -> str:
    """Ask the non-thinking fallback to serialize the plan already computed."""
    correction_rule = ""
    if prefer_latest_correction:
        correction_rule = (
            " Later self-corrections override earlier candidates: discard every "
            "plan before the last explicit correction (for example, 'Wait!', "
            "'invalid', or 'shortcut') and reconstruct the action only from the "
            "final corrected geometry, even if its call list is unfinished."
        )
    handoff = (
        "Your previous native reasoning hit the completion limit before the "
        "required JSON action. Do not restart the analysis or invent a new plan. "
        "Immediately serialize the best complete construction from that draft as "
        "one valid STEP. Keep rationale and plan to one short sentence each, use "
        "at most 20 tool calls, and preserve the draft's coordinates. If the final "
        "sentence is incomplete, use the last complete plan above it."
        f"{correction_rule}\n\n"
        "<previous_reasoning_draft>\n"
        f"{reasoning_content}\n"
        "</previous_reasoning_draft>"
    )
    return f"{base_instruction}\n\n{handoff}" if base_instruction else handoff


def _run_probe_optimizer(
    optimizer: GreedyOptimizer,
    *,
    max_steps: int,
    truncation_recovery: str,
    thinking_max_tokens: int | None = None,
    fallback_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Run a prompt probe, optionally falling back once after a truncation.

    The recovery is deliberately narrow: a thinking-on response that consumes
    the whole completion budget is followed by a thinking-off request against
    the unchanged level.  The handoff variant also exposes the already-computed
    reasoning to that one fallback request.  A second truncation or any other
    protocol error stops the case instead of repeating a deterministic failure.
    """
    if truncation_recovery not in {
        "none", "thinking_off_once", "thinking_off_handoff",
        "thinking_off_handoff_latest",
    }:
        raise ValueError(f"unknown truncation recovery: {truncation_recovery}")

    if thinking_max_tokens is not None and _thinking_enabled(optimizer.config):
        optimizer.config.llm.max_tokens = thinking_max_tokens

    if truncation_recovery == "none":
        initial_thinking = _thinking_enabled(optimizer.config)
        optimizer.run(max_steps=max_steps)
        for step in optimizer.step_records:
            step.thinking_enabled = initial_thinking
        return {
            "policy": "none",
            "attempted": False,
            "trigger_step": None,
            "fallback_calls": 0,
            "recovered_truncations": 0,
        }

    trigger_step: int | None = None
    handoff_chars = 0
    handoff_pending = False
    base_instruction = optimizer.config.optimizer.extra_instruction
    while optimizer.step_count < max_steps:
        if optimizer.termination.should_terminate(optimizer.current_result):
            break

        request_thinking = _thinking_enabled(optimizer.config)
        step = optimizer.step()
        step.thinking_enabled = request_thinking
        step.reasoning_handoff_chars = handoff_chars if handoff_pending else 0
        if handoff_pending:
            optimizer.config.optimizer.extra_instruction = base_instruction
            handoff_pending = False

        if step.truncated:
            if trigger_step is None and request_thinking:
                trigger_step = int(step.step_num)
                _set_thinking_enabled(optimizer.config, False)
                if fallback_max_tokens is not None:
                    optimizer.config.llm.max_tokens = fallback_max_tokens
                if truncation_recovery in {
                    "thinking_off_handoff", "thinking_off_handoff_latest",
                }:
                    handoff_chars = len(step.reasoning_content)
                    optimizer.config.optimizer.extra_instruction = (
                        _truncation_handoff_instruction(
                            step.reasoning_content,
                            base_instruction=base_instruction,
                            prefer_latest_correction=(
                                truncation_recovery == "thinking_off_handoff_latest"
                            ),
                        )
                    )
                    handoff_pending = True
                continue
            break
        if step.message_type in {"ERROR", "PARSE_ERROR", "STOP"}:
            break

    fallback_steps = [
        step for step in optimizer.step_records
        if trigger_step is not None and step.step_num > trigger_step
    ]
    recovered = bool(
        trigger_step is not None
        and any(step.message_type in {"STEP", "STOP"} for step in fallback_steps)
    )
    return {
        "policy": truncation_recovery,
        "attempted": trigger_step is not None,
        "trigger_step": trigger_step,
        "fallback_calls": len(fallback_steps),
        "recovered_truncations": int(recovered),
        "reasoning_handoff_chars": handoff_chars,
    }


def _matched_arm_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    e_infs = sorted(int(row["e_inf"]) for row in records)
    p95_index = max(0, (95 * len(e_infs) + 99) // 100 - 1) if e_infs else 0
    return {
        "cases": len(records),
        "completed_cases": sum(row["status"] == "complete" for row in records),
        "hard_valid": sum(row["hard_constraints"]["hard_valid"] for row in records),
        "exact": sum(row["exact"] for row in records),
        "candidate_exact_ever": sum(
            bool(row.get("candidate_exact_ever")) for row in records
        ),
        "exact_candidate_not_final": sum(
            bool(row.get("exact_candidate_not_final")) for row in records
        ),
        "exact_rate": sum(row["exact"] for row in records) / len(records) if records else 0.0,
        "e_inf_mean": sum(e_infs) / len(e_infs) if e_infs else None,
        "e_inf_p95": e_infs[p95_index] if e_infs else None,
        "e_inf_worst": max(e_infs) if e_infs else None,
        "parser_errors": sum(row["parser_errors"] for row in records),
        "unrecovered_parser_errors": sum(
            row.get("unrecovered_parser_errors", row["parser_errors"])
            for row in records
        ),
        "recovery_attempts": sum(
            bool(row.get("protocol_recovery", {}).get("attempted"))
            for row in records
        ),
        "recovered_truncations": sum(
            int(row.get("protocol_recovery", {}).get("recovered_truncations", 0))
            for row in records
        ),
        "transport_errors": sum(row["transport_errors"] for row in records),
        "tool_errors": sum(row["tool_errors"] for row in records),
        "truncated_responses": sum(row["truncated_responses"] for row in records),
        "llm_calls": sum(row["costs"]["llm_calls"] for row in records),
        "input_tokens": sum(row["costs"]["input_tokens"] for row in records),
        "output_tokens": sum(row["costs"]["output_tokens"] for row in records),
        "latency_ms": sum(row["costs"]["latency_ms"] for row in records),
        "retries": sum(row["costs"]["retries"] for row in records),
    }


def run_matched_arm(
    subset_path: Path,
    base_url: str,
    output: Path,
    *,
    arm: str,
    model_id: str,
    revision: str,
    case_limit: int | None = None,
    max_steps: int = MAX_SEMANTIC_ACTIONS,
    extra_instruction: str | None = None,
    conversation_length: int = 1,
    target_ids: list[str] | None = None,
    truncation_recovery: str = "none",
    max_tokens: int = MAX_OUTPUT_TOKENS,
    thinking_max_tokens: int | None = None,
    temperature: float = 0.0,
    request_seed_offset: int = 0,
    ablation_mode: str = "full",
    commit_policy: str = "mode_default",
    evaluator_exact_stop: str = "mode_default",
    optimizer_factory: Callable[[Any], GreedyOptimizer] | None = None,
    record_enricher: Callable[[GreedyOptimizer, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if ablation_mode not in {
        "full",
        "no_historical_feedback",
        "no_metric_observation",
        "no_evaluation_loop",
    }:
        raise ValueError(f"unsupported ablation_mode: {ablation_mode}")
    if commit_policy not in {"mode_default", "force_apply", "verified"}:
        raise ValueError(f"unsupported commit_policy: {commit_policy}")
    if evaluator_exact_stop not in {"mode_default", "on", "off"}:
        raise ValueError(
            f"unsupported evaluator_exact_stop: {evaluator_exact_stop}"
        )
    if max_tokens <= 0 or max_tokens > MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"max_tokens must be in [1, {MAX_OUTPUT_TOKENS}], got {max_tokens}"
        )
    if thinking_max_tokens is not None and not 1 <= thinking_max_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError(
            "thinking_max_tokens must be in "
            f"[1, {MAX_OUTPUT_TOKENS}], got {thinking_max_tokens}"
        )
    if temperature < 0:
        raise ValueError(f"temperature must be nonnegative, got {temperature}")
    subset = json.loads(subset_path.read_text())
    subset_without_hash = {key: value for key, value in subset.items() if key != "subset_sha256"}
    actual_subset_sha = hashlib.sha256(canonical_json_bytes(subset_without_hash)).hexdigest()
    if actual_subset_sha != subset["subset_sha256"]:
        raise RuntimeError("matched subset content hash mismatch")
    if subset["budgets"] != {
        "semantic_actions": MAX_SEMANTIC_ACTIONS,
        "verifier_calls": MAX_VERIFIER_CALLS,
        "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
        "max_tool_calls_per_action": 20,
    }:
        raise RuntimeError("matched subset budget drift")
    served_model_id = _server_model_id(base_url, expected=model_id)

    if target_ids:
        by_id = {case["target_id"]: case for case in subset["cases"]}
        missing = [target_id for target_id in target_ids if target_id not in by_id]
        if missing:
            raise ValueError(f"unknown target IDs: {missing}")
        selected_cases = [by_id[target_id] for target_id in target_ids]
    else:
        selected_cases = subset["cases"][:case_limit] if case_limit else subset["cases"]
    effective_budgets = dict(subset["budgets"])
    effective_budgets["semantic_actions"] = max_steps
    effective_budgets["max_output_tokens_per_call"] = max(
        max_tokens, thinking_max_tokens or 0
    )
    records: list[dict[str, Any]] = []
    for case in selected_cases:
        request_seed = int(case["seed"]) + int(request_seed_offset)
        config = _exact_config(
            case["player_key"],
            case["key_door"],
            model=model_id,
            base_url=base_url,
            seed=request_seed,
            extra_instruction=extra_instruction,
            conversation_length=conversation_length,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        config.optimizer.ablation_mode = ablation_mode
        config.optimizer.commit_policy = commit_policy
        config.optimizer.evaluator_exact_stop = evaluator_exact_stop
        optimizer = (
            optimizer_factory(config)
            if optimizer_factory is not None
            else GreedyOptimizer(config, verbose=False)
        )
        optimizer.initialize(seed=int(case["seed"]))
        initial_ascii = optimizer.current_level.to_string()
        initialization_sha = _sha256_text(initial_ascii)
        if initialization_sha != case["initialization_sha256"]:
            raise RuntimeError(f"initialization drift for {case['target_id']}")
        started = time.perf_counter()
        protocol_recovery = _run_probe_optimizer(
            optimizer,
            max_steps=max_steps,
            truncation_recovery=truncation_recovery,
            thinking_max_tokens=thinking_max_tokens,
            fallback_max_tokens=(max_tokens if thinking_max_tokens is not None else None),
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        transport_errors = sum(
            step.message_type == "ERROR" for step in optimizer.step_records
        )
        parser_errors = sum(
            step.message_type in {"PARSE_ERROR", "ERROR"} for step in optimizer.step_records
        )
        truncated = sum(bool(step.truncated) for step in optimizer.step_records)
        unrecovered_parser_errors = max(
            0,
            parser_errors - int(protocol_recovery["recovered_truncations"]),
        )
        tool_errors = sum(
            not item.get("result", {}).get("success", False)
            for step in optimizer.step_records
            for item in step.tool_results
            if item.get("tool_name") != "_truncated"
        )
        final_metrics = _state_snapshot(
            optimizer.current_level.to_string(), optimizer.current_result.metrics
        )["metrics"]
        hard = _hard_validity(
            final_metrics,
            parser_errors=unrecovered_parser_errors,
            tool_errors=tool_errors,
        )
        e_pk = abs(final_metrics.get("player_key", 0) - int(case["player_key"]))
        e_kd = abs(final_metrics.get("key_door", 0) - int(case["key_door"]))
        step_traces = _reasoning_action_traces(optimizer)
        candidate_exact_steps = [
            int(step["step_num"])
            for step in step_traces
            if step["candidate_exact"]
        ]
        final_exact = hard["hard_valid"] and max(e_pk, e_kd) == 0
        record = {
            "status": "complete",
            "arm": arm,
            "model_id": model_id,
            "revision": revision,
            "target_id": case["target_id"],
            "target": {
                "player_key": int(case["player_key"]),
                "key_door": int(case["key_door"]),
                "enemies": 3,
            },
            "seed": int(case["seed"]),
            "request_seed": request_seed,
            "ablation_mode": ablation_mode,
            "commit_policy": optimizer.commit_policy,
            "evaluator_exact_stop": optimizer.evaluator_exact_stop,
            "initialization_sha256": initialization_sha,
            "final_level_sha256": _sha256_text(optimizer.current_level.to_string()),
            "final_metrics": final_metrics,
            "hard_constraints": hard,
            "e_inf": max(e_pk, e_kd),
            "e_sum": e_pk + e_kd,
            "exact": final_exact,
            "candidate_exact_ever": bool(candidate_exact_steps),
            "candidate_exact_steps": candidate_exact_steps,
            "first_candidate_exact_step": (
                candidate_exact_steps[0] if candidate_exact_steps else None
            ),
            "exact_candidate_not_final": bool(candidate_exact_steps) and not final_exact,
            "parser_errors": parser_errors,
            "unrecovered_parser_errors": unrecovered_parser_errors,
            "transport_errors": transport_errors,
            "tool_errors": tool_errors,
            "truncated_responses": truncated,
            "protocol_recovery": protocol_recovery,
            "steps": step_traces,
            "costs": {
                "llm_calls": len(optimizer.step_records),
                "input_tokens": sum(step.input_tokens for step in optimizer.step_records),
                "output_tokens": sum(step.output_tokens for step in optimizer.step_records),
                "latency_ms": elapsed_ms,
                "retries": int(optimizer.llm_retry_count),
                "semantic_actions": sum(
                    step.message_type == "STEP" for step in optimizer.step_records
                ),
                "evaluator_calls": 1 + sum(
                    step.message_type == "STEP" for step in optimizer.step_records
                ),
                "verifier_calls": 0,
            },
        }
        if record_enricher is not None:
            enrichment = dict(record_enricher(optimizer, case))
            record["experiment_enrichment"] = enrichment
        records.append(record)
        partial = {
            "execution_complete": False,
            "arm": arm,
            "ablation_mode": ablation_mode,
            "commit_policy": commit_policy,
            "evaluator_exact_stop": evaluator_exact_stop,
            "model_id": model_id,
            "revision": revision,
            "server_model_id": served_model_id,
            "subset_sha256": subset["subset_sha256"],
            "budgets": effective_budgets,
            "records": records,
        }
        _write_json(output, partial)

    aggregate = _matched_arm_aggregate(records)
    case_contract = [
        [row["target_id"], row["seed"], row["initialization_sha256"]] for row in records
    ]
    payload = {
        "execution_complete": len(records) == len(selected_cases),
        "contract_gate_pass": (
            len(records) == len(selected_cases)
            and all(row["transport_errors"] == 0 for row in records)
            and all(row["costs"]["semantic_actions"] <= max_steps for row in records)
            and all(row["costs"]["verifier_calls"] <= MAX_VERIFIER_CALLS for row in records)
            and all(
                row["costs"]["output_tokens"]
                <= max_steps * max(max_tokens, thinking_max_tokens or 0)
                for row in records
            )
        ),
        "arm": arm,
        "ablation_mode": ablation_mode,
        "commit_policy": commit_policy,
        "evaluator_exact_stop": evaluator_exact_stop,
        "model_id": model_id,
        "revision": revision,
        "server_model_id": served_model_id,
        "subset_sha256": subset["subset_sha256"],
        "case_contract_sha256": hashlib.sha256(canonical_json_bytes(case_contract)).hexdigest(),
        "budgets": effective_budgets,
        "sampling": {
            "temperature": temperature,
            "top_p": 1.0,
            "request_seed_offset": int(request_seed_offset),
            "enable_thinking": THINKING_ENABLED,
            "grammar_backend": "vllm-json-schema",
        },
        "probe": {
            "case_limit": case_limit,
            "max_steps": max_steps,
            "conversation_length": conversation_length,
            "extra_instruction_sha256": (
                _sha256_text(extra_instruction) if extra_instruction else None
            ),
            "target_ids": [case["target_id"] for case in selected_cases],
            "truncation_recovery": truncation_recovery,
            "max_tokens": max_tokens,
            "thinking_max_tokens": thinking_max_tokens,
            "ablation_mode": ablation_mode,
            "commit_policy": commit_policy,
            "evaluator_exact_stop": evaluator_exact_stop,
        },
        "aggregate": aggregate,
        "records": records,
    }
    _write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError(f"matched arm contract failed: {arm}")
    return payload


def finalize_matched_arms(
    incumbent_path: Path,
    qwen38_path: Path,
    output: Path,
) -> dict[str, Any]:
    incumbent = json.loads(incumbent_path.read_text())
    qwen38 = json.loads(qwen38_path.read_text())
    component_gates = {
        "incumbent_complete": incumbent.get("contract_gate_pass") is True,
        "qwen38_complete": qwen38.get("contract_gate_pass") is True,
        "same_subset": incumbent.get("subset_sha256") == qwen38.get("subset_sha256"),
        "same_cases_seeds_initialization": (
            incumbent.get("case_contract_sha256") == qwen38.get("case_contract_sha256")
        ),
        "same_budgets": incumbent.get("budgets") == qwen38.get("budgets"),
        "same_sampling_contract": incumbent.get("sampling") == qwen38.get("sampling"),
    }
    payload = {
        "matched_contract_gate_pass": all(component_gates.values()),
        "component_gates": component_gates,
        "claim_scope": "matched model-upgrade comparison" if all(component_gates.values()) else "unmatched descriptive baselines only",
        "incumbent": {
            "model_id": incumbent["model_id"],
            "revision": incumbent["revision"],
            "aggregate": incumbent["aggregate"],
        },
        "qwen38": {
            "model_id": qwen38["model_id"],
            "revision": qwen38["revision"],
            "aggregate": qwen38["aggregate"],
        },
    }
    _write_json(output, payload)
    if not payload["matched_contract_gate_pass"]:
        raise RuntimeError(f"R061-D matched contract failed: {component_gates}")
    return payload








def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Zelda harness experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("prompt-probe")
    probe.add_argument("--subset", type=Path, required=True)
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--arm", required=True)
    probe.add_argument("--model-id", required=True)
    probe.add_argument("--revision", required=True)
    probe.add_argument("--case-limit", type=int, default=12)
    probe.add_argument("--max-steps", type=int, default=6)
    probe.add_argument("--conversation-length", type=int, default=2)
    probe.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    probe.add_argument("--thinking-max-tokens", type=int)
    probe.add_argument("--temperature", type=float, default=0.0)
    probe.add_argument("--request-seed-offset", type=int, default=0)
    probe.add_argument(
        "--ablation-mode",
        choices=(
            "full",
            "no_historical_feedback",
            "no_metric_observation",
            "no_evaluation_loop",
        ),
        default="full",
    )
    probe.add_argument(
        "--commit-policy",
        choices=("mode_default", "force_apply", "verified"),
        default="mode_default",
    )
    probe.add_argument(
        "--evaluator-exact-stop",
        choices=("mode_default", "on", "off"),
        default="mode_default",
    )
    probe.add_argument("--extra-instruction-file", type=Path)
    probe.add_argument("--target-id", action="append", dest="target_ids")
    probe.add_argument(
        "--truncation-recovery",
        choices=(
            "none", "thinking_off_once", "thinking_off_handoff",
            "thinking_off_handoff_latest",
        ),
        default="none",
    )
    args = parser.parse_args()
    extra_instruction = None
    if args.extra_instruction_file is not None:
        extra_instruction = args.extra_instruction_file.read_text().strip()
        if not extra_instruction:
            raise ValueError("extra-instruction-file is empty")
    result = run_matched_arm(
        args.subset,
        args.base_url,
        args.output,
        arm=args.arm,
        model_id=args.model_id,
        revision=args.revision,
        case_limit=args.case_limit,
        max_steps=args.max_steps,
        extra_instruction=extra_instruction,
        conversation_length=args.conversation_length,
        target_ids=args.target_ids,
        truncation_recovery=args.truncation_recovery,
        max_tokens=args.max_tokens,
        thinking_max_tokens=args.thinking_max_tokens,
        temperature=args.temperature,
        request_seed_offset=args.request_seed_offset,
        ablation_mode=args.ablation_mode,
        commit_policy=args.commit_policy,
        evaluator_exact_stop=args.evaluator_exact_stop,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

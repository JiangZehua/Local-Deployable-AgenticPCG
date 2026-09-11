"""R074: minimal Zelda-harness transfer probe on Binary Door.

This keeps the Qwen3.8 reasoning/truncation/grammar/greedy controller contract
and changes only the domain prompt, tools, evaluator, and exact target metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiments import r061_qwen38_m1 as base
from toolusepcg.protocol.grammar import build_step_grammar_pack
from toolusepcg.config import load_config
from toolusepcg.evaluator import EvalResult
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.optimizer.termination import TerminationChecker


TARGETS = (32, 40, 48, 56, 64, 72, 80, 88)
SEED = 42
MAX_STEPS = 4
MAX_TOOL_CALLS = 20
NATIVE_MAX_TOKENS = 6144
FALLBACK_MAX_TOKENS = 8192
LEGACY_ACCEPTANCE = "legacy_score"
TARGET_FIRST_ACCEPTANCE = "target_lexicographic"
ACCEPTANCE_POLICIES = (LEGACY_ACCEPTANCE, TARGET_FIRST_ACCEPTANCE)


class BinaryDoorTargetTermination(TerminationChecker):
    """Stop an opt-in Binary Door run once its exact valid target is reached."""

    def _exact_target_reached(self, result: EvalResult | None) -> bool:
        if result is None or result.problem_type != "binarydoor":
            return super()._exact_target_reached(result)
        path_target = self.config.targets.door_path
        region_target = self.config.targets.num_connected_regions
        if (
            path_target is None
            or path_target.mode != "target"
            or path_target.value is None
        ):
            return False
        expected_regions = (
            int(region_target.value)
            if region_target is not None and region_target.value is not None
            else 1
        )
        path = int(result.metrics.get("door_path", 0))
        regions = int(result.metrics.get("num_connected_regions", 0))
        return path > 0 and path == int(path_target.value) and regions == expected_regions


class TargetFirstBinaryDoorOptimizer(GreedyOptimizer):
    """Opt-in controller whose primary target quality dominates edit cost.

    The legacy scalar scorer remains the tie-breaker when the candidate and
    incumbent have identical hard-violation count and target error.  This
    keeps change penalties useful against same-quality rewrites without
    letting them reject progress toward the requested metric.
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.termination = BinaryDoorTargetTermination(config)

    def _primary_quality(self, result: EvalResult) -> tuple[int, int]:
        path = int(result.metrics.get("door_path", 0))
        regions = int(result.metrics.get("num_connected_regions", 0))
        target = int(self.config.targets.door_path.value)
        region_target = int(self.config.targets.num_connected_regions.value)
        hard_violations = int(path <= 0) + int(regions != region_target)
        return hard_violations, abs(path - target)

    def decide_accept(
        self,
        new_result: EvalResult,
        num_changes: int,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        if num_changes == 0:
            return super().decide_accept(new_result, num_changes, **kwargs)
        before = self._primary_quality(self.current_result)
        after = self._primary_quality(new_result)
        if after < before:
            return True, f"accepted_target_primary ({before} -> {after})"
        if after > before:
            return False, f"rejected_target_primary ({before} -> {after})"
        return super().decide_accept(new_result, num_changes, **kwargs)


def build_optimizer(
    config: Any,
    *,
    acceptance_policy: str = LEGACY_ACCEPTANCE,
) -> GreedyOptimizer:
    """Build the legacy controller unless the new policy is explicitly selected."""
    if acceptance_policy == LEGACY_ACCEPTANCE:
        return GreedyOptimizer(config, verbose=False)
    if acceptance_policy == TARGET_FIRST_ACCEPTANCE:
        return TargetFirstBinaryDoorOptimizer(config, verbose=False)
    raise ValueError(f"unknown acceptance policy: {acceptance_policy}")


def exact_config(
    target: int,
    *,
    model: str = base.MODEL_ID,
    base_url: str = "http://127.0.0.1:8145/v1",
    extra_instruction: str | None = None,
    max_tokens: int = FALLBACK_MAX_TOKENS,
) -> Any:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target}")
    return load_config(
        "configs/binarydoor.yaml",
        {
            "targets": {
                "door_path": {"mode": "target", "value": target},
                "num_connected_regions": {"mode": "target", "value": 1},
            },
            "optimizer": {
                "init_strategy": "empty",
                "edit_tools": ["place_tile"],
                "conversation_length": 2,
                "max_tool_calls_per_step": MAX_TOOL_CALLS,
                "extra_instruction": extra_instruction,
            },
            "llm": {
                "provider": "openai",
                "model": model,
                "base_url": base_url,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": base.THINKING_ENABLED}
                },
                "grammar_domain": "binarydoor",
                "grammar_max_tool_calls": MAX_TOOL_CALLS,
            },
            "seed": SEED,
        },
    )


def pin_registry_schema(optimizer: GreedyOptimizer) -> dict[str, Any]:
    """Pin decoding and parsing to the tools advertised by this optimizer.

    The generic domain grammar is built from a training-time full inventory,
    which is broader than a runtime config's registry.  Keep this fix isolated
    from the running Zelda jobs and make the R074 decoder/parser contract match
    the single advertised place_tile tool.
    """
    pack = build_step_grammar_pack(
        optimizer.registry,
        max_tool_calls=MAX_TOOL_CALLS,
        coord_bounds=(optimizer.current_level.height, optimizer.current_level.width),
    )
    schema = json.loads(json.dumps(pack.json_schema))
    optimizer._openai_step_schema = schema
    optimizer._openai_structured_outputs = {"json": schema}
    return schema


def _corridor_cells(endpoints: tuple[int, ...]) -> list[tuple[int, int]]:
    if len(endpoints) != 7 or endpoints[0] != 0 or endpoints[-1] != 9:
        raise ValueError("endpoints must be (0,x1,x2,x3,x4,x5,9)")
    cells = [(4, 0)]
    y, x = cells[0]
    for next_x in endpoints[1:]:
        direction = 1 if next_x > x else -1
        while x != next_x:
            x += direction
            cells.append((y, x))
        if y < 14:
            y += 1
            cells.append((y, x))
            y += 1
            cells.append((y, x))
    cells.append((15, 9))
    return cells


def construct_witness(target: int) -> tuple[Level, tuple[int, ...]]:
    """Construct a deterministic exact witness without exposing it to the model."""
    required_horizontal = target - 13
    for middle in product(range(16), repeat=5):
        endpoints = (0, *middle, 9)
        horizontal = sum(
            abs(right - left) for left, right in zip(endpoints, endpoints[1:])
        )
        if horizontal != required_horizontal:
            continue
        cells = _corridor_cells(endpoints)
        if len(cells) != len(set(cells)):
            continue
        indices = {cell: index for index, cell in enumerate(cells)}
        induced = all(
            abs(index - neighbor_index) == 1
            for index, (y, x) in enumerate(cells)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (neighbor_index := indices.get((y + dy, x + dx))) is not None
            and neighbor_index != index
        )
        if not induced:
            continue
        data = np.ones((16, 16), dtype=np.int32)
        for cell in cells:
            data[cell] = 0
        return Level(data=data, problem_type="binarydoor"), endpoints
    raise RuntimeError(f"no deterministic witness for door_path={target}")


def _trace_rows(optimizer: GreedyOptimizer) -> list[dict[str, Any]]:
    rows = []
    for step in optimizer.step_records:
        before = {
            "door_path": int(step.metrics_before.get("door_path", 0)),
            "num_connected_regions": int(
                step.metrics_before.get("num_connected_regions", 0)
            ),
        }
        candidate = {
            key: int(step.metrics_after.get(key, value))
            for key, value in before.items()
        }
        candidate_delta = {key: candidate[key] - before[key] for key in before}
        rows.append(
            {
                "step_num": int(step.step_num),
                "message_type": step.message_type,
                "thinking_enabled": getattr(step, "thinking_enabled", None),
                "reasoning_handoff_chars": getattr(
                    step, "reasoning_handoff_chars", 0
                ),
                "reasoning_content": step.reasoning_content,
                "final_json": step.parsed_message,
                "metrics_before": before,
                "candidate_metrics_after": candidate,
                "candidate_delta": candidate_delta,
                "committed_delta": (
                    candidate_delta
                    if step.accepted
                    else {key: 0 for key in candidate_delta}
                ),
                "accepted": bool(step.accepted),
                "accept_reason": step.accept_reason,
                "truncated": bool(step.truncated),
                "input_tokens": int(step.input_tokens),
                "output_tokens": int(step.output_tokens),
            }
        )
    return rows


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [int(row["absolute_error"]) for row in records]
    return {
        "cases": len(records),
        "exact": sum(bool(row["exact"]) for row in records),
        "exact_rate": (
            sum(bool(row["exact"]) for row in records) / len(records)
            if records
            else 0.0
        ),
        "hard_valid": sum(
            bool(row["hard_constraints"]["hard_valid"]) for row in records
        ),
        "mae": sum(errors) / len(errors) if errors else None,
        "worst_error": max(errors) if errors else None,
        "parser_errors": sum(int(row["parser_errors"]) for row in records),
        "unrecovered_parser_errors": sum(
            int(row["unrecovered_parser_errors"]) for row in records
        ),
        "tool_errors": sum(int(row["tool_errors"]) for row in records),
        "transport_errors": sum(int(row["transport_errors"]) for row in records),
        "truncated_responses": sum(
            int(row["truncated_responses"]) for row in records
        ),
        "recovery_attempts": sum(
            bool(row["protocol_recovery"]["attempted"]) for row in records
        ),
        "recovered_truncations": sum(
            int(row["protocol_recovery"]["recovered_truncations"])
            for row in records
        ),
        "llm_calls": sum(int(row["costs"]["llm_calls"]) for row in records),
    }


def preflight(output: Path) -> dict[str, Any]:
    witness_rows = []
    initial_hashes = set()
    schema_hashes = set()
    for target in TARGETS:
        config = exact_config(target)
        optimizer = GreedyOptimizer(config, verbose=False)
        optimizer.initialize(seed=SEED)
        initial_hashes.add(base._sha256_text(optimizer.current_level.to_string()))
        schema = pin_registry_schema(optimizer)
        schema_hashes.add(
            hashlib.sha256(base.canonical_json_bytes(schema)).hexdigest()
        )
        schema_text = json.dumps(schema, sort_keys=True)
        if '"place_tile"' not in schema_text or "generate_bsp" in schema_text:
            raise RuntimeError("Binary Door schema escaped the place_tile-only contract")

        witness, endpoints = construct_witness(target)
        evaluator = optimizer.evaluator
        metrics = evaluator.evaluate(witness).metrics
        if (
            int(metrics.get("door_path", 0)) != target
            or int(metrics.get("num_connected_regions", 0)) != 1
        ):
            raise RuntimeError(f"unreachable target in preflight: {target}: {metrics}")
        witness_rows.append(
            {
                "target": target,
                "endpoints": endpoints,
                "door_path": int(metrics["door_path"]),
                "num_connected_regions": int(metrics["num_connected_regions"]),
                "level_sha256": base._sha256_text(witness.to_string()),
            }
        )
    payload = {
        "gate_pass": len(initial_hashes) == 1 and len(schema_hashes) == 1,
        "targets": list(TARGETS),
        "target_count": len(TARGETS),
        "all_targets_oracle_reachable": True,
        "initialization_sha256": next(iter(initial_hashes)),
        "schema_sha256": next(iter(schema_hashes)),
        "tool_contract": ["place_tile"],
        "witnesses": witness_rows,
    }
    base._write_json(output, payload)
    if not payload["gate_pass"]:
        raise RuntimeError("R074 preflight failed")
    return payload


def run_probe(
    *,
    base_url: str,
    model_id: str,
    revision: str,
    instruction_path: Path,
    output: Path,
    targets: tuple[int, ...] = TARGETS,
    acceptance_policy: str = LEGACY_ACCEPTANCE,
) -> dict[str, Any]:
    instruction = instruction_path.read_text().strip()
    if not instruction:
        raise ValueError("extra instruction is empty")
    served_model_id = base._server_model_id(base_url, expected=model_id)
    records: list[dict[str, Any]] = []
    for target in targets:
        config = exact_config(
            target,
            model=model_id,
            base_url=base_url,
            extra_instruction=instruction,
        )
        optimizer = build_optimizer(config, acceptance_policy=acceptance_policy)
        optimizer.initialize(seed=SEED)
        pin_registry_schema(optimizer)
        initial_ascii = optimizer.current_level.to_string()
        started = time.perf_counter()
        recovery = base._run_probe_optimizer(
            optimizer,
            max_steps=MAX_STEPS,
            truncation_recovery="thinking_off_handoff",
            thinking_max_tokens=NATIVE_MAX_TOKENS,
            fallback_max_tokens=FALLBACK_MAX_TOKENS,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        parser_errors = sum(
            step.message_type in {"PARSE_ERROR", "ERROR"}
            for step in optimizer.step_records
        )
        transport_errors = sum(
            step.message_type == "ERROR" for step in optimizer.step_records
        )
        truncated = sum(bool(step.truncated) for step in optimizer.step_records)
        unrecovered_parser_errors = max(
            0, parser_errors - int(recovery["recovered_truncations"])
        )
        tool_errors = sum(
            not item.get("result", {}).get("success", False)
            for step in optimizer.step_records
            for item in step.tool_results
            if item.get("tool_name") != "_truncated"
        )
        final_metrics = {
            "door_path": int(optimizer.current_result.metrics.get("door_path", 0)),
            "num_connected_regions": int(
                optimizer.current_result.metrics.get("num_connected_regions", 0)
            ),
        }
        hard = {
            "doors_connected": final_metrics["door_path"] > 0,
            "regions_one": final_metrics["num_connected_regions"] == 1,
            "protocol_clean": unrecovered_parser_errors == 0,
            "tool_clean": tool_errors == 0,
        }
        hard["hard_valid"] = all(hard.values())
        absolute_error = abs(final_metrics["door_path"] - target)
        record = {
            "status": "complete",
            "target_id": f"dp{target:03d}",
            "target": {"door_path": target, "num_connected_regions": 1},
            "seed": SEED,
            "initialization_sha256": base._sha256_text(initial_ascii),
            "final_level_ascii": optimizer.current_level.to_string(),
            "final_level_sha256": base._sha256_text(
                optimizer.current_level.to_string()
            ),
            "final_metrics": final_metrics,
            "hard_constraints": hard,
            "absolute_error": absolute_error,
            "exact": hard["hard_valid"] and absolute_error == 0,
            "parser_errors": parser_errors,
            "unrecovered_parser_errors": unrecovered_parser_errors,
            "transport_errors": transport_errors,
            "tool_errors": tool_errors,
            "truncated_responses": truncated,
            "protocol_recovery": recovery,
            "steps": _trace_rows(optimizer),
            "costs": {
                "llm_calls": len(optimizer.step_records),
                "input_tokens": sum(
                    int(step.input_tokens) for step in optimizer.step_records
                ),
                "output_tokens": sum(
                    int(step.output_tokens) for step in optimizer.step_records
                ),
                "latency_ms": elapsed_ms,
                "retries": int(optimizer.llm_retry_count),
                "semantic_actions": sum(
                    step.message_type == "STEP" for step in optimizer.step_records
                ),
                "evaluator_calls": 1
                + sum(
                    step.message_type == "STEP" for step in optimizer.step_records
                ),
            },
        }
        records.append(record)
        base._write_json(
            output,
            {
                "execution_complete": False,
                "model_id": model_id,
                "revision": revision,
                "server_model_id": served_model_id,
                "records": records,
            },
        )

    payload = {
        "execution_complete": len(records) == len(targets),
        "contract_gate_pass": (
            len(records) == len(targets)
            and all(row["transport_errors"] == 0 for row in records)
            and all(row["costs"]["llm_calls"] <= MAX_STEPS for row in records)
        ),
        "claim_scope": (
            "eight-target Binary Door harness-transfer probe"
            if targets == TARGETS and acceptance_policy == LEGACY_ACCEPTANCE
            else f"{len(targets)}-target Binary Door acceptance diagnostic"
        ),
        "model_id": model_id,
        "revision": revision,
        "server_model_id": served_model_id,
        "targets": list(targets),
        "budgets": {
            "semantic_actions": MAX_STEPS,
            "max_tool_calls_per_action": MAX_TOOL_CALLS,
            "thinking_max_tokens": NATIVE_MAX_TOKENS,
            "fallback_max_tokens": FALLBACK_MAX_TOKENS,
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": base.THINKING_ENABLED,
            "grammar_backend": "vllm-json-schema",
        },
        "harness": {
            "controller": "greedy accept/rollback",
            "truncation_recovery": "thinking_off_handoff",
            "conversation_length": 2,
            "initialization": "empty",
            "edit_tools": ["place_tile"],
            "instruction_sha256": base._sha256_text(instruction),
            **(
                {"acceptance_policy": acceptance_policy}
                if acceptance_policy != LEGACY_ACCEPTANCE
                else {}
            ),
        },
        "aggregate": _aggregate(records),
        "records": records,
    }
    base._write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError("R074 Binary Door probe contract failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--extra-instruction-file", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--acceptance-policy",
        choices=ACCEPTANCE_POLICIES,
        default=LEGACY_ACCEPTANCE,
    )
    run.add_argument("--target", type=int, action="append", choices=TARGETS)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.output)
    else:
        result = run_probe(
            base_url=args.base_url,
            model_id=args.model_id,
            revision=args.revision,
            instruction_path=args.extra_instruction_file,
            output=args.output,
            targets=tuple(args.target) if args.target else TARGETS,
            acceptance_policy=args.acceptance_policy,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

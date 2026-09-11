"""R083-D1 formal Qwen3.8 SMB baseline on the frozen F_SMB-v0 dev split.

The harness is intentionally the R082 winner: exact contract, candidate-local
A* event feedback, target-lexicographic acceptance, raw tools, empty
initialization, and temperature zero.  Oracle witness levels and traces are not
loaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from experiments.r061_qwen38_m1 import (
    _server_model_id,
    _set_thinking_enabled,
    _sha256_text,
    _thinking_enabled,
    _truncation_handoff_instruction,
    _write_json,
)
from experiments.r080_smb_astar_probe import (
    TARGET_FIRST_ACCEPTANCE,
    TARGET_METRICS,
    _assert_astar,
    _config,
    _step_traces,
    build_optimizer,
)
from toolusepcg.level import Level
from toolusepcg.smb_contract import smb_target_error


DEFAULT_MANIFEST = Path("benchmarks/smb/targets.json")
EXPECTED_MANIFEST_ID = "R083-D1-F_SMB-v0-dev"
EXPECTED_CASES = 29
SMOKE_CASE_IDS = (
    "smb-dev-00-k0-c0-j1",
    "smb-dev-10-k1-c1-j2",
    "smb-dev-21-k2-c2-j3",
    "smb-dev-26-k3-c3-j0",
    "smb-dev-04-k0-c4-j4",
    "smb-dev-06-k0-c5-j5",
)
PROTOCOL_POLICY = "bounded_thinking_off_retry-v1"


def _compact_serializer_instruction(base_instruction: str | None) -> str:
    """Request a short valid action after a thinking-off truncation."""
    retry = (
        "The previous response was truncated before a valid JSON action. "
        "Retry against the unchanged current level. Return one valid STEP or "
        "STOP immediately: keep rationale and plan to one short sentence each, "
        "use at most 12 tool calls, prefer line/rect over repeated set_tile calls, "
        "and do not repeat analysis already present in the conversation."
    )
    return f"{base_instruction}\n\n{retry}" if base_instruction else retry


def _run_smb_optimizer(
    optimizer: Any,
    *,
    max_steps: int,
    max_protocol_retries: int,
    thinking_max_tokens: int = 6144,
    fallback_max_tokens: int = 8192,
) -> dict[str, Any]:
    """Run with separate semantic-action and protocol-retry budgets.

    A truncated call never consumes one of ``max_steps`` semantic actions. The
    first thinking-on truncation hands its draft to a thinking-off serializer;
    later truncations receive a compact serializer instruction. Recovery is
    bounded by ``max_protocol_retries`` and the combined call cap.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_protocol_retries < 0:
        raise ValueError("max_protocol_retries cannot be negative")

    if _thinking_enabled(optimizer.config):
        optimizer.config.llm.max_tokens = thinking_max_tokens

    semantic_steps = 0
    protocol_retries = 0
    total_truncations = 0
    pending_truncations = 0
    trigger_steps: list[int] = []
    reasoning_handoff_chars = 0
    retry_instruction_pending = False
    base_instruction = optimizer.config.optimizer.extra_instruction
    call_cap = max_steps + max_protocol_retries

    while semantic_steps < max_steps and len(optimizer.step_records) < call_cap:
        if optimizer.termination.should_terminate(optimizer.current_result):
            break

        request_thinking = _thinking_enabled(optimizer.config)
        step = optimizer.step()
        step.thinking_enabled = request_thinking
        step.reasoning_handoff_chars = (
            reasoning_handoff_chars if retry_instruction_pending else 0
        )
        if retry_instruction_pending:
            optimizer.config.optimizer.extra_instruction = base_instruction
            retry_instruction_pending = False
            reasoning_handoff_chars = 0

        if step.truncated:
            total_truncations += 1
            pending_truncations += 1
            trigger_steps.append(int(step.step_num))
            if protocol_retries >= max_protocol_retries:
                break
            protocol_retries += 1
            _set_thinking_enabled(optimizer.config, False)
            optimizer.config.llm.max_tokens = fallback_max_tokens
            if request_thinking and step.reasoning_content:
                reasoning_handoff_chars = len(step.reasoning_content)
                optimizer.config.optimizer.extra_instruction = (
                    _truncation_handoff_instruction(
                        step.reasoning_content,
                        base_instruction=base_instruction,
                    )
                )
            else:
                optimizer.config.optimizer.extra_instruction = (
                    _compact_serializer_instruction(base_instruction)
                )
            retry_instruction_pending = True
            continue

        if step.message_type in {"STEP", "STOP"}:
            semantic_steps += 1
            pending_truncations = 0
            if step.message_type == "STOP":
                break
            continue
        if step.message_type in {"ERROR", "PARSE_ERROR"}:
            break

    optimizer.config.optimizer.extra_instruction = base_instruction
    recovered_truncations = total_truncations - pending_truncations
    return {
        "policy": PROTOCOL_POLICY,
        "attempted": total_truncations > 0,
        "trigger_step": trigger_steps[0] if trigger_steps else None,
        "trigger_steps": trigger_steps,
        "fallback_calls": max(0, len(optimizer.step_records) - semantic_steps),
        "semantic_steps": semantic_steps,
        "protocol_retries": protocol_retries,
        "max_protocol_retries": max_protocol_retries,
        "call_cap": call_cap,
        "recovered_truncations": recovered_truncations,
        "unrecovered_truncations": pending_truncations,
        "reasoning_handoff_chars": sum(
            int(getattr(step, "reasoning_handoff_chars", 0))
            for step in optimizer.step_records
        ),
        "budget_exhausted": (
            pending_truncations > 0
            and (
                protocol_retries >= max_protocol_retries
                or len(optimizer.step_records) >= call_cap
            )
        ),
    }


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_cases(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    case_ids: Iterable[str] | None = None,
    seed: int = 83083,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    claimed_sha = manifest.get("semantic_sha256")
    semantic_core = dict(manifest)
    semantic_core.pop("semantic_sha256", None)
    if claimed_sha != _semantic_sha256(semantic_core):
        raise ValueError("D1 manifest semantic hash mismatch")
    if manifest.get("manifest_id") != EXPECTED_MANIFEST_ID:
        raise ValueError(f"unexpected D1 manifest: {manifest.get('manifest_id')!r}")
    if manifest.get("provisional_only"):
        raise ValueError("formal D1 cannot use a provisional target manifest")
    if manifest.get("oracle_witnesses_exposed_to_model"):
        raise ValueError("D1 manifest exposes oracle witnesses")
    settings = manifest.get("settings", {})
    expected_settings = {
        "temperature": 0.0,
        "initialization": "empty",
        "agent": "astar",
        "prompt_contract": "exact",
        "cases": EXPECTED_CASES,
    }
    if settings != expected_settings:
        raise ValueError(f"unexpected D1 settings: {settings!r}")
    if len(manifest.get("targets", [])) != EXPECTED_CASES:
        raise ValueError("D1 manifest must contain exactly 29 dev targets")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    forbidden = ("level_sha256", "event_trace_sha256", "witness_path")
    for row in manifest["targets"]:
        encoded = json.dumps(row)
        if any(field in encoded for field in forbidden):
            raise ValueError("D1 case contains oracle witness information")
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate D1 case: {case_id}")
        seen.add(case_id)
        target = row["target"]
        enemies = int(target["enemies"])
        coins = int(target["coins"])
        jumps = int(target["jumps"])
        if not (0 <= enemies <= 4 and 0 <= coins <= 5 and 0 <= jumps <= 6):
            raise ValueError(f"out-of-contract D1 target: {case_id}")
        if float(target["complete"]) != 1.0:
            raise ValueError(f"D1 target must require completion: {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "spec_id": EXPECTED_MANIFEST_ID,
                "seed": int(seed),
                "targets": {
                    "complete": 1.0,
                    "enemies": enemies,
                    "coins": coins,
                    "jumps": jumps,
                },
            }
        )

    if case_ids is not None:
        requested = tuple(case_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("repeated --case-id values are not allowed")
        by_id = {case["case_id"]: case for case in cases}
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ValueError(f"unknown D1 cases: {missing}")
        cases = [by_id[case_id] for case_id in requested]
    return manifest, cases


def smoke_case_ids() -> tuple[str, ...]:
    return SMOKE_CASE_IDS


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate zero D1 records")
    e_inf = [float(row["e_inf"]) for row in records]
    e_sum = [float(row["e_sum"]) for row in records]
    per_metric_exact = {
        metric: sum(float(row["metric_gaps"][metric]) == 0.0 for row in records)
        for metric in TARGET_METRICS
    }
    return {
        "cases": len(records),
        "playable": sum(bool(row["playable"]) for row in records),
        "hard_valid": sum(bool(row["hard_valid"]) for row in records),
        "exact": sum(bool(row["exact"]) for row in records),
        "exact_rate": sum(bool(row["exact"]) for row in records) / len(records),
        "per_metric_exact": per_metric_exact,
        "e_inf_mean": sum(e_inf) / len(e_inf),
        "e_inf_p95": _percentile_nearest_rank(e_inf, 0.95),
        "e_inf_worst": max(e_inf),
        "e_sum_mean": sum(e_sum) / len(e_sum),
        "parser_errors": sum(int(row["parser_errors"]) for row in records),
        "unrecovered_parser_errors": sum(
            int(row["unrecovered_parser_errors"]) for row in records
        ),
        "transport_errors": sum(int(row["transport_errors"]) for row in records),
        "tool_errors": sum(int(row["tool_errors"]) for row in records),
        "failure_taxonomy": {
            "hard_validity_or_playability": sum(
                not bool(row["hard_valid"]) for row in records
            ),
            "supported_target_miss": sum(
                bool(row["hard_valid"]) and not bool(row["exact"]) for row in records
            ),
            "enemy_miss": len(records) - per_metric_exact["enemies"],
            "coin_miss": len(records) - per_metric_exact["coins"],
            "jump_miss": len(records) - per_metric_exact["jumps"],
        },
    }


def preflight(
    manifest_path: Path,
    output: Path,
    *,
    case_ids: Iterable[str] | None = None,
    seed: int = 83083,
    optimizer_builder: Callable[..., Any] = build_optimizer,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, cases = build_cases(manifest_path, case_ids=case_ids, seed=seed)
    optimizer = optimizer_builder(
        _config(cases[0], model_id="preflight", base_url="http://127.0.0.1:1/v1"),
        acceptance_policy=TARGET_FIRST_ACCEPTANCE,
    )
    _assert_astar(optimizer)
    flat = np.zeros((16, 32), dtype=np.int32)
    flat[-1, :] = 1
    witness = optimizer.evaluator.evaluate(Level(flat, problem_type="smb"))
    if witness.metrics.get("complete") != 1.0 or not witness.info.get("actions"):
        raise RuntimeError("real A* flat-level execution witness failed")
    payload = {
        "gate_pass": True,
        "manifest_id": manifest["manifest_id"],
        "manifest_semantic_sha256": manifest["semantic_sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "targets": [case["targets"] for case in cases],
        "solver_config": optimizer.config.env.smb_solver,
        "resolved_player_agent": optimizer.factory.env._problem._agent,
        "astar_execution_witness": {
            "complete": witness.metrics["complete"],
            "agent_events": len(witness.info["actions"]),
            "trajectory_points": len(witness.info["locations"]),
        },
        "oracle_witnesses_loaded": False,
        "seed": seed,
    }
    payload.update(metadata or {})
    _write_json(output, payload)
    return payload


def run(
    manifest_path: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    max_protocol_retries: int = 4,
    temperature: float = 0.0,
    case_ids: Iterable[str] | None = None,
    seed: int = 83083,
    optimizer_builder: Callable[..., Any] = build_optimizer,
    harness_overrides: dict[str, Any] | None = None,
    experiment_name: str = "R083-D1",
) -> dict[str, Any]:
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    served_model_id = _server_model_id(base_url, expected=model_id)
    manifest, cases = build_cases(manifest_path, case_ids=case_ids, seed=seed)
    records: list[dict[str, Any]] = []
    for case in cases:
        case_config = _config(case, model_id=model_id, base_url=base_url)
        case_config.llm.temperature = float(temperature)
        optimizer = optimizer_builder(
            case_config,
            acceptance_policy=TARGET_FIRST_ACCEPTANCE,
        )
        _assert_astar(optimizer)
        optimizer.initialize(seed=int(case["seed"]))
        initial_level = optimizer.current_level.to_string()
        started = time.perf_counter()
        recovery = _run_smb_optimizer(
            optimizer,
            max_steps=max_steps,
            max_protocol_retries=max_protocol_retries,
            thinking_max_tokens=6144,
            fallback_max_tokens=8192,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        parser_errors = sum(
            step.message_type in {"PARSE_ERROR", "ERROR"}
            for step in optimizer.step_records
        )
        unrecovered_parser_errors = max(
            0, parser_errors - int(recovery["recovered_truncations"])
        )
        transport_errors = sum(
            step.message_type == "ERROR" for step in optimizer.step_records
        )
        tool_errors = sum(
            not item.get("result", {}).get("success", False)
            for step in optimizer.step_records
            for item in step.tool_results
            if item.get("tool_name") != "_truncated"
        )
        result = optimizer.current_result
        metrics = dict(result.metrics)
        info = result.info or {}
        target_error = smb_target_error(metrics, case["targets"])
        gaps = {
            "complete": abs(float(metrics.get("complete", 0)) - 1.0),
            "enemies": target_error.e_k,
            "coins": target_error.e_c,
            "jumps": target_error.e_j,
        }
        astar_executed = bool(info.get("engine_executed")) and bool(info.get("actions"))
        playable = float(metrics.get("complete", 0.0)) == 1.0
        hard_valid = (
            result.valid
            and playable
            and astar_executed
            and unrecovered_parser_errors == 0
            and transport_errors == 0
            and tool_errors == 0
        )
        exact = hard_valid and gaps["complete"] == 0 and target_error.exact
        records.append(
            {
                "status": "complete",
                "case_id": case["case_id"],
                "spec_id": case["spec_id"],
                "seed": case["seed"],
                "targets": case["targets"],
                "initialization_sha256": _sha256_text(initial_level),
                "final_level": optimizer.current_level.to_string(),
                "final_level_sha256": _sha256_text(optimizer.current_level.to_string()),
                "final_metrics": metrics,
                "metric_gaps": gaps,
                "e_inf": max(gaps.values()),
                "e_sum": sum(gaps.values()),
                "playable": playable,
                "hard_valid": hard_valid,
                "exact": exact,
                "astar": {
                    "configured": "astar",
                    "resolved": optimizer.factory.env._problem._agent,
                    "executed": astar_executed,
                    "skip_reason": info.get("engine_skip_reason"),
                    "agent_events": len(info.get("actions", [])),
                    "trajectory_points": len(info.get("locations", [])),
                    "game_events": len(info.get("game_events", [])),
                },
                "parser_errors": parser_errors,
                "unrecovered_parser_errors": unrecovered_parser_errors,
                "transport_errors": transport_errors,
                "tool_errors": tool_errors,
                "truncated_responses": sum(
                    bool(step.truncated) for step in optimizer.step_records
                ),
                "protocol_recovery": recovery,
                "steps": _step_traces(optimizer),
                "costs": {
                    "llm_calls": len(optimizer.step_records),
                    "input_tokens": sum(
                        step.input_tokens for step in optimizer.step_records
                    ),
                    "output_tokens": sum(
                        step.output_tokens for step in optimizer.step_records
                    ),
                    "latency_ms": elapsed_ms,
                    "retries": int(optimizer.llm_retry_count),
                },
            }
        )
        _write_json(
            output,
            {
                "execution_complete": False,
                "manifest_semantic_sha256": manifest["semantic_sha256"],
                "model_id": model_id,
                "revision": revision,
                "records": records,
            },
        )

    aggregate = aggregate_records(records)
    harness = {
        "baseline": "R082 candidate-event feedback",
        "acceptance_policy": TARGET_FIRST_ACCEPTANCE,
        "max_steps": max_steps,
        "max_protocol_retries": max_protocol_retries,
        "conversation_length": 2,
        "max_tool_calls_per_step": 20,
        "temperature": float(temperature),
        "enable_thinking": True,
        "grammar_domain": "smb",
        "truncation_recovery": PROTOCOL_POLICY,
        "thinking_max_tokens": 6144,
        "fallback_max_tokens": 8192,
        "init_strategy": "empty",
        "oracle_witnesses_loaded": False,
    }
    harness.update(harness_overrides or {})
    payload = {
        "execution_complete": len(records) == len(cases),
        "contract_gate_pass": (
            len(records) == len(cases)
            and all(row["astar"]["resolved"] == "astar" for row in records)
            and aggregate["transport_errors"] == 0
            and aggregate["unrecovered_parser_errors"] == 0
            and all(
                row["protocol_recovery"]["semantic_steps"] <= max_steps
                and row["protocol_recovery"]["protocol_retries"]
                <= max_protocol_retries
                and len(row["steps"]) <= max_steps + max_protocol_retries
                for row in records
            )
        ),
        "smoke_gate_pass": (
            len(cases) != len(SMOKE_CASE_IDS)
            or (
                aggregate["hard_valid"] == len(cases)
                and aggregate["unrecovered_parser_errors"] == 0
                and aggregate["transport_errors"] == 0
                and aggregate["tool_errors"] == 0
            )
        ),
        "model_id": model_id,
        "revision": revision,
        "server_model_id": served_model_id,
        "domain": "smb",
        "player_agent": "astar",
        "manifest": {
            "path": str(manifest_path),
            "id": manifest["manifest_id"],
            "semantic_sha256": manifest["semantic_sha256"],
            "file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "case_ids": [case["case_id"] for case in cases],
        },
        "harness": harness,
        "aggregate": aggregate,
        "records": records,
    }
    _write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError(f"{experiment_name} contract gate failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=83083)
        command.add_argument("--smoke", action="store_true")
        command.add_argument("--case-id", action="append")
        if name == "run":
            command.add_argument("--base-url", required=True)
            command.add_argument("--model-id", required=True)
            command.add_argument("--revision", required=True)
            command.add_argument("--max-steps", type=int, default=4)
            command.add_argument("--max-protocol-retries", type=int, default=4)
            command.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    case_ids = args.case_id
    if args.smoke:
        if case_ids:
            raise ValueError("--smoke and --case-id are mutually exclusive")
        case_ids = list(smoke_case_ids())
    if args.command == "preflight":
        result = preflight(
            args.manifest, args.output, case_ids=case_ids, seed=args.seed
        )
    else:
        result = run(
            args.manifest,
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            max_protocol_retries=args.max_protocol_retries,
            temperature=args.temperature,
            case_ids=case_ids,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

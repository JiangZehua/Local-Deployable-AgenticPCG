"""Independent Qwen3.8 baseline for exact static SMB tile-count control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.r061_qwen38_m1 import _server_model_id, _sha256_text, _write_json
from experiments.r083_smb_baseline import _run_smb_optimizer
from toolusepcg.config import load_config
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.smb_static_metrics import (
    STATIC_TARGET_METRICS,
    build_static_support_scaffold,
)


STATIC_CASES = (
    ("static-00-p0-e0-c0-b0", 0, 0, 0, 0),
    ("static-01-p0-e1-c1-b4", 0, 1, 1, 4),
    ("static-02-p1-e0-c2-b8", 1, 0, 2, 8),
    ("static-03-p1-e2-c0-b12", 1, 2, 0, 12),
    ("static-04-p2-e3-c3-b12", 2, 3, 3, 12),
    ("static-05-p2-e0-c6-b20", 2, 0, 6, 20),
    ("static-06-p3-e6-c0-b24", 3, 6, 0, 24),
    ("static-07-p4-e1-c6-b32", 4, 1, 6, 32),
)


def build_cases(seed: int = 98098) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "seed": int(seed),
            "targets": {
                "pipe_count": pipes,
                "enemy_tile_count": enemies,
                "coin_tile_count": coins,
                "block_tile_count": blocks,
            },
        }
        for case_id, pipes, enemies, coins, blocks in STATIC_CASES
    ]


def _config(case: dict[str, Any], *, model_id: str, base_url: str) -> Any:
    return load_config(
        "configs/smb_static.yaml",
        {
            "targets": {
                key: {"mode": "target", "value": value}
                for key, value in case["targets"].items()
            },
            "optimizer": {
                "init_strategy": "empty",
                "conversation_length": 2,
                "max_tool_calls_per_step": 20,
            },
            "llm": {
                "provider": "openai",
                "model": model_id,
                "base_url": base_url,
                "max_tokens": 4096,
                "temperature": 0.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "grammar_domain": "smb",
                "grammar_max_tool_calls": 20,
            },
            "seed": int(case["seed"]),
        },
    )


def _build_optimizer(case: dict[str, Any], *, model_id: str, base_url: str) -> GreedyOptimizer:
    optimizer = GreedyOptimizer(
        _config(case, model_id=model_id, base_url=base_url), verbose=False
    )
    if optimizer.problem_type != "smb_static":
        raise RuntimeError(f"unexpected problem type: {optimizer.problem_type!r}")
    return optimizer


def _initialize_optimizer(optimizer: GreedyOptimizer, *, seed: int) -> None:
    """Start from fixed playable support that is outside count targets."""
    optimizer.initialize(
        seed=seed,
        initial_level=Level(build_static_support_scaffold(), "smb_static"),
    )


def _step_traces(optimizer: GreedyOptimizer) -> list[dict[str, Any]]:
    rows = []
    for step in optimizer.step_records:
        rows.append(
            {
                "step_num": int(step.step_num),
                "message_type": step.message_type,
                "final_json": step.parsed_message,
                "metrics_before": {
                    key: step.metrics_before.get(key, 0) for key in STATIC_TARGET_METRICS
                },
                "metrics_after": {
                    key: step.metrics_after.get(key, 0) for key in STATIC_TARGET_METRICS
                },
                "environment_feedback_after": step.environment_feedback_after,
                "accepted": bool(step.accepted),
                "accept_reason": step.accept_reason,
                "truncated": bool(step.truncated),
                "input_tokens": int(step.input_tokens),
                "output_tokens": int(step.output_tokens),
            }
        )
    return rows


def preflight(output: Path, *, seed: int = 98098) -> dict[str, Any]:
    case = build_cases(seed)[0]
    optimizer = _build_optimizer(
        case, model_id="preflight", base_url="http://127.0.0.1:1/v1"
    )
    _initialize_optimizer(optimizer, seed=seed)
    result = optimizer.current_result
    payload = {
        "gate_pass": (
            result.problem_type == "smb_static"
            and result.info.get("evaluation_mode") == "static_grid_plus_gameplay"
            and result.info.get("simulation_called") is True
            and result.info.get("engine_executed") is True
            and result.metrics.get("complete") == 1.0
            and set(STATIC_TARGET_METRICS).issubset(result.metrics)
        ),
        "domain": result.problem_type,
        "evaluation_mode": result.info.get("evaluation_mode"),
        "case_count": len(STATIC_CASES),
        "cases": build_cases(seed),
        "simulation_called": result.info.get("simulation_called"),
        "engine_executed": result.info.get("engine_executed"),
        "complete": result.metrics.get("complete"),
    }
    if not payload["gate_pass"]:
        raise RuntimeError("SMB static preflight failed")
    _write_json(output, payload)
    return payload


def run(
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    max_protocol_retries: int = 2,
    seed: int = 98098,
) -> dict[str, Any]:
    served_model_id = _server_model_id(base_url, expected=model_id)
    records: list[dict[str, Any]] = []
    for case in build_cases(seed):
        optimizer = _build_optimizer(case, model_id=model_id, base_url=base_url)
        _initialize_optimizer(optimizer, seed=int(case["seed"]))
        started = time.perf_counter()
        recovery = _run_smb_optimizer(
            optimizer,
            max_steps=max_steps,
            max_protocol_retries=max_protocol_retries,
            thinking_max_tokens=4096,
            fallback_max_tokens=4096,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        parser_errors = sum(
            step.message_type in {"PARSE_ERROR", "ERROR"}
            for step in optimizer.step_records
        )
        unrecovered = max(
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
        gaps = {
            key: abs(int(metrics.get(key, 0)) - int(case["targets"][key]))
            for key in STATIC_TARGET_METRICS
        }
        playability_valid = (
            result.valid
            and int(metrics.get("tube_issues", 0)) == 0
            and int(metrics.get("unsupported_pipe_count", 0)) == 0
            and int(metrics.get("unsupported_enemy_count", 0)) == 0
            and result.info.get("evaluation_mode") == "static_grid_plus_gameplay"
            and result.info.get("simulation_called") is True
            and result.info.get("engine_executed") is True
            and float(metrics.get("complete", 0.0)) == 1.0
            and int(metrics.get("coin_tiles_collected", -1))
            == int(metrics.get("coin_tile_count", 0))
            and unrecovered == 0
            and transport_errors == 0
            and tool_errors == 0
        )
        static_exact = all(value == 0 for value in gaps.values())
        playable_exact = playability_valid and static_exact
        records.append(
            {
                "status": "complete",
                "case_id": case["case_id"],
                "seed": case["seed"],
                "targets": case["targets"],
                "final_level": optimizer.current_level.to_string(),
                "final_level_sha256": _sha256_text(optimizer.current_level.to_string()),
                "final_metrics": metrics,
                "metric_gaps": gaps,
                "e_inf": max(gaps.values()),
                "e_sum": sum(gaps.values()),
                "hard_valid": playability_valid,
                "playability_valid": playability_valid,
                "static_exact": static_exact,
                "exact": playable_exact,
                "playable_exact": playable_exact,
                "parser_errors": parser_errors,
                "unrecovered_parser_errors": unrecovered,
                "transport_errors": transport_errors,
                "tool_errors": tool_errors,
                "protocol_recovery": recovery,
                "steps": _step_traces(optimizer),
                "costs": {
                    "llm_calls": len(optimizer.step_records),
                    "input_tokens": sum(s.input_tokens for s in optimizer.step_records),
                    "output_tokens": sum(s.output_tokens for s in optimizer.step_records),
                    "latency_ms": elapsed_ms,
                },
            }
        )
        _write_json(output, {"execution_complete": False, "records": records})

    per_metric_exact = {
        key: sum(row["metric_gaps"][key] == 0 for row in records)
        for key in STATIC_TARGET_METRICS
    }
    aggregate = {
        "cases": len(records),
        "hard_valid": sum(row["hard_valid"] for row in records),
        "playability_valid": sum(row["playability_valid"] for row in records),
        "static_exact": sum(row["static_exact"] for row in records),
        "exact": sum(row["exact"] for row in records),
        "playable_exact": sum(row["playable_exact"] for row in records),
        "exact_rate": sum(row["exact"] for row in records) / len(records),
        "per_metric_exact": per_metric_exact,
        "e_inf_mean": sum(row["e_inf"] for row in records) / len(records),
        "e_inf_worst": max(row["e_inf"] for row in records),
        "e_sum_mean": sum(row["e_sum"] for row in records) / len(records),
        "parser_errors": sum(row["parser_errors"] for row in records),
        "unrecovered_parser_errors": sum(
            row["unrecovered_parser_errors"] for row in records
        ),
        "transport_errors": sum(row["transport_errors"] for row in records),
        "tool_errors": sum(row["tool_errors"] for row in records),
    }
    payload = {
        "execution_complete": len(records) == len(STATIC_CASES),
        "contract_gate_pass": (
            len(records) == len(STATIC_CASES)
            and aggregate["unrecovered_parser_errors"] == 0
            and aggregate["transport_errors"] == 0
            and aggregate["tool_errors"] == 0
        ),
        "domain": "smb_static",
        "evaluation_mode": "static_grid_plus_gameplay",
        "model_id": model_id,
        "server_model_id": served_model_id,
        "revision": revision,
        "harness": {
            "max_steps": max_steps,
            "max_protocol_retries": max_protocol_retries,
            "temperature": 0.0,
            "enable_thinking": False,
            "init_strategy": "fixed_platform_scaffold",
            "simulation_called": True,
            "playability_requirements": [
                "supported_pipes",
                "supported_enemies",
                "astar_engine_executed",
                "complete",
                "all_coin_tiles_collected",
            ],
            "primary_metrics": list(STATIC_TARGET_METRICS),
        },
        "aggregate": aggregate,
        "records": records,
    }
    _write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError("SMB static baseline contract gate failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--seed", type=int, default=98098)
    execute = sub.add_parser("run")
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--model-id", required=True)
    execute.add_argument("--revision", required=True)
    execute.add_argument("--max-steps", type=int, default=4)
    execute.add_argument("--max-protocol-retries", type=int, default=2)
    execute.add_argument("--seed", type=int, default=98098)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.output, seed=args.seed)
    else:
        result = run(
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            max_protocol_retries=args.max_protocol_retries,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

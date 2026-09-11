"""Qwen3.8-27B SMB transfer probe with an A*-only player evaluator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.r061_qwen38_m1 import (
    _run_probe_optimizer,
    _server_model_id,
    _sha256_text,
    _write_json,
)
from toolusepcg.config import load_config
from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level
from toolusepcg.optimizer.greedy import GreedyOptimizer
from toolusepcg.optimizer.termination import TerminationChecker
from toolusepcg.smb_contract import (
    COLLECT_EVENT,
    COUNTED_KILL_EVENTS,
    JUMP_EVENT,
    SMB_COIN_EVENT_PARAMS,
    smb_target_error,
)


DEFAULT_SPEC_PATH = Path("benchmarks/smb/engine_probe.json")
TARGET_METRICS = ("complete", "enemies", "coins", "jumps")
LEGACY_ACCEPTANCE = "legacy_score"
TARGET_FIRST_ACCEPTANCE = "target_lexicographic"
ACCEPTANCE_POLICIES = (LEGACY_ACCEPTANCE, TARGET_FIRST_ACCEPTANCE)


class SmbTargetTermination(TerminationChecker):
    """Stop once every requested SMB metric is exact on a valid level."""

    def _exact_target_reached(self, result: EvalResult | None) -> bool:
        if result is None or result.problem_type != "smb" or not result.valid:
            return super()._exact_target_reached(result)
        return all(
            result.metrics.get(name) == getattr(self.config.targets, name).value
            for name in TARGET_METRICS
        )


class TargetFirstSmbOptimizer(GreedyOptimizer):
    """Put SMB validity and requested metrics ahead of rewrite cost.

    The legacy scalar score, including the change penalty, is consulted only
    when hard validity, worst target error, and total target error are tied.
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.termination = SmbTargetTermination(config)

    def _primary_quality(self, result: EvalResult) -> tuple[int, float, float]:
        metrics = result.metrics
        hard_violations = (
            int(not result.valid)
            + int(float(metrics.get("complete", 0.0)) < 1.0)
            + int(int(metrics.get("tube_issues", 0)) != 0)
        )
        errors = [
            abs(
                float(metrics.get(name, 0.0))
                - float(getattr(self.config.targets, name).value)
            )
            for name in TARGET_METRICS
        ]
        return hard_violations, max(errors), sum(errors)

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

    @staticmethod
    def _event_tile(
        event: dict[str, Any], width: int, height: int
    ) -> tuple[int, int]:
        position = event.get("mario_event_position")
        if position is not None:
            return int(position["x"]), int(position["y"])
        x = int(float(event.get("mario_x", 0.0)) / 16) - 3
        y = int(float(event.get("mario_y", 0.0)) / 16)
        return (
            max(0, min(width - 1, x)),
            max(0, min(height - 1, y)),
        )

    @staticmethod
    def _format_positions(positions: list[tuple[int, int]]) -> str:
        if not positions:
            return "none"
        return ", ".join(f"[{x},{y}]" for x, y in positions[:12])

    @staticmethod
    def _same_cell_overwrites(tool_results: list[dict]) -> list[str]:
        writes: dict[tuple[int, int], str] = {}
        warnings: list[str] = []
        for item in tool_results:
            if item.get("tool_name") != "place_tile":
                continue
            if not item.get("result", {}).get("success", False):
                continue
            params = item.get("parameters") or {}
            if params.get("mode") != "single":
                continue
            try:
                cell = int(params["x"]), int(params["y"])
            except (KeyError, TypeError, ValueError):
                continue
            tile = str(params.get("tile_type", "unknown"))
            previous = writes.get(cell)
            if previous is not None and previous != tile:
                warnings.append(
                    f"same-cell overwrite at [{cell[0]},{cell[1]}]: "
                    f"{previous} -> {tile}; only {tile} remains"
                )
            writes[cell] = tile
        return warnings

    def _build_environment_step_feedback(
        self,
        new_level: Level,
        new_result: EvalResult,
        tool_results: list[dict],
    ) -> str:
        """Explain the candidate's actual A* events, not just aggregate gaps."""
        metrics = new_result.metrics
        lines = [
            "Candidate-specific A* attribution (the edited candidate, even if rejected):"
        ]

        residuals = []
        for name, label in (
            ("enemies", "enemy kills"),
            ("coins", "coin collections"),
            ("jumps", "ground jumps"),
        ):
            current = int(metrics.get(name, 0))
            target = int(getattr(self.config.targets, name).value)
            delta = target - current
            if delta > 0:
                residuals.append(f"{label}: need +{delta}")
            elif delta < 0:
                residuals.append(f"{label}: overshot by {-delta}")
            else:
                residuals.append(f"{label}: exact")
        lines.append("- Residual: " + "; ".join(residuals))

        unchanged = [
            name
            for name in ("enemies", "coins", "jumps", "complete")
            if metrics.get(name) == self.current_result.metrics.get(name)
        ]
        if unchanged:
            lines.append(
                "- Unchanged gameplay metrics versus incumbent: "
                + ", ".join(unchanged)
            )

        info = new_result.info or {}
        events = info.get("game_events", [])
        width = int(info.get("width", new_level.width))
        height = int(info.get("height", new_level.height))
        kills = [
            self._event_tile(event, width, height)
            for event in events
            if event.get("event_type", event.get("type")) in COUNTED_KILL_EVENTS
        ]
        coins = [
            self._event_tile(event, width, height)
            for event in events
            if event.get("event_type", event.get("type")) == COLLECT_EVENT
            and event.get("event_param", event.get("param")) in SMB_COIN_EVENT_PARAMS
        ]
        jumps = [
            self._event_tile(event, width, height)
            for event in events
            if event.get("event_type", event.get("type")) == JUMP_EVENT
        ]
        lines.append(
            f"- Mario positions at counted kill events: {self._format_positions(kills)}"
        )
        lines.append(
            f"- Mario positions at coin pickup events: {self._format_positions(coins)}"
        )
        lines.append(
            f"- Mario positions at ground-jump events: {self._format_positions(jumps)}"
        )

        grid = new_level.to_string().splitlines()
        coin_spawns = [
            (x, y)
            for y, row in enumerate(grid)
            for x, glyph in enumerate(row)
            if glyph == "C"
        ]
        enemy_spawns = [
            (x, y)
            for y, row in enumerate(grid)
            for x, glyph in enumerate(row)
            if glyph in {"G", "K", "Y"}
        ]
        lines.append(f"- Candidate coin spawn tiles: {self._format_positions(coin_spawns)}")
        lines.append(f"- Candidate enemy spawn tiles: {self._format_positions(enemy_spawns)}")

        for warning in self._same_cell_overwrites(tool_results):
            lines.append(f"- WARNING: {warning}")
        lines.append(
            "- Only actual simulation events count. Do not infer kills, collections, "
            "or jumps from placed tile counts."
        )
        return "\n".join(lines)


def build_optimizer(
    config: Any,
    *,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
) -> GreedyOptimizer:
    """Use target-first by default; legacy remains an explicit replay option."""
    if acceptance_policy == LEGACY_ACCEPTANCE:
        return GreedyOptimizer(config, verbose=False)
    if acceptance_policy == TARGET_FIRST_ACCEPTANCE:
        return TargetFirstSmbOptimizer(config, verbose=False)
    raise ValueError(f"unknown acceptance policy: {acceptance_policy}")


def build_cases(spec_path: Path = DEFAULT_SPEC_PATH) -> list[dict[str, Any]]:
    payload = json.loads(spec_path.read_text())
    seeds = [int(seed) for seed in payload["seeds"]["test"]]
    specs = [
        item for item in payload["specs"]
        if item["split"] == "test" and item["game"] == "smb"
    ]
    return [
        {
            "case_id": f"{spec['id']}_seed-{seed}",
            "spec_id": spec["id"],
            "seed": seed,
            "targets": {key: spec["targets"][key] for key in TARGET_METRICS},
        }
        for spec in specs
        for seed in seeds
    ]


def _config(case: dict[str, Any], *, model_id: str, base_url: str) -> Any:
    return load_config("configs/smb.yaml", {
        "env": {"smb_solver": "astar"},
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
            "max_tokens": 8192,
            "temperature": 0.0,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "grammar_domain": "smb",
            "grammar_max_tool_calls": 20,
        },
        "seed": int(case["seed"]),
    })


def _assert_astar(optimizer: GreedyOptimizer) -> None:
    configured = optimizer.config.env.smb_solver
    resolved = getattr(optimizer.factory.env._problem, "_agent", None)
    if configured != "astar" or resolved != "astar":
        raise RuntimeError(
            f"SMB evaluator is not A*-only: configured={configured!r}, resolved={resolved!r}"
        )


def preflight(spec_path: Path, output: Path) -> dict[str, Any]:
    cases = build_cases(spec_path)
    if len(cases) != 9 or len({row["case_id"] for row in cases}) != 9:
        raise RuntimeError("expected frozen 3 SMB test specs x 3 seeds")
    optimizer = GreedyOptimizer(
        _config(cases[0], model_id="preflight", base_url="http://127.0.0.1:1/v1"),
        verbose=False,
    )
    _assert_astar(optimizer)
    flat_level = np.zeros((16, 32), dtype=np.int32)
    flat_level[-1, :] = 1
    witness = optimizer.evaluator.evaluate(Level(flat_level, problem_type="smb"))
    if witness.metrics.get("complete") != 1.0 or not witness.info.get("actions"):
        raise RuntimeError("real A* flat-level execution witness failed")
    payload = {
        "gate_pass": True,
        "cases": cases,
        "case_count": len(cases),
        "solver_config": optimizer.config.env.smb_solver,
        "resolved_player_agent": optimizer.factory.env._problem._agent,
        "astar_execution_witness": {
            "complete": witness.metrics["complete"],
            "agent_events": len(witness.info["actions"]),
            "trajectory_points": len(witness.info["locations"]),
        },
        "spec_path": str(spec_path),
        "spec_sha256": _sha256_text(spec_path.read_text()),
    }
    _write_json(output, payload)
    return payload


def _step_traces(optimizer: GreedyOptimizer) -> list[dict[str, Any]]:
    traces = []
    for step in optimizer.step_records:
        before = {key: step.metrics_before.get(key, 0) for key in TARGET_METRICS}
        after = {key: step.metrics_after.get(key, before[key]) for key in TARGET_METRICS}
        traces.append({
            "step_num": int(step.step_num),
            "message_type": step.message_type,
            "thinking_enabled": getattr(step, "thinking_enabled", None),
            "reasoning_handoff_chars": getattr(step, "reasoning_handoff_chars", 0),
            "reasoning_content": step.reasoning_content,
            "final_json": step.parsed_message,
            "metrics_before": before,
            "metrics_after": after,
            "environment_feedback_after": step.environment_feedback_after,
            "level_before": step.level_before,
            "level_after": step.level_after,
            "accepted": bool(step.accepted),
            "accept_reason": step.accept_reason,
            "truncated": bool(step.truncated),
            "input_tokens": int(step.input_tokens),
            "output_tokens": int(step.output_tokens),
        })
    return traces


def run(
    spec_path: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
    seeds: set[int] | None = None,
) -> dict[str, Any]:
    served_model_id = _server_model_id(base_url, expected=model_id)
    cases = build_cases(spec_path)
    if seeds is not None:
        cases = [case for case in cases if int(case["seed"]) in seeds]
        if not cases:
            raise ValueError(f"no frozen SMB cases matched seeds={sorted(seeds)}")
    records: list[dict[str, Any]] = []
    for case in cases:
        optimizer = build_optimizer(
            _config(case, model_id=model_id, base_url=base_url),
            acceptance_policy=acceptance_policy,
        )
        _assert_astar(optimizer)
        optimizer.initialize(seed=int(case["seed"]))
        initial_level = optimizer.current_level.to_string()
        started = time.perf_counter()
        recovery = _run_probe_optimizer(
            optimizer,
            max_steps=max_steps,
            truncation_recovery="thinking_off_handoff",
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
            "complete": abs(
                float(metrics.get("complete", 0))
                - float(case["targets"]["complete"])
            ),
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
            and tool_errors == 0
        )
        exact = hard_valid and gaps["complete"] == 0 and target_error.exact
        records.append({
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
            "truncated_responses": sum(bool(step.truncated) for step in optimizer.step_records),
            "protocol_recovery": recovery,
            "steps": _step_traces(optimizer),
            "costs": {
                "llm_calls": len(optimizer.step_records),
                "input_tokens": sum(step.input_tokens for step in optimizer.step_records),
                "output_tokens": sum(step.output_tokens for step in optimizer.step_records),
                "latency_ms": elapsed_ms,
                "retries": int(optimizer.llm_retry_count),
            },
        })
        _write_json(output, {
            "execution_complete": False,
            "model_id": model_id,
            "revision": revision,
            "records": records,
        })

    payload = {
        "execution_complete": len(records) == len(cases),
        "contract_gate_pass": (
            len(records) == len(cases)
            and all(row["astar"]["resolved"] == "astar" for row in records)
            and all(row["transport_errors"] == 0 for row in records)
            and all(len(row["steps"]) <= max_steps for row in records)
        ),
        "model_id": model_id,
        "revision": revision,
        "server_model_id": served_model_id,
        "domain": "smb",
        "player_agent": "astar",
        "case_source": (
            "rpe_specs_v3 test split: 3 SMB specs x 3 seeds"
            if seeds is None
            else f"rpe_specs_v3 test split: 3 SMB specs x seeds {sorted(seeds)}"
        ),
        "harness": {
            "acceptance_policy": acceptance_policy,
            "max_steps": max_steps,
            "conversation_length": 2,
            "max_tool_calls_per_step": 20,
            "temperature": 0.0,
            "enable_thinking": True,
            "grammar_domain": "smb",
            "truncation_recovery": "thinking_off_handoff",
            "thinking_max_tokens": 6144,
            "fallback_max_tokens": 8192,
            "init_strategy": "empty",
        },
        "aggregate": {
            "cases": len(records),
            "playable": sum(row["playable"] for row in records),
            "hard_valid": sum(row["hard_valid"] for row in records),
            "exact": sum(row["exact"] for row in records),
            "exact_rate": sum(row["exact"] for row in records) / len(records),
            "e_inf_mean": sum(row["e_inf"] for row in records) / len(records),
            "parser_errors": sum(row["parser_errors"] for row in records),
            "unrecovered_parser_errors": sum(row["unrecovered_parser_errors"] for row in records),
            "transport_errors": sum(row["transport_errors"] for row in records),
            "tool_errors": sum(row["tool_errors"] for row in records),
        },
        "records": records,
    }
    _write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError("SMB A* probe contract failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    check.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser("run")
    probe.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--model-id", required=True)
    probe.add_argument("--revision", required=True)
    probe.add_argument("--max-steps", type=int, default=4)
    probe.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Run only this frozen test seed (repeatable); default runs all seeds.",
    )
    probe.add_argument(
        "--acceptance-policy",
        choices=ACCEPTANCE_POLICIES,
        default=TARGET_FIRST_ACCEPTANCE,
    )
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.spec_path, args.output)
    else:
        result = run(
            args.spec_path,
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            acceptance_policy=args.acceptance_policy,
            seeds=set(args.seed) if args.seed else None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

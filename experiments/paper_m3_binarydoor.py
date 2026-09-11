"""PAPER-M3 BinaryDoor held-out geometry and unified regression experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments import r061_qwen38_m1 as base
from experiments.r074_binarydoor_harness_probe import (
    FALLBACK_MAX_TOKENS,
    MAX_STEPS,
    MAX_TOOL_CALLS,
    NATIVE_MAX_TOKENS,
    TARGETS as REGRESSION_TARGETS,
    TARGET_FIRST_ACCEPTANCE,
    _trace_rows,
    build_optimizer,
    construct_witness,
    pin_registry_schema,
)
from toolusepcg.config import load_config
from toolusepcg.level import Level


EXPERIMENT_ID = "PAPER-M3-BD32-REG8"
MANIFEST_SCHEMA = "paper-m3-binarydoor-manifest-v1"
MANIFEST_SEED = 6499021
NEW_CASES = 32
NEW_PER_STRATUM = 8
STRATA = (
    ("low", 32, 39),
    ("medium", 40, 55),
    ("high", 56, 71),
    ("extreme", 72, 112),
)
GENERAL_SCAFFOLD = Path("prompts/binarydoor_general_geometry_rules_v2.txt")
REGRESSION_SCAFFOLD = Path("prompts/binarydoor_geometry_serialization_rules_v1.txt")


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(base.canonical_json_bytes(body)).hexdigest()


def _interior_neighbor(
    door: Sequence[int], *, height: int = 16, width: int = 16
) -> tuple[int, int]:
    row, col = int(door[0]), int(door[1])
    if row == 0 and 1 <= col <= width:
        return 0, col - 1
    if row == height + 1 and 1 <= col <= width:
        return height - 1, col - 1
    if col == 0 and 1 <= row <= height:
        return row - 1, 0
    if col == width + 1 and 1 <= row <= height:
        return row - 1, width - 1
    raise ValueError(f"not a valid bordered-grid door: {door}")


def exact_config(
    target: int,
    door_seed: int,
    *,
    model_id: str = base.MODEL_ID,
    base_url: str = "http://127.0.0.1:8155/v1",
    extra_instruction: str | None = None,
    ablation_mode: str = "full",
) -> Any:
    return load_config(
        "configs/binarydoor.yaml",
        {
            "targets": {
                "door_path": {"mode": "target", "value": int(target)},
                "num_connected_regions": {"mode": "target", "value": 1},
            },
            "optimizer": {
                "init_strategy": "empty",
                "edit_tools": ["place_tile"],
                "conversation_length": 2,
                "max_tool_calls_per_step": MAX_TOOL_CALLS,
                "extra_instruction": extra_instruction,
                "stable_change_accounting": True,
                "ablation_mode": ablation_mode,
            },
            "llm": {
                "provider": "openai",
                "model": model_id,
                "base_url": base_url,
                "max_tokens": FALLBACK_MAX_TOKENS,
                "temperature": 0.0,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": base.THINKING_ENABLED}
                },
                "grammar_domain": "binarydoor",
                "grammar_max_tool_calls": MAX_TOOL_CALLS,
            },
            "seed": int(door_seed),
        },
    )


def _carve_to_maze(
    data: np.ndarray,
    start: tuple[int, int],
    nodes: set[tuple[int, int]],
    *,
    horizontal_first: bool,
) -> None:
    target = min(
        nodes,
        key=lambda cell: (
            abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
            cell[0],
            cell[1],
        ),
    )
    row, col = start
    data[row, col] = 0
    axes = ("col", "row") if horizontal_first else ("row", "col")
    for axis in axes:
        if axis == "row":
            while row != target[0]:
                row += 1 if target[0] > row else -1
                data[row, col] = 0
        else:
            while col != target[1]:
                col += 1 if target[1] > col else -1
                data[row, col] = 0


def oracle_witness(door_seed: int, maze_seed: int) -> tuple[Level, dict[str, Any]]:
    """Build a deterministic perfect-maze witness independent of model output."""
    probe_config = exact_config(32, door_seed)
    optimizer = build_optimizer(probe_config, acceptance_policy=TARGET_FIRST_ACCEPTANCE)
    optimizer.initialize(seed=door_seed)
    problem = optimizer.factory.env._problem
    door1 = tuple(int(value) for value in problem._door1)
    door2 = tuple(int(value) for value in problem._door2)
    start = _interior_neighbor(door1)
    end = _interior_neighbor(door2)

    rng = random.Random(int(maze_seed))
    row_offset = rng.randrange(2)
    col_offset = rng.randrange(2)
    node_rows = list(range(row_offset, 16, 2))
    node_cols = list(range(col_offset, 16, 2))
    nodes = {(row, col) for row in node_rows for col in node_cols}
    root = min(nodes, key=lambda cell: (abs(cell[0] - start[0]) + abs(cell[1] - start[1]), cell))
    data = np.ones((16, 16), dtype=np.int32)
    visited = {root}
    stack = [root]
    data[root] = 0
    while stack:
        row, col = stack[-1]
        candidates = [
            (row + drow, col + dcol)
            for drow, dcol in ((-2, 0), (2, 0), (0, -2), (0, 2))
            if (row + drow, col + dcol) in nodes
            and (row + drow, col + dcol) not in visited
        ]
        if not candidates:
            stack.pop()
            continue
        nxt = candidates[rng.randrange(len(candidates))]
        midpoint = ((row + nxt[0]) // 2, (col + nxt[1]) // 2)
        data[midpoint] = 0
        data[nxt] = 0
        visited.add(nxt)
        stack.append(nxt)

    _carve_to_maze(data, start, nodes, horizontal_first=bool(rng.randrange(2)))
    _carve_to_maze(data, end, nodes, horizontal_first=bool(rng.randrange(2)))
    level = Level(data=data, problem_type="binarydoor")
    result = optimizer.evaluator.evaluate(level)
    target = int(result.metrics.get("door_path", 0))
    regions = int(result.metrics.get("num_connected_regions", 0))
    if target <= 0 or regions != 1:
        raise RuntimeError(
            f"oracle construction failed seed={door_seed}/{maze_seed}: "
            f"door_path={target}, regions={regions}"
        )
    return level, {
        "door_seed": int(door_seed),
        "maze_seed": int(maze_seed),
        "door1": list(door1),
        "door2": list(door2),
        "door1_inner": list(start),
        "door2_inner": list(end),
        "target": target,
        "num_connected_regions": regions,
        "witness_sha256": base._sha256_text(level.to_string()),
    }


def build_manifest() -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in STRATA}
    used_door_seeds: set[int] = set()
    for candidate_index in range(20000):
        if all(len(rows) == NEW_PER_STRATUM for rows in buckets.values()):
            break
        door_seed = MANIFEST_SEED + candidate_index
        maze_seed = int.from_bytes(
            hashlib.sha256(f"{EXPERIMENT_ID}|{door_seed}".encode()).digest()[:8],
            "big",
        )
        _, oracle = oracle_witness(door_seed, maze_seed)
        target = int(oracle["target"])
        for name, lower, upper in STRATA:
            if lower <= target <= upper and len(buckets[name]) < NEW_PER_STRATUM:
                if door_seed in used_door_seeds:
                    raise AssertionError("door seed reused")
                buckets[name].append(
                    {
                        "case_id": f"bd-new-{name}-{len(buckets[name]):02d}",
                        "block": "heldout32",
                        "stratum": name,
                        "door_seed": door_seed,
                        "request_seed": door_seed,
                        "target": target,
                        "oracle": {
                            "maze_seed": maze_seed,
                            "witness_sha256": oracle["witness_sha256"],
                        },
                        "selection_index": candidate_index,
                        "scaffold": "general_geometry_v2",
                    }
                )
                used_door_seeds.add(door_seed)
                break
    missing = {name: NEW_PER_STRATUM - len(rows) for name, rows in buckets.items() if len(rows) < NEW_PER_STRATUM}
    if missing:
        raise RuntimeError(f"could not fill deterministic feasibility strata: {missing}")

    heldout = [row for name, _, _ in STRATA for row in buckets[name]]
    regression = []
    for target in REGRESSION_TARGETS:
        witness, _ = construct_witness(target)
        regression.append(
            {
                "case_id": f"bd-reg-dp{target:03d}",
                "block": "regression8",
                "stratum": "known_seed42",
                "door_seed": 42,
                "request_seed": 42,
                "target": int(target),
                "oracle": {
                    "witness_sha256": base._sha256_text(witness.to_string()),
                },
                "scaffold": "seed42_geometry_v1",
            }
        )
    heldout_columns = [
        "case_id", "stratum", "door_seed", "request_seed", "target",
        "maze_seed", "witness_sha256", "selection_index", "scaffold",
    ]
    regression_columns = ["case_id", "target", "witness_sha256", "scaffold"]
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "selection_status": "outcome-independent deterministic oracle feasibility only",
        "master_seed": MANIFEST_SEED,
        "strata": [
            {"name": name, "min_inclusive": low, "max_inclusive": high, "count": NEW_PER_STRATUM}
            for name, low, high in STRATA
        ],
        "heldout_count": len(heldout),
        "regression_count": len(regression),
        "heldout_columns": heldout_columns,
        "heldout_rows": [
            [
                row["case_id"], row["stratum"], row["door_seed"],
                row["request_seed"], row["target"], row["oracle"]["maze_seed"],
                row["oracle"]["witness_sha256"], row["selection_index"],
                row["scaffold"],
            ]
            for row in heldout
        ],
        "regression_columns": regression_columns,
        "regression_rows": [
            [
                row["case_id"], row["target"],
                row["oracle"]["witness_sha256"], row["scaffold"],
            ]
            for row in regression
        ],
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def validate_manifest(path: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text())
    if frozen.get("manifest_sha256") != _manifest_hash(frozen):
        raise RuntimeError("BinaryDoor manifest hash mismatch")
    if frozen.get("heldout_columns") != [
        "case_id", "stratum", "door_seed", "request_seed", "target",
        "maze_seed", "witness_sha256", "selection_index", "scaffold",
    ] or frozen.get("regression_columns") != [
        "case_id", "target", "witness_sha256", "scaffold",
    ]:
        raise RuntimeError("BinaryDoor compact manifest columns drifted")
    heldout = [
        {
            "case_id": row[0], "block": "heldout32", "stratum": row[1],
            "door_seed": row[2], "request_seed": row[3], "target": row[4],
            "oracle": {"maze_seed": row[5], "witness_sha256": row[6]},
            "selection_index": row[7], "scaffold": row[8],
        }
        for row in frozen.get("heldout_rows", [])
    ]
    regression = [
        {
            "case_id": row[0], "block": "regression8",
            "stratum": "known_seed42", "door_seed": 42,
            "request_seed": 42, "target": row[1],
            "oracle": {"witness_sha256": row[2]}, "scaffold": row[3],
        }
        for row in frozen.get("regression_rows", [])
    ]
    cases = heldout + regression
    if len(heldout) != NEW_CASES or len(regression) != len(REGRESSION_TARGETS):
        raise RuntimeError("BinaryDoor manifest denominator drift")
    if len({int(row["door_seed"]) for row in heldout}) != NEW_CASES:
        raise RuntimeError("held-out door geometries are not unique")
    for row in heldout:
        witness, oracle = oracle_witness(int(row["door_seed"]), int(row["oracle"]["maze_seed"]))
        if (
            int(oracle["target"]) != int(row["target"])
            or base._sha256_text(witness.to_string()) != row["oracle"]["witness_sha256"]
        ):
            raise RuntimeError(f"oracle drift for {row['case_id']}")
    manifest = dict(frozen)
    manifest["cases"] = cases
    return manifest


def _case_instruction(case: Mapping[str, Any], general: str, regression: str) -> str:
    return regression if case["scaffold"] == "seed42_geometry_v1" else general


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [int(row["absolute_error"]) for row in records]
    return {
        "cases": len(records),
        "exact": sum(bool(row["exact"]) for row in records),
        "hard_valid": sum(bool(row["hard_constraints"]["hard_valid"]) for row in records),
        "endpoint_attached": sum(bool(row["hard_constraints"]["endpoint_attached"]) for row in records),
        "mae": sum(errors) / len(errors) if errors else None,
        "worst_error": max(errors) if errors else None,
        "parser_errors": sum(int(row["parser_errors"]) for row in records),
        "unrecovered_parser_errors": sum(int(row["unrecovered_parser_errors"]) for row in records),
        "tool_errors": sum(int(row["tool_errors"]) for row in records),
        "transport_errors": sum(int(row["transport_errors"]) for row in records),
        "truncated_responses": sum(int(row["truncated_responses"]) for row in records),
        "recovered_truncations": sum(int(row["protocol_recovery"]["recovered_truncations"]) for row in records),
        "llm_calls": sum(int(row["costs"]["llm_calls"]) for row in records),
        "input_tokens": sum(int(row["costs"]["input_tokens"]) for row in records),
        "output_tokens": sum(int(row["costs"]["output_tokens"]) for row in records),
        "latency_ms": sum(int(row["costs"]["latency_ms"]) for row in records),
    }


def run_arm(
    manifest_path: Path,
    *,
    arm: str,
    base_url: str,
    model_id: str,
    revision: str,
    general_scaffold_path: Path,
    regression_scaffold_path: Path,
    output: Path,
) -> dict[str, Any]:
    if arm not in {"h0", "h3"}:
        raise ValueError(f"unknown arm: {arm}")
    manifest = validate_manifest(manifest_path)
    general = general_scaffold_path.read_text().strip()
    regression = regression_scaffold_path.read_text().strip()
    selected = [
        row for row in manifest["cases"]
        if row["block"] == "heldout32" or arm == "h3"
    ]
    served_model_id = base._server_model_id(base_url, expected=model_id)
    ablation_mode = "no_evaluation_loop" if arm == "h0" else "full"
    records: list[dict[str, Any]] = []
    for case in selected:
        instruction = None if arm == "h0" else _case_instruction(case, general, regression)
        config = exact_config(
            int(case["target"]),
            int(case["door_seed"]),
            model_id=model_id,
            base_url=base_url,
            extra_instruction=instruction,
            ablation_mode=ablation_mode,
        )
        optimizer = build_optimizer(config, acceptance_policy=TARGET_FIRST_ACCEPTANCE)
        optimizer.initialize(seed=int(case["door_seed"]))
        pin_registry_schema(optimizer)
        problem = optimizer.factory.env._problem
        endpoints = [
            _interior_neighbor(problem._door1),
            _interior_neighbor(problem._door2),
        ]
        started = time.perf_counter()
        recovery = base._run_probe_optimizer(
            optimizer,
            max_steps=MAX_STEPS,
            truncation_recovery=("none" if arm == "h0" else "thinking_off_handoff"),
            thinking_max_tokens=(None if arm == "h0" else NATIVE_MAX_TOKENS),
            fallback_max_tokens=FALLBACK_MAX_TOKENS,
        )
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        parser_errors = sum(step.message_type in {"PARSE_ERROR", "ERROR"} for step in optimizer.step_records)
        transport_errors = sum(step.message_type == "ERROR" for step in optimizer.step_records)
        unrecovered = max(0, parser_errors - int(recovery["recovered_truncations"]))
        tool_errors = sum(
            not item.get("result", {}).get("success", False)
            for step in optimizer.step_records
            for item in step.tool_results
            if item.get("tool_name") != "_truncated"
        )
        metrics = {
            "door_path": int(optimizer.current_result.metrics.get("door_path", 0)),
            "num_connected_regions": int(optimizer.current_result.metrics.get("num_connected_regions", 0)),
        }
        attached = all(int(optimizer.current_level.data[row, col]) == 0 for row, col in endpoints)
        hard = {
            "doors_connected": metrics["door_path"] > 0,
            "regions_one": metrics["num_connected_regions"] == 1,
            "endpoint_attached": attached,
            "protocol_clean": unrecovered == 0,
            "tool_clean": tool_errors == 0,
        }
        hard["hard_valid"] = all(hard.values())
        error = abs(metrics["door_path"] - int(case["target"]))
        trace = _trace_rows(optimizer)
        first_executable = next((row for row in trace if row["message_type"] == "STEP"), None)
        record = {
            "status": "complete",
            "case_id": case["case_id"],
            "block": case["block"],
            "stratum": case["stratum"],
            "door_seed": int(case["door_seed"]),
            "request_seed": int(case["request_seed"]),
            "target": int(case["target"]),
            "ablation_mode": ablation_mode,
            "scaffold": None if arm == "h0" else case["scaffold"],
            "instruction_sha256": base._sha256_text(instruction) if instruction else None,
            "door_inner_endpoints": [list(value) for value in endpoints],
            "final_level_sha256": base._sha256_text(optimizer.current_level.to_string()),
            "final_metrics": metrics,
            "hard_constraints": hard,
            "absolute_error": error,
            "exact": hard["hard_valid"] and error == 0,
            "first_executable_candidate_exact": bool(
                first_executable
                and first_executable["candidate_metrics_after"]["door_path"] == int(case["target"])
                and first_executable["candidate_metrics_after"]["num_connected_regions"] == 1
            ),
            "parser_errors": parser_errors,
            "unrecovered_parser_errors": unrecovered,
            "transport_errors": transport_errors,
            "tool_errors": tool_errors,
            "truncated_responses": sum(bool(step.truncated) for step in optimizer.step_records),
            "protocol_recovery": recovery,
            "steps": trace,
            "costs": {
                "llm_calls": len(optimizer.step_records),
                "input_tokens": sum(int(step.input_tokens) for step in optimizer.step_records),
                "output_tokens": sum(int(step.output_tokens) for step in optimizer.step_records),
                "latency_ms": elapsed_ms,
                "retries": int(optimizer.llm_retry_count),
            },
        }
        records.append(record)
        base._write_json(output, {"execution_complete": False, "arm": arm, "records": records})

    blocks = {
        name: _aggregate([row for row in records if row["block"] == name])
        for name in ("heldout32", "regression8")
        if any(row["block"] == name for row in records)
    }
    payload = {
        "execution_complete": len(records) == len(selected),
        "contract_gate_pass": (
            len(records) == len(selected)
            and all(row["transport_errors"] == 0 for row in records)
            and all(row["costs"]["llm_calls"] <= MAX_STEPS for row in records)
        ),
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "ablation_mode": ablation_mode,
        "model_id": model_id,
        "revision": revision,
        "server_model_id": served_model_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "planned_denominator": len(selected),
        "budgets": {
            "semantic_actions": MAX_STEPS,
            "max_tool_calls_per_action": MAX_TOOL_CALLS,
            "native_max_tokens": FALLBACK_MAX_TOKENS if arm == "h0" else NATIVE_MAX_TOKENS,
            "fallback_max_tokens": None if arm == "h0" else FALLBACK_MAX_TOKENS,
        },
        "sampling": {"temperature": 0.0, "request_seeded": True, "enable_thinking": True},
        "aggregate": _aggregate(records),
        "blocks": blocks,
        "records": records,
    }
    base._write_json(output, payload)
    if not payload["contract_gate_pass"]:
        raise RuntimeError(f"{EXPERIMENT_ID} {arm} contract failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    materialize = sub.add_parser("materialize-manifest")
    materialize.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--arm", choices=("h0", "h3"), required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--general-scaffold", type=Path, default=GENERAL_SCAFFOLD)
    run.add_argument("--regression-scaffold", type=Path, default=REGRESSION_SCAFFOLD)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize-manifest":
        result = build_manifest()
        base._write_json(args.output, result)
    elif args.command == "validate-manifest":
        result = validate_manifest(args.manifest)
    else:
        result = run_arm(
            args.manifest,
            arm=args.arm,
            base_url=args.base_url,
            model_id=args.model_id,
            revision=args.revision,
            general_scaffold_path=args.general_scaffold,
            regression_scaffold_path=args.regression_scaffold,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

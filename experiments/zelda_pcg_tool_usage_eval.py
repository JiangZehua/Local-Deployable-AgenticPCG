"""Evaluate proactive PCG-tool routing for frozen Zelda design tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from experiments.zelda_bsp_open_prompt_eval import detect_room_boxes


PCG_TOOLS = {
    "generate_zelda_random",
    "generate_zelda_maze",
    "generate_zelda_bsp",
    "generate_zelda_ca",
    "generate_zelda_connect",
    "generate_zelda_digger",
    "generate_zelda_wfc",
}


def trace_calls(path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        event = json.loads(raw)
        if event.get("event") != "step":
            continue
        parsed_calls = (event.get("parsed_message") or {}).get("tool_calls") or []
        results = event.get("tool_results") or []
        for index, call in enumerate(parsed_calls):
            result = results[index].get("result", {}) if index < len(results) else {}
            calls.append(
                {
                    "step": int(event["step_num"]),
                    "index": index,
                    "tool_name": call.get("tool_name"),
                    "parameters": call.get("parameters") or {},
                    "tool_success": bool(result.get("success")),
                    "step_accepted": bool(event.get("accepted")),
                    "accepted_successful_call": bool(event.get("accepted")) and bool(result.get("success")),
                }
            )
    return calls


def _wall_density(level: Sequence[str]) -> float:
    interior = [tile for row in level[1:-1] for tile in row[1:-1]]
    return sum(tile == "#" for tile in interior) / len(interior) if interior else 0.0


def _sequence_status(expected: Sequence[str], names: Sequence[str]) -> dict[str, Any]:
    positions: list[int] = []
    cursor = 0
    for tool in expected:
        try:
            found = names.index(tool, cursor)
        except ValueError:
            break
        positions.append(found)
        cursor = found + 1
    used = set(names)
    return {
        "expected_sequence": list(expected),
        "expected_tools_used": [tool for tool in expected if tool in used],
        "expected_tool_coverage": len(set(expected) & used) / len(set(expected)) if expected else 1.0,
        "all_expected_tools_used": set(expected) <= used,
        "expected_sequence_observed": len(positions) == len(expected),
        "first_call_matches_expected": bool(expected and names and names[0] == expected[0]),
        "first_pcg_matches_expected": bool(
            expected
            and next((name for name in names if name in PCG_TOOLS), None) == expected[0]
        ),
    }


def _constraint_pass(value: Any, constraint: Mapping[str, Any]) -> bool:
    if "eq" in constraint and value != constraint["eq"]:
        return False
    if "min" in constraint and float(value) < float(constraint["min"]):
        return False
    if "max" in constraint and float(value) > float(constraint["max"]):
        return False
    return True


def evaluate_run(task: Mapping[str, Any], label: str, run_dir: Path) -> dict[str, Any]:
    level = (run_dir / "final_level.txt").read_text().splitlines()
    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = summary.get("final_metrics") or {}
    functional = bool(
        metrics.get("players") == 1
        and metrics.get("keys") == 1
        and metrics.get("doors") == 1
        and metrics.get("enemies") == 3
        and metrics.get("regions") == 1
        and float(metrics.get("player_key") or 0) > 0
        and float(metrics.get("key_door") or 0) > 0
    )
    calls = trace_calls(run_dir / "trace.jsonl")
    names = [str(call["tool_name"]) for call in calls]
    pcg_calls = [call for call in calls if call["tool_name"] in PCG_TOOLS]
    structure = {
        "functional_valid": functional,
        "detected_rooms": len(detect_room_boxes(level)),
        "interior_wall_density": _wall_density(level),
        "transition_density": float(metrics.get("transition_density") or 0.0),
        "regions": int(metrics.get("regions") or 0),
        "solution_length": float(metrics.get("solution_length") or 0.0),
    }
    constraints = task.get("outcome_constraints") or {}
    constraint_results = {
        key: _constraint_pass(structure[key], value)
        for key, value in constraints.items()
    }
    routing = _sequence_status(task["expected_sequence"], names)
    routing.update(
        {
            "first_tool": names[0] if names else None,
            "first_pcg_tool": next((name for name in names if name in PCG_TOOLS), None),
            "total_tool_calls": len(calls),
            "pcg_call_count": len(pcg_calls),
            "accepted_pcg_call_count": sum(call["accepted_successful_call"] for call in pcg_calls),
            "tool_counts": dict(sorted(Counter(names).items())),
            "accepted_tool_counts": dict(
                sorted(Counter(call["tool_name"] for call in calls if call["accepted_successful_call"]).items())
            ),
            "calls": calls,
        }
    )
    return {
        "task_id": task["task_id"],
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "routing": routing,
        "structure": structure,
        "outcome_constraint_results": constraint_results,
        "proxy_outcome_success": bool(constraint_results) and all(constraint_results.values()),
        "accepted_steps": int(summary.get("accepted_steps") or 0),
        "rejected_steps": int(summary.get("rejected_steps") or 0),
    }


def aggregate(manifest: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    records = []
    for task in manifest["tasks"]:
        for label in manifest["arms"]:
            records.append(evaluate_run(task, label, run_root / task["task_id"] / label))
    all_tools = [row for row in records if row["label"] == "all_tools"]
    manual = [row for row in records if row["label"] == "manual_only"]
    all_tool_counts: Counter[str] = Counter()
    accepted_tool_counts: Counter[str] = Counter()
    for row in all_tools:
        all_tool_counts.update(row["routing"]["tool_counts"])
        accepted_tool_counts.update(row["routing"]["accepted_tool_counts"])
    return {
        "schema_version": "zelda_pcg_tool_usage_eval.v1",
        "claim_boundary": manifest["claim_boundary"],
        "records": records,
        "all_tools_aggregate": {
            "tasks": len(all_tools),
            "any_expected_tool_used": sum(row["routing"]["expected_tool_coverage"] > 0 for row in all_tools),
            "all_expected_tools_used": sum(row["routing"]["all_expected_tools_used"] for row in all_tools),
            "expected_sequence_observed": sum(row["routing"]["expected_sequence_observed"] for row in all_tools),
            "first_call_matches_expected": sum(row["routing"]["first_call_matches_expected"] for row in all_tools),
            "first_pcg_matches_expected": sum(row["routing"]["first_pcg_matches_expected"] for row in all_tools),
            "mean_expected_tool_coverage": mean(row["routing"]["expected_tool_coverage"] for row in all_tools),
            "functional_valid": sum(row["structure"]["functional_valid"] for row in all_tools),
            "proxy_outcome_success": sum(row["proxy_outcome_success"] for row in all_tools),
            "tool_counts": dict(sorted(all_tool_counts.items())),
            "accepted_tool_counts": dict(sorted(accepted_tool_counts.items())),
            "total_tool_calls": sum(row["routing"]["total_tool_calls"] for row in all_tools),
            "total_pcg_calls": sum(row["routing"]["pcg_call_count"] for row in all_tools),
            "accepted_pcg_calls": sum(row["routing"]["accepted_pcg_call_count"] for row in all_tools),
        },
        "manual_aggregate": {
            "tasks": len(manual),
            "functional_valid": sum(row["structure"]["functional_valid"] for row in manual),
            "proxy_outcome_success": sum(row["proxy_outcome_success"] for row in manual),
            "total_tool_calls": sum(row["routing"]["total_tool_calls"] for row in manual),
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != "zelda_pcg_tool_routing.v1":
        raise ValueError("unsupported manifest schema")
    task_ids = [task["task_id"] for task in payload["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task_id")
    for task in payload["tasks"]:
        if not task.get("instruction") or not task.get("expected_sequence"):
            raise ValueError(f"{task.get('task_id')}: instruction and expected_sequence required")
        instruction = str(task["instruction"]).lower()
        if any(name.lower() in instruction for name in PCG_TOOLS):
            raise ValueError(f"{task['task_id']}: prompt leaks an exact tool name")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    payload = aggregate(manifest, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

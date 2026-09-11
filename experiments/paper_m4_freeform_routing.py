"""Paper M4: confirmatory free-form routing to Zelda PCG macros."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from experiments.zelda_bsp_open_prompt_eval import detect_room_boxes
from experiments.zelda_pcg_tool_usage_eval import PCG_TOOLS, trace_calls


SCHEMA_VERSION = "paper-m4-freeform-routing.v1"


def _wall_density(level: Sequence[str]) -> float:
    interior = [tile for row in level[1:-1] for tile in row[1:-1]]
    return sum(tile == "#" for tile in interior) / len(interior) if interior else 0.0


def _contains_subsequence(expected: Sequence[str], observed: Sequence[str]) -> bool:
    cursor = 0
    for name in observed:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _route_status(expected: Sequence[str], calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    called = [str(call["tool_name"]) for call in calls]
    successful = [
        str(call["tool_name"]) for call in calls if call["tool_success"]
    ]
    accepted = [
        str(call["tool_name"])
        for call in calls
        if call["accepted_successful_call"]
    ]
    called_pcg = [name for name in called if name in PCG_TOOLS]
    successful_pcg = [name for name in successful if name in PCG_TOOLS]
    accepted_pcg = [name for name in accepted if name in PCG_TOOLS]
    expected_set = set(expected)
    accepted_set = set(accepted_pcg)
    return {
        "expected_sequence": list(expected),
        "called_sequence_observed": _contains_subsequence(expected, called),
        "successful_sequence_observed": _contains_subsequence(expected, successful),
        "accepted_sequence_observed": _contains_subsequence(expected, accepted),
        "accepted_expected_coverage": (
            len(expected_set & accepted_set) / len(expected_set) if expected_set else 1.0
        ),
        "first_called_pcg": called_pcg[0] if called_pcg else None,
        "first_successful_pcg": successful_pcg[0] if successful_pcg else None,
        "first_accepted_pcg": accepted_pcg[0] if accepted_pcg else None,
        "first_called_pcg_matches": bool(expected and called_pcg and called_pcg[0] == expected[0]),
        "first_accepted_pcg_matches": bool(
            expected and accepted_pcg and accepted_pcg[0] == expected[0]
        ),
        "called_pcg_count": len(called_pcg),
        "successful_pcg_count": len(successful_pcg),
        "accepted_pcg_count": len(accepted_pcg),
        "called_tool_counts": dict(sorted(Counter(called).items())),
        "successful_tool_counts": dict(sorted(Counter(successful).items())),
        "accepted_tool_counts": dict(sorted(Counter(accepted).items())),
        "calls": list(calls),
    }


def _constraint_pass(value: Any, constraint: Mapping[str, Any]) -> bool:
    if "eq" in constraint and value != constraint["eq"]:
        return False
    if "min" in constraint and float(value) < float(constraint["min"]):
        return False
    if "max" in constraint and float(value) > float(constraint["max"]):
        return False
    return True


def evaluate_run(task: Mapping[str, Any], arm: str, run_dir: Path) -> dict[str, Any]:
    missing = [name for name in ("final_level.txt", "summary.json", "trace.jsonl")
               if not (run_dir / name).is_file()]
    level = (run_dir / "final_level.txt").read_text().splitlines() if not missing else []
    summary = json.loads((run_dir / "summary.json").read_text()) if not missing else {}
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
    structure = {
        "functional_valid": functional,
        "detected_rooms": len(detect_room_boxes(level)) if level else 0,
        "interior_wall_density": _wall_density(level) if level else 0.0,
        "transition_density": float(metrics.get("transition_density") or 0.0),
        "regions": int(metrics.get("regions") or 0),
        "solution_length": float(metrics.get("solution_length") or 0.0),
    }
    constraint_results = {
        name: _constraint_pass(structure[name], constraint)
        for name, constraint in (task.get("outcome_constraints") or {}).items()
    }
    proxy_success = bool(constraint_results) and all(constraint_results.values())
    calls = trace_calls(run_dir / "trace.jsonl") if not missing else []
    routing = _route_status(task["expected_sequence"], calls)
    macro_route = bool(routing["accepted_sequence_observed"])
    manual_equivalent = bool(
        task.get("manual_equivalent_allowed")
        and not macro_route
        and functional
        and proxy_success
    )
    routing.update({
        "accepted_macro_route": macro_route,
        "verified_manual_equivalent": manual_equivalent,
        "expected_or_equivalent_route": macro_route or manual_equivalent,
    })
    return {
        "artifact_present": not missing,
        "missing_artifacts": missing,
        "task_id": task["task_id"],
        "family_id": task["family_id"],
        "paraphrase_id": task["paraphrase_id"],
        "seed": int(task["seed"]),
        "arm": arm,
        "run_dir": str(run_dir.resolve()),
        "routing": routing,
        "structure": structure,
        "outcome_constraint_results": constraint_results,
        "proxy_outcome_success": proxy_success,
        "accepted_steps": int(summary.get("accepted_steps") or 0),
        "rejected_steps": int(summary.get("rejected_steps") or 0),
    }


def aggregate(manifest: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    records = [
        evaluate_run(task, arm, run_root / task["task_id"] / arm)
        for task in manifest["tasks"]
        for arm in manifest["arms"]
    ]
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in manifest["arms"]:
        rows = [row for row in records if row["arm"] == arm]
        by_arm[arm] = {
            "cases": len(rows),
            "functional_valid": sum(row["structure"]["functional_valid"] for row in rows),
            "proxy_outcome_success": sum(row["proxy_outcome_success"] for row in rows),
            "accepted_macro_route": sum(row["routing"]["accepted_macro_route"] for row in rows),
            "expected_or_equivalent_route": sum(
                row["routing"]["expected_or_equivalent_route"] for row in rows
            ),
            "first_called_pcg_matches": sum(
                row["routing"]["first_called_pcg_matches"] for row in rows
            ),
            "first_accepted_pcg_matches": sum(
                row["routing"]["first_accepted_pcg_matches"] for row in rows
            ),
            "called_pcg_calls": sum(row["routing"]["called_pcg_count"] for row in rows),
            "successful_pcg_calls": sum(
                row["routing"]["successful_pcg_count"] for row in rows
            ),
            "accepted_pcg_calls": sum(row["routing"]["accepted_pcg_count"] for row in rows),
            "mean_accepted_expected_coverage": mean(
                row["routing"]["accepted_expected_coverage"] for row in rows
            ),
        }
    paired = []
    for task in manifest["tasks"]:
        rows = {
            row["arm"]: row for row in records if row["task_id"] == task["task_id"]
        }
        all_tools = rows["all_tools"]
        manual = rows["manual_only"]
        paired.append({
            "task_id": task["task_id"],
            "family_id": task["family_id"],
            "all_tools_functional": all_tools["structure"]["functional_valid"],
            "manual_functional": manual["structure"]["functional_valid"],
            "all_tools_proxy": all_tools["proxy_outcome_success"],
            "manual_proxy": manual["proxy_outcome_success"],
            "all_tools_accepted_macro": all_tools["routing"]["accepted_macro_route"],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "claim_boundary": manifest["claim_boundary"],
        "records": records,
        "aggregate_by_arm": by_arm,
        "paired": paired,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    tasks = payload.get("tasks") or []
    if len(tasks) != 18:
        raise ValueError("paper M4 requires exactly 18 tasks")
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task_id")
    family_counts = Counter(task["family_id"] for task in tasks)
    if len(family_counts) != 6 or set(family_counts.values()) != {3}:
        raise ValueError("paper M4 requires six families with three paraphrases each")
    for task in tasks:
        instruction = str(task.get("instruction") or "").lower()
        if not instruction or not task.get("expected_sequence"):
            raise ValueError(f"{task.get('task_id')}: missing instruction/route")
        if any(name.lower() in instruction for name in PCG_TOOLS):
            raise ValueError(f"{task['task_id']}: exact tool name leaked")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(load_manifest(args.manifest), args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

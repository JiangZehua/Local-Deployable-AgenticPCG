"""PAPER-M7 fixed-RNG routing evaluation and CPU keyword baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "paper-m7-fixed-rng36.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Fixed-RNG-36 schema")
    tasks = payload.get("tasks") or []
    if len(tasks) != 18:
        raise ValueError("Fixed-RNG-36 requires exactly 18 tasks")
    task_ids = [task["task_id"] for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id")
    counts = Counter(task["family_id"] for task in tasks)
    if len(counts) != 6 or set(counts.values()) != {3}:
        raise ValueError("expected six families with three paraphrases")
    for family_id in counts:
        rows = [task for task in tasks if task["family_id"] == family_id]
        if {task["paraphrase_id"] for task in rows} != {"p1", "p2", "p3"}:
            raise ValueError(f"{family_id}: expected unique p1/p2/p3 paraphrases")
        if len({int(task["seed"]) for task in rows}) != 1:
            raise ValueError(f"{family_id}: paraphrases do not share one RNG seed")
        initials = {task.get("initial", payload["default_initial"]) for task in rows}
        if len(initials) != 1:
            raise ValueError(f"{family_id}: paraphrases do not share one initial map")
    if set(payload.get("arms") or {}) != {"all_tools", "manual_only"}:
        raise ValueError("paired all_tools/manual_only arms required")
    return payload


def keyword_route(instruction: str) -> list[str]:
    text = instruction.lower()
    if any(token in text for token in ("disconnected", "split dungeon", "互不连通", "连通区域", "connecting passages")):
        return ["generate_zelda_connect"]
    if any(token in text for token in ("repeating", "recur", "neighborhood patterns", "局部墙面纹样", "风格统一")):
        return ["generate_zelda_wfc"]
    if any(token in text for token in ("noise", "noisy", "smooth", "rounding", "平滑", "带噪声", "圆润洞穴")):
        return ["generate_zelda_random", "generate_zelda_ca", "generate_zelda_connect"]
    if any(token in text for token in ("maze", "labyrinth", "迷宫", "carve", "开辟通路")):
        return ["generate_zelda_random", "generate_zelda_maze"]
    if any(token in text for token in ("cave", "cavern", "洞穴", "洞室", "meandering", "弯曲")):
        return ["generate_zelda_digger"]
    if any(token in text for token in ("room", "chamber", "房间", "矩形")):
        return ["generate_zelda_bsp"]
    return []


def keyword_baseline(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for task in manifest["tasks"]:
        predicted = keyword_route(task["instruction"])
        expected = list(task["expected_sequence"])
        records.append({
            "task_id": task["task_id"],
            "family_id": task["family_id"],
            "expected_sequence": expected,
            "predicted_sequence": predicted,
            "route_exact": predicted == expected,
        })
    return {
        "baseline": "ordered-keyword-router-v1",
        "aggregate": {
            "cases": len(records),
            "route_exact": sum(row["route_exact"] for row in records),
        },
        "records": records,
    }


def preflight(manifest_path: Path, output: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    baseline = keyword_baseline(manifest)
    family_contract = {}
    for task in manifest["tasks"]:
        family_contract.setdefault(task["family_id"], {
            "seed": int(task["seed"]),
            "initial": task.get("initial", manifest["default_initial"]),
        })
    result = {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha256(manifest_path),
        "episodes": len(manifest["tasks"]) * len(manifest["arms"]),
        "family_rng_and_initial_contract": family_contract,
        "keyword_baseline": baseline,
        "gate_pass": True,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _costs(run_dir: Path, calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    return {
        "agent_steps": int(summary.get("total_steps") or 0),
        "accepted_steps": int(summary.get("accepted_steps") or 0),
        "rejected_steps": int(summary.get("rejected_steps") or 0),
        "trace_tool_calls": len(calls),
        "input_tokens": int(summary.get("total_input_tokens") or 0),
        "output_tokens": int(summary.get("total_output_tokens") or 0),
        "total_tokens": int(summary.get("total_tokens") or 0),
        "agent_step_time_s": float(summary.get("total_time") or 0.0),
    }


def _entity_positions(level: Sequence[str]) -> dict[str, list[tuple[int, int]]]:
    return {
        tile: sorted(
            (x, y)
            for y, row in enumerate(level)
            for x, observed in enumerate(row)
            if observed == tile
        )
        for tile in ("P", "K", "D", "E")
    }


def _strict_reconnect_equivalent(
    task: Mapping[str, Any], run_dir: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a manual reconnect from raw trace rather than outcome proxies."""

    audit = {
        "eligible_family": task["family_id"] == "reconnect_regions",
        "no_accepted_expected_macro": not record["routing"]["accepted_macro_route"],
        "initial_regions_gt_one": False,
        "final_regions_eq_one": record["structure"]["regions"] == 1,
        "functional_valid": bool(record["structure"]["functional_valid"]),
        "entity_coordinates_preserved": False,
        "accepted_calls_place_tile_only": False,
        "accepted_diffs_wall_to_empty_only": False,
        "accepted_diff_count": 0,
    }
    initial_level: list[str] | None = None
    accepted_calls = 0
    accepted_diffs: list[Mapping[str, Any]] = []
    accepted_tool_names: list[str] = []
    if not (run_dir / "trace.jsonl").is_file():
        return {**audit, "verified": False}
    for raw in (run_dir / "trace.jsonl").read_text().splitlines():
        event = json.loads(raw)
        if event.get("event") == "initial_level":
            initial_level = str(event["level"]).splitlines()
            audit["initial_regions_gt_one"] = int((event.get("metrics") or {}).get("regions") or 0) > 1
        if event.get("event") != "step" or not event.get("accepted"):
            continue
        parsed_calls = (event.get("parsed_message") or {}).get("tool_calls") or []
        results = event.get("tool_results") or []
        for index, call in enumerate(parsed_calls):
            result = results[index].get("result", {}) if index < len(results) else {}
            if not result.get("success"):
                continue
            accepted_calls += 1
            accepted_tool_names.append(str(call.get("tool_name")))
            accepted_diffs.extend((result.get("data") or {}).get("diff") or [])

    final_level = (run_dir / "final_level.txt").read_text().splitlines()
    audit["entity_coordinates_preserved"] = bool(
        initial_level is not None
        and _entity_positions(initial_level) == _entity_positions(final_level)
    )
    audit["accepted_calls_place_tile_only"] = bool(
        accepted_calls and all(name == "place_tile" for name in accepted_tool_names)
    )
    audit["accepted_diff_count"] = len(accepted_diffs)
    audit["accepted_diffs_wall_to_empty_only"] = bool(
        accepted_diffs
        and all(diff.get("old") == "wall" and diff.get("new") == "empty" for diff in accepted_diffs)
    )
    audit["verified"] = all(
        value for key, value in audit.items()
        if key not in {"accepted_diff_count", "verified"}
    )
    return audit


def evaluate(manifest_path: Path, run_root: Path, output: Path,
             run_paths: Mapping[str, Path] | None = None) -> dict[str, Any]:
    from experiments.paper_m4_freeform_routing import evaluate_run

    manifest = load_manifest(manifest_path)
    records = []
    for task in manifest["tasks"]:
        for arm in manifest["arms"]:
            run_dir = (run_paths or {}).get(task["task_id"] + "__" + arm,
                                             run_root / task["task_id"] / arm)
            record = evaluate_run(task, arm, run_dir)
            reconnect_audit = _strict_reconnect_equivalent(task, run_dir, record)
            record["routing"]["legacy_proxy_manual_equivalent"] = bool(
                record["routing"].get("verified_manual_equivalent")
            )
            record["routing"]["manual_equivalent_audit"] = reconnect_audit
            record["routing"]["verified_manual_equivalent"] = reconnect_audit["verified"]
            record["routing"]["expected_or_equivalent_route"] = bool(
                record["routing"]["accepted_macro_route"] or reconnect_audit["verified"]
            )
            record["costs"] = _costs(run_dir, record["routing"]["calls"])
            record["rng_contract"] = {
                "request_seed": int(task["seed"]),
                "python_process_seed": int(task["seed"]),
                "numpy_process_seed": int(task["seed"]),
                "initial": task.get("initial", manifest["default_initial"]),
            }
            records.append(record)

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in manifest["arms"]:
        rows = [row for row in records if row["arm"] == arm]
        by_arm[arm] = {
            "cases": len(rows),
            "functional_valid": sum(row["structure"]["functional_valid"] for row in rows),
            "proxy_outcome_success": sum(row["proxy_outcome_success"] for row in rows),
            "accepted_macro_route": sum(row["routing"]["accepted_macro_route"] for row in rows),
            "expected_or_equivalent_route": sum(row["routing"]["expected_or_equivalent_route"] for row in rows),
            "first_accepted_pcg_matches": sum(row["routing"]["first_accepted_pcg_matches"] for row in rows),
            "agent_steps": sum(row["costs"]["agent_steps"] for row in rows),
            "trace_tool_calls": sum(row["costs"]["trace_tool_calls"] for row in rows),
            "pcg_macro_calls": sum(row["routing"]["called_pcg_count"] for row in rows),
            "total_tokens": sum(row["costs"]["total_tokens"] for row in rows),
            "mean_agent_step_time_s": mean(row["costs"]["agent_step_time_s"] for row in rows),
        }

    family_rows: dict[str, dict[str, Any]] = {}
    for family_id in sorted({task["family_id"] for task in manifest["tasks"]}):
        rows = [row for row in records if row["arm"] == "all_tools" and row["family_id"] == family_id]
        family_rows[family_id] = {
            "cases": len(rows),
            "accepted_macro_route": sum(row["routing"]["accepted_macro_route"] for row in rows),
            "expected_or_equivalent_route": sum(row["routing"]["expected_or_equivalent_route"] for row in rows),
            "functional_valid": sum(row["structure"]["functional_valid"] for row in rows),
            "proxy_outcome_success": sum(row["proxy_outcome_success"] for row in rows),
            "seed": rows[0]["seed"],
        }

    all_tools = by_arm["all_tools"]
    manual = by_arm["manual_only"]
    tool_call_reduction = (
        (manual["trace_tool_calls"] - all_tools["trace_tool_calls"]) / manual["trace_tool_calls"]
        if manual["trace_tool_calls"] else 0.0
    )
    expected_keys = {
        (task["task_id"], arm)
        for task in manifest["tasks"]
        for arm in manifest["arms"]
    }
    observed_keys = {(row["task_id"], row["arm"]) for row in records if row.get("artifact_present", True)}
    artifact_completeness = {
        "expected_episode_keys": len(expected_keys),
        "observed_episode_keys": len(observed_keys),
        "missing_episode_keys": sorted(f"{task_id}:{arm}" for task_id, arm in expected_keys - observed_keys),
        "extra_episode_keys": sorted(f"{task_id}:{arm}" for task_id, arm in observed_keys - expected_keys),
    }
    criteria = {
        "all_36_episodes_present": observed_keys == expected_keys and len(records) == 36,
        "appropriate_or_equivalent_at_least_15_of_18": all_tools["expected_or_equivalent_route"] >= 15,
        "at_least_2_of_3_each_family": all(row["expected_or_equivalent_route"] >= 2 for row in family_rows.values()),
        "all_tools_functional_18_of_18": all_tools["functional_valid"] == 18,
        "trace_tool_calls_reduced_at_least_25_percent": tool_call_reduction >= 0.25,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha256(manifest_path),
        "claim_boundary": manifest["claim_boundary"],
        "criteria": criteria,
        "scientific_gate_pass": all(criteria.values()),
        "trace_tool_call_reduction": tool_call_reduction,
        "artifact_completeness": artifact_completeness,
        "aggregate_by_arm": by_arm,
        "aggregate_by_family": family_rows,
        "keyword_baseline": keyword_baseline(manifest),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path)
    run = sub.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.manifest, args.output)
    else:
        result = evaluate(args.manifest, args.run_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

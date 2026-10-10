"""PAPER-M8-SMB-DYNAMIC-BUILDERS: stitching vs search vs legacy library lookup.

CPU-only.  Arms resolve every exact target of the expanded dynamic envelope
(enemies 0..4 x coins 0..5 x jumps 0..6 = 210) through the unchanged
FunctionalResolver contract.  The language replay re-resolves the recorded
PAPER-M6 parser payloads (no new model calls: the model only extracts
targets and never sees builder output) with the new builder.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from experiments.smb_patch_library import _signature
from toolusepcg.functional_resolver import FunctionalSpec, MetricConstraint


RUN_ID = "PAPER-M8-SMB-DYNAMIC-BUILDERS"
REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS: dict[str, dict[str, Any]] = {
    "legacy_r108": {"budget": 64},
    "stitch": {"budget": 16},
    "search_b64": {"budget": 64},
    "search_b256": {"budget": 256},
    "stitch_then_search": {"budget": 64},
    # PAPER-M8b: ceiling-free patches, zero hidden takeoffs required
    "stitch_v2": {"budget": 16},
    "search_v2_b64": {"budget": 64},
    "search_v2_b256": {"budget": 256},
    "stitch_then_search_v2": {"budget": 64},
}


def arm_mode_contract(arm: str) -> tuple[str, str]:
    contract = "v2" if arm.endswith("_v2") or "_v2_" in arm else "v1"
    base = arm.replace("_v2", "")
    mode = "search" if base.startswith("search") else base
    return mode, contract
DETERMINISM_REPEATS = 2
# Mac M6 typed-exact failures (MAC_M6_READOUT.md); both stopped before the
# resolver, so the other 22 valid cases parsed to the manifest target.
MAC_PARSE_FAILURES = ("natural-en-08", "natural-zh-17")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def exact_spec(target: Mapping[str, int]) -> FunctionalSpec:
    return FunctionalSpec(
        "smb_dynamic",
        {name: MetricConstraint("exact", int(target[name])) for name in ("enemies", "coins", "jumps")},
    )


_RESOLVER = None
_ARM = None


def _resolver_for(arm: str):
    global _RESOLVER, _ARM
    if _RESOLVER is None or _ARM != arm:
        from experiments.smb_dynamic_builders import build_resolver
        mode, contract = arm_mode_contract(arm)
        _RESOLVER = build_resolver(mode, contract=contract)
        _ARM = arm
    return _RESOLVER


def _adapter(resolver):
    return resolver._adapters["smb_dynamic"]


def resolve_record(arm: str, spec: FunctionalSpec) -> dict[str, Any]:
    started = time.perf_counter()
    if arm == "legacy_r108":
        # The legacy adapter keeps a per-instance discovery cache; rebuild per
        # case exactly as PAPER-M6 did.
        from experiments.r108_smb_functional_harness import build_resolver as legacy
        resolver = legacy()
    else:
        resolver = _resolver_for(arm)
    resolution = resolver.resolve(spec, allow_search=True, max_evaluator_calls=ARMS[arm]["budget"])
    payload = resolution.as_dict(include_level=True)
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    payload["width"] = resolution.level.width if resolution.level is not None else None
    if resolution.status == "exact" and resolution.level is not None:
        adapter = _adapter(resolver)
        signatures = {_signature(adapter.evaluate(resolution.level)) for _ in range(DETERMINISM_REPEATS)}
        payload["determinism_repeats"] = DETERMINISM_REPEATS
        payload["deterministic"] = len(signatures) == 1
    return payload


def _grid_job(job: tuple[str, dict[str, int]]) -> dict[str, Any]:
    arm, target = job
    record = resolve_record(arm, exact_spec(target))
    search = record.get("search", {})
    return {
        "arm": arm,
        "target": target,
        "status": record["status"],
        "exact": record["status"] == "exact",
        "deterministic": record.get("deterministic"),
        "source": record["source"],
        "strategy": record["strategy"],
        "evaluator_calls": record["evaluator_calls"],
        "wall_seconds": record["wall_seconds"],
        "width": record["width"],
        "level_sha256": record["level_sha256"],
        "level": record.get("level"),
        "stitch_attempts": len(search.get("stitch_attempts", [])),
        "stitch_first_try": search.get("stitch_first_try"),
        "search_calls": search.get("search_calls"),
        "provenance": record["provenance"],
        "observed": {k: (record["observed_metrics"] or {}).get(k) for k in ("enemies", "coins", "jumps")}
        if record["observed_metrics"] else None,
        "contract": arm_mode_contract(arm)[1] if arm != "legacy_r108" else "legacy",
    }


def run_grid(arms: list[str], output: Path, workers: int) -> dict[str, Any]:
    from experiments.smb_dynamic_builders import envelope_targets

    targets = envelope_targets()
    jobs = [(arm, target) for arm in arms for target in targets]
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_grid_job, jobs, chunksize=1))
    else:
        rows = [_grid_job(job) for job in jobs]
    result = {
        "run_id": RUN_ID,
        "experiment": "structured-envelope-grid",
        "code_commit": _git_commit(),
        "arms": {arm: ARMS[arm] for arm in arms},
        "targets": len(targets),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "records": rows,
    }
    _write_json(output, result)
    return result


def _replay_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    from experiments.r108_smb_functional_harness import (
        _canonical_parser_payload,
        derive_parser_disposition,
        is_fail_closed,
        parser_payload_to_spec,
    )

    arm, row = job
    parsed = row["raw_parser_payload"]
    disposition = derive_parser_disposition(parsed)
    resolution = None
    if disposition["status"] == "resolved":
        resolution = resolve_record(arm, parser_payload_to_spec(parsed))
    expected_status = row["expected"]["status"]
    valid_target = bool(row.get("role") == "valid_target"
                        or (row.get("role", "legacy") == "legacy" and expected_status == "resolved"))
    functional_exact = bool(valid_target and row["parse_exact"] and resolution
                            and resolution["status"] == "exact")
    fail_closed = bool(not valid_target and is_fail_closed(
        row["expected"]["status"], disposition["status"], resolution))
    return {
        "arm": arm,
        "case_id": row["case_id"],
        "role": row.get("role", "legacy"),
        "valid_target": valid_target,
        "recorded_disposition": row["disposition"]["status"],
        "replayed_disposition": disposition["status"],
        "disposition_unchanged": disposition["status"] == row["disposition"]["status"],
        "parse_exact": row["parse_exact"],
        "functional_exact": functional_exact,
        "fail_closed": fail_closed,
        "false_resolve": bool(not valid_target and disposition["status"] == "resolved"),
        "recorded_functional_exact": row["functional_exact"],
        "resolution": None if resolution is None else {
            k: resolution.get(k) for k in ("status", "source", "strategy", "selected_target",
                                            "evaluator_calls", "level_sha256", "width",
                                            "deterministic", "level")
        },
        "canonical_parsed": _canonical_parser_payload(parsed, status=disposition["status"]),
    }


def run_replay(records_path: Path, manifest_path: Path | None, arms: list[str], output: Path,
               workers: int) -> dict[str, Any]:
    recorded = json.loads(records_path.read_text())
    manifest = json.loads(manifest_path.read_text()) if manifest_path is not None else {"cases": []}
    rows = recorded["records"]
    jobs = [(arm, row) for arm in arms for row in rows]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            out = list(executor.map(_replay_job, jobs, chunksize=1))
    else:
        out = [_replay_job(job) for job in jobs]
    # Mac deployment: resolved specs are the manifest targets for every
    # typed-exact valid case; replay those targets directly.
    mac_jobs = [] if manifest_path is None else [
        (arm, case) for arm in arms for case in manifest["cases"]
        if case["role"] == "valid_target" and case["case_id"] not in MAC_PARSE_FAILURES
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mac_rows = list(executor.map(_mac_job, mac_jobs, chunksize=1))
    else:
        mac_rows = [_mac_job(job) for job in mac_jobs]
    summary = {}
    for arm in arms:
        valid = [r for r in out if r["arm"] == arm and r["valid_target"]]
        guards = [r for r in out if r["arm"] == arm and not r["valid_target"]]
        mac = [r for r in mac_rows if r["arm"] == arm]
        summary[arm] = {
            "server_valid_cases": len(valid),
            "server_parse_exact": sum(r["parse_exact"] for r in valid),
            "server_functional_exact": sum(r["functional_exact"] for r in valid),
            "guards": len(guards),
            "guards_fail_closed": sum(r["fail_closed"] for r in guards),
            "false_resolve": sum(r["false_resolve"] for r in guards),
            "dispositions_unchanged": all(r["disposition_unchanged"] for r in out if r["arm"] == arm),
        }
        if manifest_path is not None:
            summary[arm].update({
                "mac_valid_cases": 24,
                "mac_typed_exact": 24 - len(MAC_PARSE_FAILURES),
                "mac_functional_exact": sum(r["exact"] for r in mac),
            })
    result = {
        "run_id": RUN_ID,
        "experiment": "language-replay",
        "code_commit": _git_commit(),
        "records_path": str(records_path),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "split": recorded.get("split"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None,
        "mac_parse_failures": list(MAC_PARSE_FAILURES) if manifest_path else [],
        "summary": summary,
        "records": out,
        "mac_records": mac_rows,
    }
    _write_json(output, result)
    return result


def _mac_job(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    from experiments.r108_smb_functional_harness import parser_payload_to_spec

    arm, case = job
    record = resolve_record(arm, parser_payload_to_spec(case["expected"]))
    return {"arm": arm, "case_id": case["case_id"], "exact": record["status"] == "exact",
            "status": record["status"], "evaluator_calls": record["evaluator_calls"],
            "level_sha256": record["level_sha256"], "width": record["width"]}


def _audit_job(row: dict[str, Any]) -> dict[str, Any]:
    from toolusepcg.level import Level
    from toolusepcg.smb_patches import WidthEvaluatorPool

    result = WidthEvaluatorPool().evaluate_audited(Level.from_string(row["level"], problem_type="smb"))
    audit = result.info["takeoff_audit"]
    return {"arm": row["arm"], "target": row["target"], "level_sha256": row["level_sha256"],
            "takeoffs": audit["takeoffs"], "jump_events": audit["jump_events"],
            "hidden_takeoffs": audit["hidden_takeoffs"]}


def run_audit(grid_path: Path, output: Path, workers: int) -> dict[str, Any]:
    """Post-hoc takeoff audit of every successful level in an existing grid."""
    grid = json.loads(grid_path.read_text())
    rows = [r for r in grid["records"] if r["exact"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            out = list(executor.map(_audit_job, rows, chunksize=2))
    else:
        out = [_audit_job(r) for r in rows]
    summary = {}
    for arm in grid["arms"]:
        arm_rows = [r for r in out if r["arm"] == arm]
        summary[arm] = {"successes": len(arm_rows),
                        "clean": sum(r["hidden_takeoffs"] == 0 for r in arm_rows),
                        "with_hidden_takeoffs": sum(r["hidden_takeoffs"] > 0 for r in arm_rows),
                        "hidden_takeoffs_total": sum(r["hidden_takeoffs"] for r in arm_rows)}
    result = {"run_id": RUN_ID, "experiment": "post-hoc-takeoff-audit", "code_commit": _git_commit(),
              "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
              "summary": summary, "records": out}
    _write_json(output, result)
    return result


def summarize(grid_path: Path, replay_path: Path | None, output: Path) -> dict[str, Any]:
    from experiments.r107_smb_mechanic_library import BACKBONES

    grid = json.loads(grid_path.read_text())
    covered = {tuple(v["behavior"]) for v in BACKBONES.values()}
    arms: dict[str, Any] = {}
    for arm in grid["arms"]:
        rows = [r for r in grid["records"] if r["arm"] == arm]
        old = [r for r in rows if (r["target"]["enemies"], r["target"]["jumps"]) in covered]
        new = [r for r in rows if (r["target"]["enemies"], r["target"]["jumps"]) not in covered]
        exact = [r for r in rows if r["exact"]]
        calls = sorted(r["evaluator_calls"] for r in exact)
        arms[arm] = {
            "exact": len(exact),
            "targets": len(rows),
            "exact_on_legacy_kj": f"{sum(r['exact'] for r in old)}/{len(old)}",
            "exact_on_new_kj": f"{sum(r['exact'] for r in new)}/{len(new)}",
            "deterministic_successes": sum(bool(r["deterministic"]) for r in exact),
            "stitch_first_try": sum(bool(r["stitch_first_try"]) for r in rows),
            "median_calls_success": calls[len(calls) // 2] if calls else None,
            "max_calls_success": calls[-1] if calls else None,
            "total_calls": sum(r["evaluator_calls"] for r in rows),
            "width_range_success": [min(r["width"] for r in exact), max(r["width"] for r in exact)]
            if exact else None,
            "strategies": sorted({r["strategy"] for r in exact if r["strategy"]}),
            "misses": [r["target"] for r in rows if not r["exact"]],
        }
    dev = json.loads((REPO_ROOT / "benchmarks/smb/targets.json").read_text())
    dev_targets = [{k: int(t["target"][k]) for k in ("enemies", "coins", "jumps")} for t in dev["targets"]]
    for arm in grid["arms"]:
        rows = [r for r in grid["records"] if r["arm"] == arm and r["target"] in dev_targets]
        arms[arm]["dev29_exact"] = f"{sum(r['exact'] for r in rows)}/{len(dev_targets)}"
    result = {"run_id": RUN_ID, "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
              "code_commit": grid["code_commit"], "arms": arms}
    if replay_path is not None:
        replay = json.loads(replay_path.read_text())
        result["language_replay"] = replay["summary"]
        result["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--workers", type=int, default=1)
    replay = sub.add_parser("replay")
    replay.add_argument("--records", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, help="M6 manifest; enables the Mac replay")
    replay.add_argument("--arms", nargs="+", default=["legacy_r108", "stitch", "stitch_then_search"],
                        choices=list(ARMS))
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--workers", type=int, default=1)
    audit = sub.add_parser("audit")
    audit.add_argument("--grid", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--workers", type=int, default=1)
    summary = sub.add_parser("summary")
    summary.add_argument("--grid", type=Path, required=True)
    summary.add_argument("--replay", type=Path)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "grid":
        result = run_grid(args.arms, args.output, args.workers)
        print(json.dumps({"records": len(result["records"]), "wall_seconds": result["wall_seconds"]}))
    elif args.command == "audit":
        print(json.dumps(run_audit(args.grid, args.output, args.workers)["summary"], indent=2))
    elif args.command == "replay":
        result = run_replay(args.records, args.manifest, args.arms, args.output, args.workers)
        print(json.dumps(result["summary"], indent=2))
    else:
        print(json.dumps(summarize(args.grid, args.replay, args.output)["arms"], indent=2, default=str))


if __name__ == "__main__":
    main()

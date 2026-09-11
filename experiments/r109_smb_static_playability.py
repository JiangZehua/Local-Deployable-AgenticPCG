"""R109 deterministic playability gate for every SMB-static target case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.paper_m1_smb_language import parser_payload_to_spec
from experiments.r061_qwen38_m1 import _write_json
from experiments.r098_smb_static_baseline import build_cases
from experiments.r108_smb_functional_harness import build_resolver
from toolusepcg.functional_resolver import FunctionalSpec, MetricConstraint
from toolusepcg.smb_static_metrics import STATIC_TARGET_METRICS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "benchmarks/smb/static_language.json"
)
RUN_ID = "R109-SMB-STATIC-PLAYABILITY-GATE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_resolution(case_id: str, suite: str, spec: FunctionalSpec) -> dict[str, Any]:
    resolution = build_resolver().resolve(
        spec, allow_search=True, max_evaluator_calls=4
    )
    payload = resolution.as_dict(include_level=True)
    metrics = payload.get("observed_metrics") or {}
    info = payload.get("evaluation_info") or {}
    selected = payload.get("selected_target") or {}
    counts_exact = all(
        int(metrics.get(name, -1)) == int(selected.get(name, -2))
        for name in STATIC_TARGET_METRICS
    )
    coins_collected = int(metrics.get("coin_tiles_collected", -1))
    coin_tiles = int(metrics.get("coin_tile_count", -2))
    passed = bool(
        payload["status"] == "exact"
        and payload.get("level")
        and counts_exact
        and int(metrics.get("tube_issues", -1)) == 0
        and int(metrics.get("unsupported_pipe_count", -1)) == 0
        and int(metrics.get("unsupported_enemy_count", -1)) == 0
        and info.get("evaluation_mode") == "static_grid_plus_gameplay"
        and info.get("simulation_called") is True
        and info.get("engine_executed") is True
        and float(metrics.get("complete", 0.0)) == 1.0
        and coins_collected == coin_tiles
        and int(metrics.get("trajectory_steps", 0)) > 0
    )
    return {
        "case_id": case_id,
        "suite": suite,
        "passed": passed,
        "selected_target": selected,
        "observed_counts": {
            name: int(metrics.get(name, -1)) for name in STATIC_TARGET_METRICS
        },
        "tube_issues": int(metrics.get("tube_issues", -1)),
        "unsupported_pipe_count": int(metrics.get("unsupported_pipe_count", -1)),
        "unsupported_enemy_count": int(metrics.get("unsupported_enemy_count", -1)),
        "simulation_called": info.get("simulation_called"),
        "engine_executed": info.get("engine_executed"),
        "complete": float(metrics.get("complete", 0.0)),
        "coin_tiles_collected": coins_collected,
        "engine_total_coins": int(metrics.get("coins_collected", -1)),
        "coin_tile_count": coin_tiles,
        "trajectory_steps": int(metrics.get("trajectory_steps", 0)),
        "evaluator_calls": int(payload["evaluator_calls"]),
        "level_sha256": payload.get("level_sha256"),
        "level": payload.get("level"),
        "errors": payload.get("errors", []),
    }


def run(output: Path, manifest: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    frozen = json.loads(manifest.read_text())
    records: list[dict[str, Any]] = []

    for case in build_cases():
        constraints = {
            name: MetricConstraint("exact", int(value))
            for name, value in case["targets"].items()
        }
        records.append(
            _check_resolution(
                case["case_id"],
                "r098_baseline",
                FunctionalSpec("smb_static", constraints),
            )
        )

    for case in frozen["cases"]:
        if case.get("role") != "valid_target":
            continue
        records.append(
            _check_resolution(
                case["case_id"],
                "r109_language_structured",
                parser_payload_to_spec(case["expected"]),
            )
        )

    result = {
        "run_id": RUN_ID,
        "evaluation_contract": {
            "static_counts_exact": True,
            "supported_pipes_and_enemies": True,
            "simulation_called": True,
            "engine_executed": True,
            "astar_complete": True,
            "all_coin_tiles_collected": True,
        },
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "case_count": len(records),
        "passed": sum(row["passed"] for row in records),
        "gate_pass": len(records) == 15 and all(row["passed"] for row in records),
        "records": records,
    }
    _write_json(output, result)
    if not result["gate_pass"]:
        failed = [row["case_id"] for row in records if not row["passed"]]
        raise RuntimeError(f"SMB static playability gate failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

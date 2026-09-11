"""R083-A SMB evaluator contract and deterministic synthetic witness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from toolusepcg.config import get_control_ranges, load_config
from toolusepcg.evaluator import EvalResult, StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level, SMBTiles
from toolusepcg.smb_contract import (
    COLLECT_EVENT,
    FALL_KILL_EVENT,
    JUMP_EVENT,
    SMB_COIN_EVENT_PARAMS,
    SMB_EVENT_ADAPTER_VERSION,
    SMB_EVENT_ENUM,
    STOMP_KILL_EVENT,
    iter_smb_targets,
    smb_target_error,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = REPO_ROOT / "vendor"
CONTRACT_ID = "r083-smb-mastery-v0"
DEFAULT_REPEATS = 20


def _flat_backbone() -> np.ndarray:
    data = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
    data[-1, :] = int(SMBTiles.SOLID)
    return data


def synthetic_levels() -> dict[str, Level]:
    """Return deterministic positive and negative contract witnesses."""
    levels: dict[str, np.ndarray] = {}

    levels["flat"] = _flat_backbone()

    coin = _flat_backbone()
    coin[14, 10] = int(SMBTiles.COIN)
    levels["coin_pickup"] = coin

    ground_jump = _flat_backbone()
    ground_jump[15, 10:12] = int(SMBTiles.EMPTY)
    levels["ground_jump"] = ground_jump

    stomp = _flat_backbone()
    stomp[14, 15] = int(SMBTiles.GOOMBA)
    stomp[15, 12:15] = int(SMBTiles.EMPTY)
    levels["stomp_kill"] = stomp

    fall = _flat_backbone()
    fall[14, 8] = int(SMBTiles.GOOMBA)
    fall[15, 7] = int(SMBTiles.EMPTY)
    levels["fall_kill_excluded"] = fall

    tube = _flat_backbone()
    tube[14, 10] = int(SMBTiles.TUBE)
    levels["invalid_tube"] = tube

    empty_gate = np.full((16, 32), int(SMBTiles.SOLID), dtype=np.int32)
    levels["empty_gate_skip"] = empty_gate

    floating = _flat_backbone()
    floating[13, 10] = int(SMBTiles.GOOMBA)
    levels["floating_enemy_skip"] = floating

    return {
        name: Level(data, problem_type="smb") for name, data in levels.items()
    }


def build_evaluator() -> tuple[StatsEvaluator, Any]:
    """Build the frozen 16x32 A*-only evaluator without an LLM client."""
    config = load_config(
        "configs/smb.yaml",
        {
            "env": {"height": 16, "width": 32, "smb_solver": "astar"},
            "targets": {
                "complete": {"mode": "target", "value": 1.0},
                "enemies": {"mode": "target", "value": 0},
                "coins": {"mode": "target", "value": 0},
                "jumps": {"mode": "target", "value": 1},
            },
        },
    )
    factory = LevelFactory(config)
    return StatsEvaluator(config, factory.env), factory.env._problem


def _events(result: EvalResult, event_type: int) -> list[dict[str, Any]]:
    return [
        event
        for event in (result.info or {}).get("game_events", [])
        if event.get("event_type", event.get("type")) == event_type
    ]


def _positive_assertions(name: str, result: EvalResult) -> dict[str, bool]:
    metrics = result.metrics
    checks: dict[str, bool] = {
        "hard_valid": result.valid,
        "complete": metrics.get("complete") == 1.0,
        "engine_executed": bool((result.info or {}).get("engine_executed")),
        "nonempty_actions": bool((result.info or {}).get("actions")),
        "nonempty_trajectory": bool((result.info or {}).get("locations")),
        "bounded_event_coordinates": all(
            0 <= event["mario_event_position"]["x"] < 32
            and 0 <= event["mario_event_position"]["y"] < 16
            for event in (result.info or {}).get("game_events", [])
        ),
    }
    if name == "coin_pickup":
        pickups = [
            event
            for event in _events(result, COLLECT_EVENT)
            if event.get("event_param", event.get("param")) in SMB_COIN_EVENT_PARAMS
        ]
        checks.update(
            {
                "one_coin_collected": metrics.get("coins") == 1,
                "coin_event_present": len(pickups) == 1,
                "event_position_is_not_spawn_identity": bool(pickups)
                and pickups[0]["mario_event_position"]
                != (result.info or {})["spawn_positions"]["coins"][0],
            }
        )
    elif name == "ground_jump":
        checks.update(
            {
                "two_ground_jumps": metrics.get("jumps") == 2,
                "jump_events_match_metric": len(_events(result, JUMP_EVENT))
                == metrics.get("jumps"),
            }
        )
    elif name == "stomp_kill":
        stomps = _events(result, STOMP_KILL_EVENT)
        checks.update(
            {
                "one_counted_kill": metrics.get("enemies") == 1,
                "stomp_event_present": len(stomps) == 1,
                "stomp_bounce_not_ground_jump": bool(stomps)
                and not stomps[0]["ground_jump"],
                "upper_y_is_clamped": bool(stomps)
                and stomps[0]["mario_event_position"]["y"] == 15,
            }
        )
    elif name == "fall_kill_excluded":
        falls = _events(result, FALL_KILL_EVENT)
        checks.update(
            {
                "fall_event_present": len(falls) == 1,
                "fall_not_counted": metrics.get("enemies") == 0,
                "fall_semantic_flag": bool(falls)
                and falls[0]["fall_kill_excluded"],
            }
        )
    return checks


def _negative_assertions(name: str, result: EvalResult) -> dict[str, bool]:
    reason = (result.info or {}).get("engine_skip_reason") or ""
    checks = {
        "hard_invalid": not result.valid,
        "engine_not_executed": not (result.info or {}).get("engine_executed", True),
        "empty_actions": not (result.info or {}).get("actions"),
        "skip_reason_exposed": bool(reason),
    }
    expected = {
        "invalid_tube": "tube",
        "empty_gate_skip": "empty_ratio",
        "floating_enemy_skip": "floating_enemy_ratio",
    }[name]
    checks["expected_skip_reason"] = expected in reason
    if name == "invalid_tube":
        checks["tube_issue_hard_invalid"] = result.metrics.get("tube_issues", 0) > 0
    return checks


def _canonical_signature(result: EvalResult) -> str:
    payload = {
        "valid": result.valid,
        "metrics": {
            name: result.metrics.get(name)
            for name in (
                "complete",
                "enemies",
                "coins",
                "jumps",
                "tube_issues",
                "empty_ratio",
                "floating_enemy_ratio",
            )
        },
        "normalized_event_trace": (result.info or {}).get(
            "normalized_event_trace", []
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_output(root: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine_identity(problem: Any) -> dict[str, Any]:
    source_paths = {
        "problem": ENGINE_ROOT / "pcg_benchmark/probs/smbtile/problem.py",
        "event_enum": ENGINE_ROOT / "pcg_benchmark/probs/smbtile/engine/helper.py",
        "astar": ENGINE_ROOT / "pcg_benchmark/probs/smbtile/engine/agents/astar.py",
        "semantic_adapter": REPO_ROOT / "toolusepcg/smb_contract.py",
        "harness_evaluator": REPO_ROOT / "toolusepcg/evaluator.py",
    }
    return {
        "contract_id": CONTRACT_ID,
        "main_commit": _git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "engine_commit": _git_output(ENGINE_ROOT, "rev-parse", "HEAD"),
        "engine_dirty": bool(_git_output(ENGINE_ROOT, "status", "--porcelain")),
        "agent": getattr(problem, "_agent", None),
        "timer": getattr(problem, "_timer", None),
        "solver_expansions": getattr(problem, "_solver", None),
        "padding_tiles": 3,
        "event_enum": dict(SMB_EVENT_ENUM),
        "event_adapter_version": SMB_EVENT_ADAPTER_VERSION,
        "source_sha256": {
            name: _source_sha256(path) for name, path in source_paths.items()
        },
    }


def run_audit(output_dir: Path, *, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("determinism audit requires at least 2 repeats")
    evaluator, problem = build_evaluator()
    levels = synthetic_levels()
    positive = ("coin_pickup", "ground_jump", "stomp_kill", "fall_kill_excluded")
    negative = ("invalid_tube", "empty_gate_skip", "floating_enemy_skip")

    results: dict[str, EvalResult] = {
        name: evaluator.evaluate(level) for name, level in levels.items()
    }
    witnesses: dict[str, Any] = {}
    all_checks: list[bool] = []
    for name, level in levels.items():
        result = results[name]
        if name in positive:
            assertions = _positive_assertions(name, result)
        elif name in negative:
            assertions = _negative_assertions(name, result)
        else:
            assertions = _positive_assertions(name, result)
        all_checks.extend(assertions.values())
        witnesses[name] = {
            "level": level.to_string(),
            "level_sha256": _sha256_text(level.to_string()),
            "valid": result.valid,
            "errors": result.errors,
            "metrics": result.metrics,
            "engine_executed": (result.info or {}).get("engine_executed"),
            "engine_skip_reason": (result.info or {}).get("engine_skip_reason"),
            "engine_gate_status": (result.info or {}).get("engine_gate_status"),
            "spawn_positions": (result.info or {}).get("spawn_positions"),
            "normalized_event_trace": (result.info or {}).get(
                "normalized_event_trace", []
            ),
            "assertions": assertions,
            "pass": all(assertions.values()),
        }

    determinism = {}
    for name in positive:
        signatures = [
            _canonical_signature(evaluator.evaluate(levels[name]))
            for _ in range(repeats)
        ]
        unique = sorted(set(signatures))
        stable = len(unique) == 1
        all_checks.append(stable)
        determinism[name] = {
            "repeats": repeats,
            "byte_identical": stable,
            "unique_signatures": len(unique),
            "signature_sha256": [_sha256_text(value) for value in unique],
        }

    native_ranges = get_control_ranges("smb", 16, 32, smb_envelope="native")
    expanded_ranges = get_control_ranges("smb", 16, 32, smb_envelope="expanded")
    range_contract = {
        "native": {
            "count": len(list(iter_smb_targets("native"))),
            "ranges": {
                name: [value.min_value, value.max_value]
                for name, value in native_ranges.items()
            },
        },
        "expanded": {
            "count": len(list(iter_smb_targets("expanded"))),
            "ranges": {
                name: [value.min_value, value.max_value]
                for name, value in expanded_ranges.items()
            },
        },
    }
    range_ok = range_contract == {
        "native": {
            "count": 140,
            "ranges": {"enemies": [0, 4], "coins": [0, 3], "jumps": [0, 6]},
        },
        "expanded": {
            "count": 210,
            "ranges": {"enemies": [0, 4], "coins": [0, 5], "jumps": [0, 6]},
        },
    }
    all_checks.append(range_ok)

    exact_probe = smb_target_error(
        {"enemies": 1, "coins": 2, "jumps": 3},
        {"enemies": 1, "coins": 2, "jumps": 3},
    )
    exact_contract = exact_probe.as_dict()
    all_checks.append(bool(exact_contract["exact"]))

    payload = {
        "run_id": "R083-A",
        "gate_pass": all(all_checks),
        "identity": _engine_identity(problem),
        "range_contract": range_contract,
        "range_contract_pass": range_ok,
        "exact_contract": exact_contract,
        "witnesses": witnesses,
        "determinism": determinism,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, level in levels.items():
        (output_dir / f"{name}.txt").write_text(level.to_string() + "\n")
    (output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/r083-smb-contract"),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    payload = run_audit(args.output_dir, repeats=args.repeats)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

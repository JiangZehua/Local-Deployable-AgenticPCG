"""R101-A deterministic, cause-labeled SMB jump-episode atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.r083_smb_atlas import (
    CERTIFIED,
    UNKNOWN,
    ArchiveEntry,
    Candidate,
    _level_sha256,
    _signature,
    _trajectory_coin_sites,
)
from experiments.r083_smb_contract import CONTRACT_ID, build_evaluator
from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level, SMBTiles
from toolusepcg.smb_contract import (
    ALL_KILL_EVENTS,
    JUMP_EVENT,
    SMB_EVENT_ADAPTER_VERSION,
    STOMP_KILL_EVENT,
)


RUN_ID = "R101-A"
SCHEMA_VERSION = "r101-smb-jump-episode-atlas-v1"
TARGETS = (
    (0, 0, 1),
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 5, 5),
)
OBSTACLE_PROFILES: dict[int, tuple[int, ...]] = {
    1: (),
    2: (11,),
    3: (11, 27),
    4: (7, 19, 27),
    5: (5, 11, 20, 26),
}


@dataclass(frozen=True)
class SearchConfig:
    recertify_repeats: int = 20
    max_coin_candidates_per_target: int = 64

    def validate(self) -> None:
        if self.recertify_repeats < 2:
            raise ValueError("recertify_repeats must be at least 2")
        if self.max_coin_candidates_per_target <= 0:
            raise ValueError("max_coin_candidates_per_target must be positive")


def jump_episode_level(jumps: int) -> Level:
    data = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
    data[15, :] = int(SMBTiles.SOLID)
    for x in OBSTACLE_PROFILES[jumps]:
        data[14, x] = int(SMBTiles.SOLID)
    return Level(data, problem_type="smb")


def _event_type(event: dict[str, Any]) -> int:
    return int(event.get("event_type", event.get("type", -1)))


def classify_jump_causes(
    result: EvalResult, obstacle_x: tuple[int, ...]
) -> dict[str, Any]:
    jumps = [
        event for event in (result.info or {}).get("game_events", [])
        if _event_type(event) == JUMP_EVENT
    ]
    remaining = set(obstacle_x)
    episodes: list[dict[str, Any]] = []
    counts = {"terminal": 0, "obstacle": 0, "gap": 0, "unknown": 0}
    for event in jumps:
        position = event.get("mario_event_position") or {}
        x, y = int(position.get("x", -1)), int(position.get("y", -1))
        if x >= 31:
            cause = "terminal"
            feature_x = 31
        else:
            ahead = sorted(value for value in remaining if 0 < value - x <= 8)
            if ahead:
                cause = "obstacle"
                feature_x = ahead[0]
                remaining.remove(feature_x)
            else:
                cause = "unknown"
                feature_x = None
        counts[cause] += 1
        episodes.append(
            {
                "mario_position": {"x": x, "y": y},
                "cause": cause,
                "feature_x": feature_x,
                "ground_jump": bool(event.get("ground_jump")),
                "stomp_bounce": bool(event.get("stomp_bounce")),
            }
        )
    return {"counts": counts, "episodes": episodes}


def is_exact_jump_witness(
    result: EvalResult,
    target: tuple[int, int, int],
    obstacle_x: tuple[int, ...],
) -> bool:
    kills, coins, jumps = target
    metrics = result.metrics
    causes = classify_jump_causes(result, obstacle_x)
    events = list((result.info or {}).get("game_events", []))
    kill_events = [event for event in events if _event_type(event) in ALL_KILL_EVENTS]
    stomp_events = [event for event in events if _event_type(event) == STOMP_KILL_EVENT]
    return bool(
        result.valid
        and metrics.get("complete") == 1.0
        and metrics.get("enemies") == kills == 0
        and metrics.get("coins") == coins
        and metrics.get("jumps") == jumps
        and not kill_events
        and not stomp_events
        and causes["counts"]
        == {
            "terminal": 1,
            "obstacle": jumps - 1,
            "gap": 0,
            "unknown": 0,
        }
        and all(
            episode["ground_jump"] and not episode["stomp_bounce"]
            for episode in causes["episodes"]
        )
    )


def coin_candidates(
    base: Level, base_result: EvalResult, coins: int, jumps: int
) -> Iterable[Candidate]:
    candidate = Candidate(
        base,
        "forced_obstacle_jump_episode",
        {"jumps": jumps, "obstacle_x": list(OBSTACLE_PROFILES[jumps])},
    )
    if coins == 0:
        yield candidate
        return
    entry = ArchiveEntry(
        behavior=(0, 0, jumps),
        level=base,
        family=candidate.family,
        parameters=candidate.parameters,
        primitive_cost=0,
        level_sha256=_level_sha256(base),
        result=base_result,
    )
    sites = _trajectory_coin_sites(entry)
    for selected in combinations(sites, coins):
        data = base.data.copy()
        if any(int(data[y, x]) != int(SMBTiles.EMPTY) for x, y in selected):
            continue
        for x, y in selected:
            data[y, x] = int(SMBTiles.COIN)
        yield Candidate(
            Level(data, problem_type="smb"),
            "trajectory_coin_overlay_on_jump_episode",
            {
                "jumps": jumps,
                "obstacle_x": list(OBSTACLE_PROFILES[jumps]),
                "coin_sites": [[x, y] for x, y in selected],
                "requested_coins": coins,
            },
        )


def _target(target: tuple[int, int, int]) -> dict[str, int]:
    return {"enemies": target[0], "coins": target[1], "jumps": target[2]}


def run_atlas(output_dir: Path, *, config: SearchConfig) -> dict[str, Any]:
    config.validate()
    evaluator, problem = build_evaluator()
    evaluations = structural_evaluations = coin_evaluations = recertification = 0
    base_cache: dict[int, tuple[Level, EvalResult]] = {}
    rows: list[dict[str, Any]] = []
    witness_levels: dict[str, str] = {}
    for target in TARGETS:
        _, coins, jumps = target
        if jumps not in base_cache:
            base = jump_episode_level(jumps)
            base_result = evaluator.evaluate(base)
            evaluations += 1
            structural_evaluations += 1
            base_cache[jumps] = (base, base_result)
        base, base_result = base_cache[jumps]
        witness: tuple[Candidate, EvalResult] | None = None
        searched = 0
        for candidate in coin_candidates(base, base_result, coins, jumps):
            if searched >= config.max_coin_candidates_per_target:
                break
            searched += 1
            if coins == 0:
                result = base_result
            else:
                result = evaluator.evaluate(candidate.level)
                evaluations += 1
                coin_evaluations += 1
            if is_exact_jump_witness(
                result, target, OBSTACLE_PROFILES[jumps]
            ):
                witness = candidate, result
                break
        if witness is None:
            rows.append(
                {
                    "target": _target(target),
                    "status": UNKNOWN,
                    "search_evidence": {
                        "coin_candidates_evaluated": searched,
                        "reason": "bounded deterministic search found no certified exact jump episode",
                    },
                }
            )
            continue
        candidate, result = witness
        expected = _signature(result)
        stable = True
        for _ in range(config.recertify_repeats - 1):
            repeat = evaluator.evaluate(candidate.level)
            evaluations += 1
            recertification += 1
            if (
                _signature(repeat) != expected
                or not is_exact_jump_witness(
                    repeat, target, OBSTACLE_PROFILES[jumps]
                )
            ):
                stable = False
                break
        digest = _level_sha256(candidate.level)
        if stable:
            witness_levels[digest] = candidate.level.to_string()
        trace = (result.info or {}).get("normalized_event_trace", [])
        rows.append(
            {
                "target": _target(target),
                "status": CERTIFIED if stable else UNKNOWN,
                "level_sha256": digest,
                "event_trace_sha256": hashlib.sha256(
                    json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "family": candidate.family,
                "parameters": candidate.parameters,
                "jump_causes": classify_jump_causes(
                    result, OBSTACLE_PROFILES[jumps]
                ),
                "repeat_stable": stable,
                "repeats_required": config.recertify_repeats,
                "coin_candidates_evaluated": searched,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    witness_dir = output_dir / "witnesses"
    witness_dir.mkdir(exist_ok=True)
    for digest, text in witness_levels.items():
        (witness_dir / f"{digest}.txt").write_text(text + "\n")
    certified = sum(row["status"] == CERTIFIED for row in rows)
    payload = {
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_id": CONTRACT_ID,
        "event_adapter_version": SMB_EVENT_ADAPTER_VERSION,
        "oracle_only": True,
        "unknown_is_not_infeasible": True,
        "mastery_claim": False,
        "pre_registered_search": {
            "targets": [_target(target) for target in TARGETS],
            "obstacle_profiles": {
                str(jumps): list(columns)
                for jumps, columns in OBSTACLE_PROFILES.items()
            },
            "causal_contract": "one terminal jump plus jumps-1 forced obstacle jumps; no gap/stomp/unknown",
            "max_coin_candidates_per_target": config.max_coin_candidates_per_target,
            "recertify_repeats": config.recertify_repeats,
        },
        "engine": {
            "agent": getattr(problem, "_agent", None),
            "timer": getattr(problem, "_timer", None),
            "solver_expansions": getattr(problem, "_solver", None),
        },
        "search": {
            "evaluator_calls_total": evaluations,
            "structural_evaluations": structural_evaluations,
            "coin_evaluations": coin_evaluations,
            "recertification_evaluations": recertification,
        },
        "coverage": {"certified": certified, "total": len(TARGETS)},
        "targets": rows,
    }
    (output_dir / "atlas.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/r101-smb-jump-episode-atlas"),
    )
    parser.add_argument("--recertify-repeats", type=int, default=20)
    parser.add_argument("--max-coin-candidates-per-target", type=int, default=64)
    args = parser.parse_args()
    payload = run_atlas(
        args.output_dir,
        config=SearchConfig(
            recertify_repeats=args.recertify_repeats,
            max_coin_candidates_per_target=args.max_coin_candidates_per_target,
        ),
    )
    print(json.dumps({"coverage": payload["coverage"], "search": payload["search"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

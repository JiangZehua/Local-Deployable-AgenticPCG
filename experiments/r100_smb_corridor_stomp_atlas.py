"""R100 deterministic corridor/stomp atlas for the R099 kill boundary.

This is a CPU oracle experiment, not a model result.  It searches a bounded,
pre-registered family of low-ceiling corridors, requires every counted kill to
be an observed stomp (and no fall kill), then adds coins only at cells on the
observed Mario trajectory.  A search miss is ``unknown``, never infeasible.
"""

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
    COUNTED_KILL_EVENTS,
    FALL_KILL_EVENT,
    SMB_EVENT_ADAPTER_VERSION,
    STOMP_KILL_EVENT,
)


RUN_ID = "R100-A"
SCHEMA_VERSION = "r100-smb-corridor-stomp-atlas-v1"
TARGETS = (
    (1, 0, 0),
    (1, 5, 0),
    (2, 4, 0),
    (2, 5, 0),
    (3, 2, 0),
    (3, 3, 0),
)

# Ordered independently of the probe outcomes.  The first profile for every
# count is the smallest evenly spaced corridor instance; alternatives make a
# bounded miss meaningful without claiming infeasibility.
ENEMY_PROFILES: dict[int, tuple[tuple[int, ...], ...]] = {
    1: ((10,), (14,), (18,), (22,)),
    2: ((10, 20), (8, 18), (12, 22), (8, 22)),
    3: ((6, 14, 22), (8, 16, 24), (5, 10, 15), (10, 20, 28)),
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


def _corridor_level(enemy_x: tuple[int, ...]) -> Level:
    data = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
    data[12, :] = int(SMBTiles.SOLID)
    data[14:16, :] = int(SMBTiles.SOLID)
    for x in enemy_x:
        data[13, x] = int(SMBTiles.GOOMBA)
    return Level(data, "smb")


def structural_candidates(kills: int) -> list[Candidate]:
    return [
        Candidate(
            _corridor_level(profile),
            "low_ceiling_stomp_corridor",
            {
                "ceiling_y": 12,
                "entity_y": 13,
                "floor_top_y": 14,
                "enemy_x": list(profile),
                "profile_index": index,
            },
        )
        for index, profile in enumerate(ENEMY_PROFILES[kills])
    ]


def _events(result: EvalResult) -> list[dict[str, Any]]:
    return list((result.info or {}).get("game_events", []))


def event_counts(result: EvalResult) -> dict[str, int]:
    types = [event.get("event_type", event.get("type")) for event in _events(result)]
    return {
        "stomp_kills": sum(value == STOMP_KILL_EVENT for value in types),
        "fall_kills": sum(value == FALL_KILL_EVENT for value in types),
        "counted_kills": sum(value in COUNTED_KILL_EVENTS for value in types),
    }


def is_exact_stomp_witness(
    result: EvalResult, target: tuple[int, int, int]
) -> bool:
    kills, coins, jumps = target
    metrics = result.metrics
    counts = event_counts(result)
    return bool(
        result.valid
        and metrics.get("complete") == 1.0
        and metrics.get("enemies") == kills
        and metrics.get("coins") == coins
        and metrics.get("jumps") == jumps
        and counts["stomp_kills"] == kills
        and counts["counted_kills"] == kills
        and counts["fall_kills"] == 0
    )


def coin_overlay_candidates(
    base: Candidate,
    base_result: EvalResult,
    coin_count: int,
) -> Iterable[Candidate]:
    if coin_count == 0:
        yield base
        return
    entry = ArchiveEntry(
        behavior=(
            int(base_result.metrics["enemies"]),
            int(base_result.metrics["coins"]),
            int(base_result.metrics["jumps"]),
        ),
        level=base.level,
        family=base.family,
        parameters=base.parameters,
        primitive_cost=0,
        level_sha256=_level_sha256(base.level),
        result=base_result,
    )
    sites = _trajectory_coin_sites(entry)
    for selected in combinations(sites, coin_count):
        data = base.level.data.copy()
        if any(int(data[y, x]) != int(SMBTiles.EMPTY) for x, y in selected):
            continue
        for x, y in selected:
            data[y, x] = int(SMBTiles.COIN)
        yield Candidate(
            Level(data, "smb"),
            "trajectory_coin_overlay_on_stomp_corridor",
            {
                **base.parameters,
                "base_level_sha256": _level_sha256(base.level),
                "coin_sites": [[x, y] for x, y in selected],
                "requested_coins": coin_count,
            },
        )


def _trace_sha256(result: EvalResult) -> str:
    trace = (result.info or {}).get("normalized_event_trace", [])
    text = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _target_payload(target: tuple[int, int, int]) -> dict[str, int]:
    return {"enemies": target[0], "coins": target[1], "jumps": target[2]}


def run_atlas(output_dir: Path, *, config: SearchConfig) -> dict[str, Any]:
    config.validate()
    evaluator, problem = build_evaluator()
    structural_cache: dict[int, tuple[Candidate, EvalResult] | None] = {}
    evaluations = 0
    structural_evaluations = 0
    coin_evaluations = 0
    recertification_evaluations = 0

    rows: list[dict[str, Any]] = []
    witnesses: dict[str, str] = {}
    for target in TARGETS:
        kills, coins, _ = target
        if kills not in structural_cache:
            structural_cache[kills] = None
            for candidate in structural_candidates(kills):
                result = evaluator.evaluate(candidate.level)
                evaluations += 1
                structural_evaluations += 1
                if is_exact_stomp_witness(result, (kills, 0, 0)):
                    structural_cache[kills] = (candidate, result)
                    break

        base_pair = structural_cache[kills]
        witness: tuple[Candidate, EvalResult] | None = None
        searched = 0
        if base_pair is not None:
            base, base_result = base_pair
            for candidate in coin_overlay_candidates(base, base_result, coins):
                if searched >= config.max_coin_candidates_per_target:
                    break
                searched += 1
                if coins == 0 and candidate is base:
                    result = base_result
                else:
                    result = evaluator.evaluate(candidate.level)
                    evaluations += 1
                    coin_evaluations += 1
                if is_exact_stomp_witness(result, target):
                    witness = (candidate, result)
                    break

        if witness is None:
            rows.append(
                {
                    "target": _target_payload(target),
                    "status": UNKNOWN,
                    "search_evidence": {
                        "coin_candidates_evaluated": searched,
                        "reason": "bounded deterministic oracle found no certified exact stomp witness",
                    },
                }
            )
            continue

        candidate, result = witness
        expected_signature = _signature(result)
        stable = True
        for _ in range(config.recertify_repeats - 1):
            repeat = evaluator.evaluate(candidate.level)
            evaluations += 1
            recertification_evaluations += 1
            if (
                _signature(repeat) != expected_signature
                or not is_exact_stomp_witness(repeat, target)
            ):
                stable = False
                break

        digest = _level_sha256(candidate.level)
        status = CERTIFIED if stable else UNKNOWN
        if stable:
            witnesses[digest] = candidate.level.to_string()
        rows.append(
            {
                "target": _target_payload(target),
                "status": status,
                "level_sha256": digest,
                "event_trace_sha256": _trace_sha256(result),
                "family": candidate.family,
                "parameters": candidate.parameters,
                "event_counts": event_counts(result),
                "repeat_stable": stable,
                "repeats_required": config.recertify_repeats,
                "coin_candidates_evaluated": searched,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    witness_dir = output_dir / "witnesses"
    witness_dir.mkdir(exist_ok=True)
    for digest, text in witnesses.items():
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
            "targets": [_target_payload(target) for target in TARGETS],
            "enemy_profiles": {
                str(kills): [list(profile) for profile in profiles]
                for kills, profiles in ENEMY_PROFILES.items()
            },
            "coin_sites": "combinations of up to 8 evenly spread observed trajectory cells",
            "max_coin_candidates_per_target": config.max_coin_candidates_per_target,
            "recertify_repeats": config.recertify_repeats,
            "required_kill_semantics": "all counted kills are STOMP_KILL; FALL_KILL=0",
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
            "recertification_evaluations": recertification_evaluations,
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
        default=Path("runs/r100-smb-corridor-stomp-atlas"),
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
    print(
        json.dumps(
            {"coverage": payload["coverage"], "search": payload["search"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

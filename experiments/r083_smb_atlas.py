"""R083-B deterministic SMB behavior-state feasibility atlas.

This is an oracle/search arm, never a model-mastery arm.  It enumerates a
finite, deterministic library of structural candidates, archives every novel
hard-valid ``(kills, coins, jumps)`` outcome, then lays coins directly on the
observed Mario trajectory to probe joint outcomes.  Failure to find a witness
is always reported as ``unknown``--never as infeasible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.r083_smb_contract import CONTRACT_ID, build_evaluator
from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level, SMBTiles
from toolusepcg.smb_contract import (
    SMB_EVENT_ADAPTER_VERSION,
    get_smb_target_envelope,
    iter_smb_targets,
)


ATLAS_SCHEMA_VERSION = "r083-smb-atlas-v1"
CERTIFIED = "certified_feasible"
UNKNOWN = "unknown"
ALLOWED_STATUSES = frozenset({CERTIFIED, UNKNOWN})


@dataclass(frozen=True)
class Candidate:
    level: Level
    family: str
    parameters: dict[str, Any]


@dataclass
class ArchiveEntry:
    behavior: tuple[int, int, int]
    level: Level
    family: str
    parameters: dict[str, Any]
    primitive_cost: int
    level_sha256: str
    result: EvalResult


def _flat_data() -> np.ndarray:
    data = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
    data[15, :] = int(SMBTiles.SOLID)
    return data


def _level_sha256(level: Level) -> str:
    return hashlib.sha256(level.to_string().encode("utf-8")).hexdigest()


def _primitive_cost(level: Level) -> int:
    return int(np.count_nonzero(level.data != _flat_data()))


def _subsets(values: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    yield ()
    for size in range(1, len(values) + 1):
        yield from combinations(values, size)


def structural_candidates() -> list[Candidate]:
    """Generate the bounded structural search corpus in deterministic order."""
    candidates: list[Candidate] = [
        Candidate(Level(_flat_data(), "smb"), "flat", {})
    ]

    anchors = (6, 10, 14, 18, 22, 26)
    for width in (1, 2, 3):
        for starts in _subsets(anchors):
            if not starts:
                continue
            data = _flat_data()
            for start in starts:
                data[15, start : min(32, start + width)] = int(SMBTiles.EMPTY)
            candidates.append(
                Candidate(
                    Level(data, "smb"),
                    "gap_subset",
                    {"starts": list(starts), "width": width},
                )
            )

    obstacle_anchors = (7, 11, 15, 19, 23, 27)
    for height in (1, 2):
        for columns in _subsets(obstacle_anchors):
            if not columns:
                continue
            data = _flat_data()
            for column in columns:
                data[15 - height : 15, column] = int(SMBTiles.SOLID)
            candidates.append(
                Candidate(
                    Level(data, "smb"),
                    "obstacle_subset",
                    {"columns": list(columns), "height": height},
                )
            )

    for enemy_x in range(10, 29):
        for gap_width in (1, 2, 3, 4):
            data = _flat_data()
            data[14, enemy_x] = int(SMBTiles.GOOMBA)
            data[15, enemy_x - gap_width : enemy_x] = int(SMBTiles.EMPTY)
            candidates.append(
                Candidate(
                    Level(data, "smb"),
                    "single_stomp_probe",
                    {"enemy_x": enemy_x, "left_gap_width": gap_width},
                )
            )

    for offset in (0, 1, 2):
        positions = tuple(value + offset for value in (9, 15, 21, 27))
        for gap_width in (2, 3):
            for enemies in _subsets(positions):
                if len(enemies) < 2:
                    continue
                data = _flat_data()
                for enemy_x in enemies:
                    data[14, enemy_x] = int(SMBTiles.GOOMBA)
                    data[15, enemy_x - gap_width : enemy_x] = int(SMBTiles.EMPTY)
                candidates.append(
                    Candidate(
                        Level(data, "smb"),
                        "multi_stomp_probe",
                        {
                            "enemy_x": list(enemies),
                            "left_gap_width": gap_width,
                            "offset": offset,
                        },
                    )
                )

    deduplicated: dict[str, Candidate] = {}
    for candidate in candidates:
        deduplicated.setdefault(_level_sha256(candidate.level), candidate)
    return list(deduplicated.values())


def _behavior(result: EvalResult) -> tuple[int, int, int] | None:
    if not result.valid:
        return None
    values = []
    envelope = get_smb_target_envelope("expanded")
    for name in ("enemies", "coins", "jumps"):
        value = result.metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        if not numeric.is_integer():
            return None
        integer = int(numeric)
        lower, upper = envelope[name]
        if not lower <= integer <= upper:
            return None
        values.append(integer)
    return values[0], values[1], values[2]


def _archive_candidate(
    archive: dict[tuple[int, int, int], ArchiveEntry],
    candidate: Candidate,
    result: EvalResult,
) -> bool:
    behavior = _behavior(result)
    if behavior is None:
        return False
    level_hash = _level_sha256(candidate.level)
    cost = _primitive_cost(candidate.level)
    current = archive.get(behavior)
    if current is not None and (current.primitive_cost, current.level_sha256) <= (
        cost,
        level_hash,
    ):
        return False
    archive[behavior] = ArchiveEntry(
        behavior=behavior,
        level=candidate.level.copy(),
        family=candidate.family,
        parameters=dict(candidate.parameters),
        primitive_cost=cost,
        level_sha256=level_hash,
        result=result,
    )
    return True


def _trajectory_coin_sites(entry: ArchiveEntry) -> list[tuple[int, int]]:
    """Select empty cells Mario actually occupied, ordered across the level."""
    level = entry.level
    by_x: dict[int, list[int]] = {}
    for x_px, y_px in (entry.result.info or {}).get("locations", []):
        x = max(0, min(level.width - 1, int(float(x_px) / 16) - 3))
        y = max(0, min(level.height - 1, int(float(y_px) / 16)))
        if not 2 <= x <= 29:
            continue
        if int(level.data[y, x]) != int(SMBTiles.EMPTY):
            continue
        by_x.setdefault(x, []).append(y)

    sites = []
    for x in sorted(by_x):
        # Prefer Mario's lowest (largest-y) occupied tile at each column.
        y = max(by_x[x])
        sites.append((x, y))
    if len(sites) <= 8:
        return sites
    # Evenly spread the candidate sites so prefixes do not cluster at spawn.
    indexes = np.linspace(0, len(sites) - 1, num=8, dtype=int)
    return [sites[int(index)] for index in indexes]


def _coin_overlay_candidates(entry: ArchiveEntry) -> Iterable[Candidate]:
    sites = _trajectory_coin_sites(entry)
    if not sites:
        return
    profiles = [sites, list(reversed(sites))]
    if len(sites) >= 5:
        profiles.append(sites[::2] + sites[1::2])
    seen: set[str] = set()
    for profile_index, profile in enumerate(profiles):
        for count in range(1, min(5, len(profile)) + 1):
            data = entry.level.data.copy()
            selected = profile[:count]
            if any(int(data[y, x]) != int(SMBTiles.EMPTY) for x, y in selected):
                continue
            for x, y in selected:
                data[y, x] = int(SMBTiles.COIN)
            candidate = Candidate(
                Level(data, "smb"),
                "trajectory_coin_overlay",
                {
                    "base_behavior": list(entry.behavior),
                    "base_level_sha256": entry.level_sha256,
                    "profile": profile_index,
                    "requested_coin_spawns": count,
                    "sites": [[x, y] for x, y in selected],
                },
            )
            digest = _level_sha256(candidate.level)
            if digest not in seen:
                seen.add(digest)
                yield candidate


def _signature(result: EvalResult) -> str:
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
        "trace": (result.info or {}).get("normalized_event_trace", []),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _entry_payload(entry: ArchiveEntry, *, stable: bool) -> dict[str, Any]:
    trace = (entry.result.info or {}).get("normalized_event_trace", [])
    trace_text = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    return {
        "status": CERTIFIED if stable else UNKNOWN,
        "behavior": {
            "enemies": entry.behavior[0],
            "coins": entry.behavior[1],
            "jumps": entry.behavior[2],
        },
        "level_sha256": entry.level_sha256,
        "event_trace_sha256": hashlib.sha256(trace_text.encode()).hexdigest(),
        "primitive_cost": entry.primitive_cost,
        "family": entry.family,
        "parameters": entry.parameters,
        "repeat_stable": stable,
    }


def build_manifest(
    envelope: str,
    certified: dict[tuple[int, int, int], dict[str, Any]],
    *,
    candidate_evaluations: int,
) -> list[dict[str, Any]]:
    """Build a complete envelope manifest with no infeasible status."""
    rows = []
    for target in iter_smb_targets(envelope):
        key = (target["enemies"], target["coins"], target["jumps"])
        witness = certified.get(key)
        if witness is None or witness["status"] != CERTIFIED:
            rows.append(
                {
                    "target": target,
                    "status": UNKNOWN,
                    "search_evidence": {
                        "candidate_evaluations": candidate_evaluations,
                        "reason": "bounded deterministic oracle did not find a stable exact witness",
                    },
                }
            )
        else:
            rows.append({"target": target, **witness})
    if any(row["status"] not in ALLOWED_STATUSES for row in rows):
        raise AssertionError("atlas emitted a forbidden feasibility status")
    return rows


def run_atlas(
    output_dir: Path,
    *,
    max_structural: int | None = None,
    verify_repeats: int = 2,
) -> dict[str, Any]:
    if verify_repeats < 2:
        raise ValueError("certification requires at least 2 identical evaluations")
    evaluator, problem = build_evaluator()
    candidates = structural_candidates()
    if max_structural is not None:
        candidates = candidates[:max_structural]

    archive: dict[tuple[int, int, int], ArchiveEntry] = {}
    evaluations = 0
    for candidate in candidates:
        result = evaluator.evaluate(candidate.level)
        evaluations += 1
        _archive_candidate(archive, candidate, result)

    structural_entries = list(archive.values())
    for entry in structural_entries:
        for candidate in _coin_overlay_candidates(entry):
            result = evaluator.evaluate(candidate.level)
            evaluations += 1
            _archive_candidate(archive, candidate, result)

    certified: dict[tuple[int, int, int], dict[str, Any]] = {}
    witness_levels: dict[str, str] = {}
    for behavior, entry in sorted(archive.items()):
        expected = _signature(entry.result)
        stable = True
        for _ in range(verify_repeats - 1):
            repeat = evaluator.evaluate(entry.level)
            evaluations += 1
            if _signature(repeat) != expected:
                stable = False
                break
        payload = _entry_payload(entry, stable=stable)
        certified[behavior] = payload
        if stable:
            witness_levels[entry.level_sha256] = entry.level.to_string()

    native = build_manifest(
        "native", certified, candidate_evaluations=evaluations
    )
    expanded = build_manifest(
        "expanded", certified, candidate_evaluations=evaluations
    )
    native_found = sum(row["status"] == CERTIFIED for row in native)
    expanded_found = sum(row["status"] == CERTIFIED for row in expanded)

    single_axis = {
        "kills": sorted({
            behavior[0]
            for behavior, witness in certified.items()
            if witness["status"] == CERTIFIED
        }),
        "coins": sorted({
            behavior[1]
            for behavior, witness in certified.items()
            if witness["status"] == CERTIFIED
        }),
        "jumps": sorted({
            behavior[2]
            for behavior, witness in certified.items()
            if witness["status"] == CERTIFIED
        }),
    }
    payload = {
        "run_id": "R083-B1-initial",
        "schema_version": ATLAS_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "event_adapter_version": SMB_EVENT_ADAPTER_VERSION,
        "oracle_only": True,
        "unknown_is_not_infeasible": True,
        "engine": {
            "agent": getattr(problem, "_agent", None),
            "timer": getattr(problem, "_timer", None),
            "solver_expansions": getattr(problem, "_solver", None),
        },
        "search": {
            "structural_candidates": len(candidates),
            "candidate_evaluations": evaluations,
            "verify_repeats": verify_repeats,
            "behavior_archive_states": len(archive),
        },
        "single_axis_observed_values": single_axis,
        "coverage": {
            "native": {"certified_feasible": native_found, "total": 140},
            "expanded": {"certified_feasible": expanded_found, "total": 210},
        },
        "native_manifest": native,
        "expanded_manifest": expanded,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    witness_dir = output_dir / "witnesses"
    witness_dir.mkdir(exist_ok=True)
    for level_hash, level_text in witness_levels.items():
        (witness_dir / f"{level_hash}.txt").write_text(level_text + "\n")
    (output_dir / "atlas.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/r083-smb-atlas"),
    )
    parser.add_argument("--max-structural", type=int)
    parser.add_argument("--verify-repeats", type=int, default=2)
    args = parser.parse_args()
    payload = run_atlas(
        args.output_dir,
        max_structural=args.max_structural,
        verify_repeats=args.verify_repeats,
    )
    print(json.dumps({"search": payload["search"], "coverage": payload["coverage"], "single_axis_observed_values": payload["single_axis_observed_values"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

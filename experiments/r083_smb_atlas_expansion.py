"""R083-B3 target-conditioned deterministic SMB atlas expansion.

The search combines jump and stomp geometry instead of extending the narrow,
mutually exclusive R083-B1 grammar.  It archives every new hard-valid behavior
state, stops only under a pre-registered wave budget/saturation rule, and keeps
all search misses as ``unknown``.  This remains an oracle arm and never seeds a
model episode with a witness level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product, zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from experiments.r083_smb_atlas import (
    ALLOWED_STATUSES,
    CERTIFIED,
    UNKNOWN,
    ArchiveEntry,
    Candidate,
    _archive_candidate,
    _behavior,
    _coin_overlay_candidates,
    _entry_payload,
    _level_sha256,
    _signature,
    build_manifest,
)
from experiments.r083_smb_contract import CONTRACT_ID, build_evaluator
from toolusepcg.level import Level, SMBTiles
from toolusepcg.smb_contract import SMB_EVENT_ADAPTER_VERSION


SCHEMA_VERSION = "r083-smb-atlas-expansion-v1"
RUN_ID = "R083-B3"
FRONTIER_KJ = frozenset(
    {(4, 0), (4, 1), (0, 0), (0, 6)}
    | {(3, jump) for jump in range(2, 7)}
    | {(2, jump) for jump in range(3, 7)}
    | {(1, jump) for jump in range(4, 7)}
)


@dataclass(frozen=True)
class SearchConfig:
    wave_budget: int = 500
    max_waves: int = 6
    saturation_patience: int = 2
    recertify_repeats: int = 20

    def validate(self) -> None:
        if self.wave_budget <= 0 or self.max_waves <= 0:
            raise ValueError("wave_budget and max_waves must be positive")
        if self.saturation_patience < 2:
            raise ValueError("saturation_patience must be at least 2")
        if self.recertify_repeats < 2:
            raise ValueError("recertify_repeats must be at least 2")


def _flat_data() -> np.ndarray:
    data = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
    data[15, :] = int(SMBTiles.SOLID)
    return data


def _candidate(data: np.ndarray, family: str, **parameters: Any) -> Candidate:
    return Candidate(Level(data, "smb"), family, parameters)


def _deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    result: dict[str, Candidate] = {}
    for candidate in candidates:
        result.setdefault(_level_sha256(candidate.level), candidate)
    return list(result.values())


def _jump_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for spacing in (3, 4, 5):
        anchors = tuple(range(5, 30, spacing))
        for count in range(4, min(7, len(anchors)) + 1):
            for columns in combinations(anchors, count):
                for profile_name, heights in (
                    ("low", (1,) * count),
                    ("high", (2,) * count),
                    ("alternating", tuple(1 + index % 2 for index in range(count))),
                ):
                    data = _flat_data()
                    for column, height in zip(columns, heights, strict=True):
                        data[15 - height : 15, column] = int(SMBTiles.SOLID)
                    candidates.append(
                        _candidate(
                            data,
                            "b3_jump_obstacles",
                            columns=list(columns),
                            heights=list(heights),
                            spacing=spacing,
                            profile=profile_name,
                        )
                    )
        for count in range(4, min(6, len(anchors)) + 1):
            for starts in combinations(anchors, count):
                for width in (1, 2):
                    data = _flat_data()
                    for start in starts:
                        data[15, start : min(31, start + width)] = int(SMBTiles.EMPTY)
                    candidates.append(
                        _candidate(
                            data,
                            "b3_jump_gaps",
                            starts=list(starts),
                            width=width,
                            spacing=spacing,
                        )
                    )
    return _deduplicate(candidates)


def _kill_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    width_profiles = (
        (2, 2, 2, 2, 2),
        (3, 3, 3, 3, 3),
        (4, 4, 4, 4, 4),
        (2, 3, 2, 3, 2),
        (3, 2, 3, 2, 3),
    )
    for count in (4, 5):
        for start in range(8, 13):
            for spacings in product((4, 5, 6, 7), repeat=count - 1):
                positions = [start]
                for spacing in spacings:
                    positions.append(positions[-1] + spacing)
                if positions[-1] > 29:
                    continue
                for widths in width_profiles:
                    data = _flat_data()
                    actual_widths = widths[:count]
                    valid = True
                    for enemy_x, width in zip(positions, actual_widths, strict=True):
                        gap_start = enemy_x - width
                        if gap_start < 3:
                            valid = False
                            break
                        data[15, gap_start:enemy_x] = int(SMBTiles.EMPTY)
                        data[14, enemy_x] = int(SMBTiles.GOOMBA)
                    if valid:
                        candidates.append(
                            _candidate(
                                data,
                                "b3_multi_stomp",
                                enemy_x=positions,
                                left_gap_widths=list(actual_widths),
                            )
                        )
    return _deduplicate(candidates)


def _load_structural_seeds(atlas_path: Path) -> list[tuple[tuple[int, int], Level]]:
    atlas = json.loads(atlas_path.read_text())
    witness_dir = atlas_path.parent / "witnesses"
    result: list[tuple[tuple[int, int], Level]] = []
    for row in atlas["native_manifest"]:
        target = row["target"]
        if row.get("status") != CERTIFIED or int(target["coins"]) != 0:
            continue
        level = Level.from_string(
            (witness_dir / f"{row['level_sha256']}.txt").read_text().strip(), "smb"
        )
        result.append(((int(target["enemies"]), int(target["jumps"])), level))
    result.sort(key=lambda item: item[0])
    return result


def _coupling_candidates(atlas_path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    anchors = (5, 9, 13, 17, 21, 25, 29)
    for (kills, jumps), seed in _load_structural_seeds(atlas_path):
        if kills > 0:
            for count in (1, 2, 3):
                for columns in combinations(anchors, count):
                    data = seed.data.copy()
                    if any(int(data[14, column]) != int(SMBTiles.EMPTY) for column in columns):
                        continue
                    if any(int(data[15, column]) != int(SMBTiles.SOLID) for column in columns):
                        continue
                    for column in columns:
                        data[14, column] = int(SMBTiles.SOLID)
                    candidates.append(
                        _candidate(
                            data,
                            "b3_stomp_plus_obstacles",
                            seed_kills=kills,
                            seed_jumps=jumps,
                            columns=list(columns),
                        )
                    )
        if kills == 0 and jumps >= 3:
            for enemy_x in (10, 14, 18, 22, 26, 29):
                for gap_width in (2, 3):
                    data = seed.data.copy()
                    gap_start = enemy_x - gap_width
                    if int(data[14, enemy_x]) != int(SMBTiles.EMPTY):
                        continue
                    data[15, gap_start:enemy_x] = int(SMBTiles.EMPTY)
                    data[14, enemy_x] = int(SMBTiles.GOOMBA)
                    candidates.append(
                        _candidate(
                            data,
                            "b3_jump_plus_stomp",
                            seed_jumps=jumps,
                            enemy_x=enemy_x,
                            left_gap_width=gap_width,
                        )
                    )
    return _deduplicate(candidates)


def target_conditioned_candidates(atlas_path: Path) -> list[Candidate]:
    """Interleave three search families so every wave probes each frontier."""
    families = [_kill_candidates(), _jump_candidates(), _coupling_candidates(atlas_path)]
    interleaved: list[Candidate] = []
    for group in zip_longest(*families):
        interleaved.extend(candidate for candidate in group if candidate is not None)
    return _deduplicate(interleaved)


def _load_initial_archive(
    atlas_path: Path, evaluator: Any
) -> tuple[dict[tuple[int, int, int], ArchiveEntry], int]:
    atlas = json.loads(atlas_path.read_text())
    witness_dir = atlas_path.parent / "witnesses"
    archive: dict[tuple[int, int, int], ArchiveEntry] = {}
    evaluations = 0
    for row in atlas["expanded_manifest"]:
        if row.get("status") != CERTIFIED:
            continue
        level = Level.from_string(
            (witness_dir / f"{row['level_sha256']}.txt").read_text().strip(), "smb"
        )
        result = evaluator.evaluate(level)
        evaluations += 1
        candidate = Candidate(level, f"initial:{row['family']}", dict(row["parameters"]))
        if _behavior(result) != (
            int(row["target"]["enemies"]),
            int(row["target"]["coins"]),
            int(row["target"]["jumps"]),
        ):
            raise RuntimeError(f"initial atlas witness drift: {row['level_sha256']}")
        _archive_candidate(archive, candidate, result)
    return archive, evaluations


def _frontier_hits(archive: dict[tuple[int, int, int], ArchiveEntry]) -> list[list[int]]:
    return [
        [kills, jumps]
        for kills, jumps in sorted({(key[0], key[2]) for key in archive})
        if (kills, jumps) in FRONTIER_KJ
    ]


def run_expansion(
    atlas_path: Path,
    output_dir: Path,
    *,
    config: SearchConfig,
) -> dict[str, Any]:
    config.validate()
    evaluator, problem = build_evaluator()
    archive, evaluations = _load_initial_archive(atlas_path, evaluator)
    initial_behaviors = set(archive)
    candidates = target_conditioned_candidates(atlas_path)
    seen_levels = {entry.level_sha256 for entry in archive.values()}
    candidates = [candidate for candidate in candidates if _level_sha256(candidate.level) not in seen_levels]

    waves: list[dict[str, Any]] = []
    family_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"evaluations": 0, "new_behavior_states": 0}
    )
    no_gain_waves = 0
    stop_reason = "max_waves_reached"
    cursor = 0
    new_structural_entries: list[ArchiveEntry] = []
    for wave_index in range(config.max_waves):
        wave_candidates = candidates[cursor : cursor + config.wave_budget]
        cursor += len(wave_candidates)
        if not wave_candidates:
            stop_reason = "candidate_corpus_exhausted"
            break
        before = set(archive)
        discovered_in_wave: set[tuple[int, int, int]] = set()
        for candidate in wave_candidates:
            result = evaluator.evaluate(candidate.level)
            evaluations += 1
            family_stats[candidate.family]["evaluations"] += 1
            added = _archive_candidate(archive, candidate, result)
            behavior = _behavior(result)
            if added and behavior not in before and behavior not in discovered_in_wave:
                discovered_in_wave.add(behavior)
                entry = archive[behavior]
                new_structural_entries.append(entry)
                family_stats[candidate.family]["new_behavior_states"] += 1
        gained = len(set(archive) - before)
        no_gain_waves = no_gain_waves + 1 if gained == 0 else 0
        waves.append(
            {
                "wave": wave_index + 1,
                "evaluations": len(wave_candidates),
                "new_behavior_states": gained,
                "archive_states_after": len(archive),
                "consecutive_no_gain_waves": no_gain_waves,
                "frontier_hits": _frontier_hits(archive),
            }
        )
        if no_gain_waves >= config.saturation_patience:
            stop_reason = "pre_registered_saturation"
            break

    # Probe coin overlays only for genuinely new structural outcomes.
    for entry in list({entry.behavior: entry for entry in new_structural_entries}.values()):
        for candidate in _coin_overlay_candidates(entry):
            result = evaluator.evaluate(candidate.level)
            evaluations += 1
            _archive_candidate(archive, candidate, result)

    certified: dict[tuple[int, int, int], dict[str, Any]] = {}
    witness_levels: dict[str, str] = {}
    recertification_evaluations = 0
    for behavior, entry in sorted(archive.items()):
        expected = _signature(entry.result)
        stable = True
        for _ in range(config.recertify_repeats - 1):
            repeat = evaluator.evaluate(entry.level)
            evaluations += 1
            recertification_evaluations += 1
            if _signature(repeat) != expected:
                stable = False
                break
        payload = _entry_payload(entry, stable=stable)
        certified[behavior] = payload
        if stable:
            witness_levels[entry.level_sha256] = entry.level.to_string()

    native = build_manifest("native", certified, candidate_evaluations=evaluations)
    expanded = build_manifest("expanded", certified, candidate_evaluations=evaluations)
    if any(row["status"] not in ALLOWED_STATUSES for row in native + expanded):
        raise AssertionError("forbidden feasibility status")

    output_dir.mkdir(parents=True, exist_ok=True)
    witness_dir = output_dir / "witnesses"
    witness_dir.mkdir(exist_ok=True)
    for digest, text in witness_levels.items():
        (witness_dir / f"{digest}.txt").write_text(text + "\n")

    payload = {
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract_id": CONTRACT_ID,
        "event_adapter_version": SMB_EVENT_ADAPTER_VERSION,
        "oracle_only": True,
        "unknown_is_not_infeasible": True,
        "mastery_support_frozen": False,
        "artifacts": {
            "run_directory": str(output_dir),
            "witness_directory": str(witness_dir),
        },
        "source_atlas": {
            "path": str(atlas_path),
            "sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
        },
        "engine": {
            "agent": getattr(problem, "_agent", None),
            "timer": getattr(problem, "_timer", None),
            "solver_expansions": getattr(problem, "_solver", None),
        },
        "pre_registered_search": {
            "wave_budget": config.wave_budget,
            "max_waves": config.max_waves,
            "saturation_patience": config.saturation_patience,
            "recertify_repeats": config.recertify_repeats,
            "candidate_corpus": len(candidates),
            "frontier_kill_jump": [list(item) for item in sorted(FRONTIER_KJ)],
        },
        "search": {
            "stop_reason": stop_reason,
            "waves": waves,
            "family_stats": dict(sorted(family_stats.items())),
            "evaluator_calls_total": evaluations,
            "recertification_evaluations": recertification_evaluations,
            "initial_archive_states": len(initial_behaviors),
            "final_archive_states": len(archive),
            "new_behavior_states": len(set(archive) - initial_behaviors),
            "frontier_hits": _frontier_hits(archive),
        },
        "coverage": {
            "native": {
                "certified_feasible": sum(row["status"] == CERTIFIED for row in native),
                "total": 140,
            },
            "expanded": {
                "certified_feasible": sum(row["status"] == CERTIFIED for row in expanded),
                "total": 210,
            },
        },
        "native_manifest": native,
        "expanded_manifest": expanded,
    }
    (output_dir / "atlas.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("benchmarks/smb/search_seeds/atlas.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/r083-smb-atlas-b3"),
    )
    parser.add_argument("--timestamp")
    parser.add_argument("--wave-budget", type=int, default=500)
    parser.add_argument("--max-waves", type=int, default=6)
    parser.add_argument("--saturation-patience", type=int, default=2)
    parser.add_argument("--recertify-repeats", type=int, default=20)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"R083_B3_{timestamp}"
    payload = run_expansion(
        args.atlas,
        run_dir,
        config=SearchConfig(
            wave_budget=args.wave_budget,
            max_waves=args.max_waves,
            saturation_patience=args.saturation_patience,
            recertify_repeats=args.recertify_repeats,
        ),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "atlas.json", args.output_root / "atlas.json")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "search": payload["search"],
                "coverage": payload["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

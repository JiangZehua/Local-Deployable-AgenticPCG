"""Paired high-level free-form language-control evaluation for SMB and Zelda.

The protocol mirrors the four-tier ToolUsePCG study while keeping generation,
tile-native semantic checks, gameplay validity, and blinded judgment separate.
Failed or missing generations remain in the denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "high_level_freeform_eval.v1"
LABEL_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0, "unverifiable": 0.0}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_suite(path: Path) -> dict[str, Any]:
    suite = yaml.safe_load(path.read_text())
    if not isinstance(suite, dict):
        raise ValueError("suite must be a YAML mapping")
    if suite.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"suite schema_version must be {SCHEMA_VERSION!r}")
    environment = suite.get("environment")
    cases = suite.get("cases")
    if not isinstance(environment, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("suite requires environment and at least one case")
    seeds = environment.get("replicate_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("environment.replicate_seeds must be a non-empty unique list")
    case_ids: set[str] = set()
    tiers: set[int] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every case must be a mapping")
        case_id = case.get("case_id")
        tier = case.get("tier")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError(f"invalid or duplicate case_id: {case_id!r}")
        if tier not in {1, 2, 3, 4}:
            raise ValueError(f"{case_id}: tier must be 1, 2, 3, or 4")
        required = case.get("required_atoms")
        programmatic = case.get("programmatic_atoms", [])
        judged = case.get("judgment_atoms", [])
        if not isinstance(required, list) or not required or len(required) != len(set(required)):
            raise ValueError(f"{case_id}: required_atoms must be a non-empty unique list")
        if set(programmatic) | set(judged) != set(required):
            raise ValueError(
                f"{case_id}: programmatic_atoms and judgment_atoms must partition required_atoms"
            )
        if set(programmatic) & set(judged):
            raise ValueError(f"{case_id}: an atom cannot use both verifier classes")
        if not str(case.get("instruction", "")).strip():
            raise ValueError(f"{case_id}: instruction is required")
        negatives = case.get("hard_negatives")
        if not isinstance(negatives, list) or len(negatives) < 4:
            raise ValueError(f"{case_id}: at least four hard negatives are required")
        case_ids.add(case_id)
        tiers.add(int(tier))
    if tiers != {1, 2, 3, 4}:
        raise ValueError("the frozen suite must cover all four semantic tiers")
    return suite


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _cell_id(condition: str, seed: int, case_id: str | None = None) -> str:
    if condition == "null":
        return f"null__seed-{seed}"
    return f"matched__{case_id}__seed-{seed}"


def build_plan(
    suite: Mapping[str, Any],
    run_dir: Path,
    *,
    config_override: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build the shared-control paired generation plan without executing it."""
    environment = suite["environment"]
    config = (
        config_override.resolve()
        if config_override is not None
        else _resolve_project_path(str(environment["config"])).resolve()
    )
    initial_level = _resolve_project_path(str(environment["initial_level"])).resolve()
    for required_path in (config, initial_level):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    common = [
        sys.executable,
        str(ROOT / "main.py"),
        "--config",
        str(config),
        "--init",
        str(initial_level),
        "--max-steps",
        str(int(environment["max_steps"])),
        "--no-render-stats",
        "--no-render-sim",
        "--quiet",
        "--prompt-profile", "freeform",
    ]
    optional = (
        ("--provider", provider),
        ("--model", model),
        ("--base-url", base_url),
        ("--max-tokens", max_tokens),
        ("--temperature", temperature),
    )
    for flag, value in optional:
        if value is not None:
            common.extend([flag, str(value)])

    cells: list[dict[str, Any]] = []
    for seed_value in environment["replicate_seeds"]:
        seed = int(seed_value)
        cell_id = _cell_id("null", seed)
        cell_dir = (run_dir / "cells" / cell_id).resolve()
        command = [*common, "--seed", str(seed), "--run-dir", str(cell_dir)]
        cells.append(
            {
                "cell_id": cell_id,
                "condition": "null",
                "case_id": None,
                "tier": None,
                "seed": seed,
                "instruction": None,
                "run_dir": str(cell_dir),
                "command": command,
            }
        )
    for case in suite["cases"]:
        for seed_value in environment["replicate_seeds"]:
            seed = int(seed_value)
            cell_id = _cell_id("matched", seed, str(case["case_id"]))
            cell_dir = (run_dir / "cells" / cell_id).resolve()
            command = [
                *common,
                "--seed",
                str(seed),
                "--run-dir",
                str(cell_dir),
                "--extra-instruction",
                str(case["instruction"]),
            ]
            cells.append(
                {
                    "cell_id": cell_id,
                    "condition": "matched",
                    "case_id": case["case_id"],
                    "tier": int(case["tier"]),
                    "seed": seed,
                    "instruction": case["instruction"],
                    "run_dir": str(cell_dir),
                    "command": command,
                }
            )
    suite_payload = dict(suite)
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite["suite_id"],
        "suite_sha256": sha256_json(suite_payload),
        "resolved_config": str(config),
        "config_sha256": sha256_file(config),
        "initial_level_sha256": sha256_file(initial_level),
        "shared_null_controls": True,
        "expected_generation_attempts": len(cells),
        "expected_paired_comparisons": len(suite["cases"])
        * len(environment["replicate_seeds"]),
        "cells": cells,
    }


def freeze_plan(suite_path: Path, suite: Mapping[str, Any], plan: Mapping[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "frozen_suite.yaml"
    source = suite_path.read_bytes()
    if snapshot.exists() and snapshot.read_bytes() != source:
        raise ValueError(f"frozen suite differs from existing snapshot: {snapshot}")
    if not snapshot.exists():
        snapshot.write_bytes(source)
    plan_path = run_dir / "run_plan.json"
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        if prior != plan:
            raise ValueError(f"resolved plan differs from existing plan: {plan_path}")
    else:
        write_json(plan_path, plan)


def _cell_complete(cell_dir: Path) -> bool:
    result_path = cell_dir / "process_result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    return (
        result.get("returncode") == 0
        and (cell_dir / "summary.json").is_file()
        and (cell_dir / "final_level.txt").is_file()
    )


def execute_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    completed = skipped = 0
    for index, cell in enumerate(plan["cells"], 1):
        cell_dir = Path(cell["run_dir"])
        if _cell_complete(cell_dir):
            skipped += 1
            print(f"[{index}/{len(plan['cells'])}] skip complete {cell['cell_id']}", flush=True)
            continue
        if cell_dir.exists() and any(cell_dir.iterdir()):
            raise RuntimeError(
                f"refusing to mix a retry with partial artifacts in {cell_dir}; "
                "use a new --run-dir or move the failed cell aside"
            )
        cell_dir.mkdir(parents=True, exist_ok=True)
        identity = {key: value for key, value in cell.items() if key != "command"}
        identity["command_sha256"] = sha256_json(cell["command"])
        write_json(cell_dir / "cell.json", identity)
        print(f"[{index}/{len(plan['cells'])}] run {cell['cell_id']}", flush=True)
        with (cell_dir / "process.log").open("w") as log:
            process_env = os.environ.copy()
            process_env.setdefault("RPE_PCG_OPENAI_SEED", "on")
            process = subprocess.run(
                list(cell["command"]),
                cwd=ROOT,
                env=process_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        write_json(cell_dir / "process_result.json", {"returncode": process.returncode})
        if process.returncode != 0:
            raise RuntimeError(
                f"generation failed for {cell['cell_id']} (exit {process.returncode}); "
                f"see {cell_dir / 'process.log'}"
            )
        completed += 1
    return {"completed_now": completed, "skipped_complete": skipped}


Cell = tuple[int, int]


def positions(level: Sequence[str], tile: str) -> set[Cell]:
    return {
        (row, col)
        for row, line in enumerate(level)
        for col, value in enumerate(line)
        if value == tile
    }


def components(mask: set[Cell]) -> list[set[Cell]]:
    unseen = set(mask)
    groups: list[set[Cell]] = []
    while unseen:
        start = unseen.pop()
        group = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)


def bounding_box(mask: set[Cell]) -> tuple[int, int, int, int] | None:
    if not mask:
        return None
    rows = [row for row, _ in mask]
    cols = [col for _, col in mask]
    return min(rows), max(rows), min(cols), max(cols)


def frame_cells(top: int, bottom: int, left: int, right: int) -> set[Cell]:
    return {
        (row, col)
        for row in range(top, bottom + 1)
        for col in range(left, right + 1)
        if row in {top, bottom} or col in {left, right}
    }


def mask_iou(left: set[Cell], right: set[Cell]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def t1_metrics(level: Sequence[str]) -> dict[str, Any]:
    groups = components(positions(level, "B"))
    candidates: list[dict[str, Any]] = []
    for group in groups:
        box = bounding_box(group)
        if box is None:
            continue
        top, bottom, left, right = box
        boundary = frame_cells(top, bottom, left, right)
        interior = {
            (row, col)
            for row in range(top + 1, bottom)
            for col in range(left + 1, right)
        }
        candidates.append(
            {
                "tiles": len(group),
                "height": bottom - top + 1,
                "width": right - left + 1,
                "frame_iou": mask_iou(group, boundary),
                "closed": boundary <= group,
                "hollow": not bool(interior & group),
            }
        )
    best = max(candidates, key=lambda item: (item["frame_iou"], item["tiles"]), default={})
    atoms = {
        "one_large_brick_structure": bool(
            best and best["tiles"] >= 12 and best["height"] >= 3 and best["width"] >= 4
        ),
        "closed_rectangular_contour": bool(best and best["closed"] and best["frame_iou"] >= 0.8),
        "hollow_interior": bool(best and best["hollow"]),
    }
    return {"brick_components": len(groups), "best_frame": best or None, "atoms": atoms}


def _template(rows: Sequence[str]) -> set[Cell]:
    return {
        (row, col)
        for row, line in enumerate(rows)
        for col, value in enumerate(line)
        if value == "1"
    }


LETTER_TEMPLATES = {
    "S": _template(["1111", "1000", "1111", "0001", "1111"]),
    "M": _template(["10001", "11001", "10101", "10011", "10001"]),
    "B": _template(["1111", "1001", "1111", "1001", "1111"]),
}


def _normalize(mask: set[Cell]) -> set[Cell]:
    box = bounding_box(mask)
    if box is None:
        return set()
    top, _, left, _ = box
    return {(row - top, col - left) for row, col in mask}


def _scaled(mask: set[Cell], scale_y: int, scale_x: int) -> set[Cell]:
    return {
        (row * scale_y + dy, col * scale_x + dx)
        for row, col in mask
        for dy in range(scale_y)
        for dx in range(scale_x)
    }


def best_template_iou(observed: set[Cell], template: set[Cell]) -> float:
    normalized = _normalize(observed)
    if not normalized:
        return 0.0
    return max(
        mask_iou(normalized, _scaled(template, scale_y, scale_x))
        for scale_y in (1, 2, 3)
        for scale_x in (1, 2, 3)
    )


def _column_segments(mask: set[Cell]) -> list[tuple[int, int]]:
    columns = sorted({col for _, col in mask})
    if not columns:
        return []
    result: list[tuple[int, int]] = []
    start = previous = columns[0]
    for col in columns[1:]:
        if col != previous + 1:
            result.append((start, previous))
            start = col
        previous = col
    result.append((start, previous))
    return result


def t2_metrics(level: Sequence[str]) -> dict[str, Any]:
    bricks = positions(level, "B")
    segments = _column_segments(bricks)
    regions = [
        {(row, col) for row, col in bricks if left <= col <= right}
        for left, right in segments
    ]
    letters = ("S", "M", "B")
    scores = {
        letter: best_template_iou(region, LETTER_TEMPLATES[letter])
        for letter, region in zip(letters, regions, strict=False)
    }
    atoms = {
        "three_distinct_brick_characters": len(segments) == 3,
        "uppercase_s_shape": scores.get("S", 0.0) >= 0.5,
        "uppercase_m_shape": scores.get("M", 0.0) >= 0.5,
        "uppercase_b_shape": scores.get("B", 0.0) >= 0.5,
        "left_to_right_smb_order": len(segments) == 3 and all(scores.get(x, 0.0) >= 0.5 for x in letters),
    }
    return {"glyph_regions": len(segments), "template_iou": scores, "atoms": atoms}


def _horizontal_brick_runs(level: Sequence[str], minimum_length: int = 3) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row, line in enumerate(level):
        start: int | None = None
        for col, tile in enumerate(line + "."):
            if tile == "B" and start is None:
                start = col
            elif tile != "B" and start is not None:
                end = col - 1
                if end - start + 1 >= minimum_length:
                    runs.append({"row": row, "left": start, "right": end, "center": (start + end) / 2})
                start = None
    return runs


def _entity_on(level: Sequence[str], platform: Mapping[str, Any], entity: str) -> bool:
    row = int(platform["row"]) - 1
    return row >= 0 and any(level[row][col] == entity for col in range(platform["left"], platform["right"] + 1))


def t3_metrics(level: Sequence[str]) -> dict[str, Any]:
    platforms = _horizontal_brick_runs(level)
    staircases: list[list[dict[str, Any]]] = []
    for candidate in itertools.combinations(platforms, 3):
        if len({item["row"] for item in candidate}) != 3:
            continue
        ordered = sorted(candidate, key=lambda item: item["row"], reverse=True)
        centers = [item["center"] for item in ordered]
        if centers[0] < centers[1] < centers[2] or centers[0] > centers[1] > centers[2]:
            staircases.append(ordered)
    goomba_low = any(_entity_on(level, staircase[0], "G") for staircase in staircases)
    koopa_high = any(_entity_on(level, staircase[2], "K") for staircase in staircases)
    matching = [
        staircase
        for staircase in staircases
        if _entity_on(level, staircase[0], "G") and _entity_on(level, staircase[2], "K")
    ]
    atoms = {
        "three_distinct_brick_platforms": len(platforms) >= 3,
        "platforms_form_ascending_staircase": bool(staircases),
        "goomba_on_lowest_platform": goomba_low,
        "koopa_on_highest_platform": koopa_high,
    }
    return {
        "platform_count": len(platforms),
        "staircase_count": len(staircases),
        "matching_staircase_count": len(matching),
        "atoms": atoms,
    }


def t4_metrics(level: Sequence[str]) -> dict[str, Any]:
    brick_groups = components(positions(level, "B"))
    tower = brick_groups[0] if brick_groups else set()
    box = bounding_box(tower)
    height = width = None
    tower_pass = koopa_top = False
    if box:
        top, bottom, left, right = box
        height, width = bottom - top + 1, right - left + 1
        roof = all((top, col) in tower for col in range(left, right + 1))
        sides = all((row, left) in tower and (row, right) in tower for row in range(top, bottom + 1))
        tower_pass = height >= 5 and width >= 3 and roof and sides
        koopa_top = any(
            left <= col <= right and abs(row - top) <= 1
            for row, col in positions(level, "K")
        )
    atoms = {
        "one_tall_brick_tower": tower_pass,
        "at_least_one_koopa": bool(positions(level, "K")),
        "koopa_near_tower_top": koopa_top,
    }
    return {"tower_height": height, "tower_width": width, "atoms": atoms}


def _interior_walls(level: Sequence[str]) -> set[Cell]:
    height = len(level)
    width = len(level[0]) if level else 0
    return {
        (row, col)
        for row, col in positions(level, "#")
        if 0 < row < height - 1 and 0 < col < width - 1
    }


def _best_u_shape(level: Sequence[str]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for group in components(_interior_walls(level)):
        box = bounding_box(group)
        if box is None:
            continue
        top, bottom, left, right = box
        height, width = bottom - top + 1, right - left + 1
        if height < 4 or width < 5:
            continue
        left_ratio = mean((row, left) in group for row in range(top, bottom + 1))
        right_ratio = mean((row, right) in group for row in range(top, bottom + 1))
        bottom_ratio = mean((bottom, col) in group for col in range(left, right + 1))
        top_open_ratio = mean((top, col) not in group for col in range(left + 1, right))
        interior = {
            (row, col)
            for row in range(top + 1, bottom)
            for col in range(left + 1, right)
        }
        hollow_ratio = mean(cell not in group for cell in interior) if interior else 0.0
        score = mean((left_ratio, right_ratio, bottom_ratio, top_open_ratio, hollow_ratio))
        candidates.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "height": height,
                "width": width,
                "left_wall_ratio": left_ratio,
                "right_wall_ratio": right_ratio,
                "bottom_wall_ratio": bottom_ratio,
                "top_open_ratio": top_open_ratio,
                "hollow_ratio": hollow_ratio,
                "u_shape_score": score,
                "tiles": len(group),
            }
        )
    return max(candidates, key=lambda item: (item["u_shape_score"], item["tiles"]), default=None)


def zelda_t1_metrics(level: Sequence[str]) -> dict[str, Any]:
    best = _best_u_shape(level)
    walls = components(_interior_walls(level))
    u_shape = bool(
        best
        and best["left_wall_ratio"] >= 0.8
        and best["right_wall_ratio"] >= 0.8
        and best["bottom_wall_ratio"] >= 0.8
    )
    atoms = {
        "one_large_interior_wall_structure": bool(best and best["tiles"] >= 11),
        "u_shaped_courtyard": u_shape,
        "open_top_entrance": bool(best and best["top_open_ratio"] >= 0.6),
        "hollow_walkable_interior": bool(best and best["hollow_ratio"] >= 0.9),
    }
    return {"interior_wall_components": len(walls), "best_u_shape": best, "atoms": atoms}


Z_TEMPLATE = _template(["11111", "00011", "00110", "01100", "11111"])


def zelda_t2_metrics(level: Sequence[str]) -> dict[str, Any]:
    groups = components(_interior_walls(level))
    glyph = max(groups, key=lambda group: best_template_iou(group, Z_TEMPLATE), default=set())
    score = best_template_iou(glyph, Z_TEMPLATE)
    box = bounding_box(glyph)
    top_bar = bottom_bar = diagonal = False
    if box:
        top, bottom, left, right = box
        width = right - left + 1
        top_bar = mean((top, col) in glyph for col in range(left, right + 1)) >= 0.8
        bottom_bar = mean((bottom, col) in glyph for col in range(left, right + 1)) >= 0.8
        diagonal_hits = []
        span = max(1, bottom - top)
        for row in range(top + 1, bottom):
            expected = round(right - (row - top) * (right - left) / span)
            diagonal_hits.append(any((row, col) in glyph for col in range(max(left, expected - 1), min(right, expected + 1) + 1)))
        diagonal = bool(diagonal_hits) and mean(diagonal_hits) >= 0.7
    atoms = {
        "one_connected_interior_wall_glyph": bool(glyph and len(glyph) >= 11),
        "z_upper_horizontal_bar": top_bar,
        "z_descending_diagonal": diagonal,
        "z_lower_horizontal_bar": bottom_bar,
        "recognizable_uppercase_z": score >= 0.55 and top_bar and bottom_bar and diagonal,
    }
    return {"interior_wall_components": len(groups), "z_template_iou": score, "atoms": atoms}


def _single_position(level: Sequence[str], tile: str) -> Cell | None:
    found = positions(level, tile)
    return next(iter(found)) if len(found) == 1 else None


def zelda_t3_metrics(level: Sequence[str]) -> dict[str, Any]:
    player = _single_position(level, "P")
    key = _single_position(level, "K")
    door = _single_position(level, "D")
    enemies = positions(level, "E")
    enemy_rows = {row for row, _ in enemies}
    horizontal_line = len(enemies) == 3 and len(enemy_rows) == 1
    contiguous = False
    if horizontal_line:
        columns = sorted(col for _, col in enemies)
        contiguous = columns == list(range(columns[0], columns[0] + 3))
    atoms = {
        "player_left_of_key": bool(player and key and player[1] < key[1]),
        "key_left_of_door": bool(key and door and key[1] < door[1]),
        "three_enemy_horizontal_guard_line": horizontal_line and contiguous,
        "enemy_line_below_key": bool(key and horizontal_line and next(iter(enemy_rows)) > key[0]),
    }
    return {
        "player": player,
        "key": key,
        "door": door,
        "enemies": sorted(enemies),
        "atoms": atoms,
    }


def zelda_t4_metrics(level: Sequence[str]) -> dict[str, Any]:
    vault = _best_u_shape(level)
    key = _single_position(level, "K")
    door = _single_position(level, "D")
    enemies = positions(level, "E")
    u_shape = bool(vault and vault["u_shape_score"] >= 0.8)
    key_inside = bool(
        vault
        and key
        and vault["top"] < key[0] < vault["bottom"]
        and vault["left"] < key[1] < vault["right"]
    )
    door_at_entrance = bool(
        vault
        and door
        and abs(door[0] - vault["top"]) <= 1
        and vault["left"] < door[1] < vault["right"]
    )
    enemies_outside = bool(
        vault
        and len(enemies) == 3
        and all(row < vault["top"] and vault["left"] <= col <= vault["right"] for row, col in enemies)
    )
    atoms = {
        "u_shaped_wall_vault": u_shape,
        "key_inside_vault": key_inside,
        "door_at_open_entrance": door_at_entrance,
        "three_enemies_guard_outside": enemies_outside,
    }
    return {"best_u_shape": vault, "key": key, "door": door, "enemies": sorted(enemies), "atoms": atoms}


def _best_gated_courtyard(level: Sequence[str]) -> dict[str, Any] | None:
    """Find a hollow wall courtyard with one short, passable top-side gate.

    Unlike the earlier U-shape probe, the requested structure is an otherwise
    closed rectangular wall with a deliberate 1--3 tile entrance. The gap is
    part of the target geometry, not a failure: it lets the Zelda walkable
    region remain connected while the key sits inside the courtyard.
    """
    candidates: list[dict[str, Any]] = []
    for group in components(_interior_walls(level)):
        box = bounding_box(group)
        if box is None:
            continue
        top, bottom, left, right = box
        height, width = bottom - top + 1, right - left + 1
        if height < 6 or width < 7:
            continue

        top_inner = [(top, col) for col in range(left + 1, right)]
        missing_top = [cell for cell in top_inner if cell not in group]
        gap_columns = sorted(col for _, col in missing_top)
        one_gap = bool(gap_columns) and gap_columns == list(
            range(gap_columns[0], gap_columns[-1] + 1)
        ) and 1 <= len(gap_columns) <= 3

        left_ratio = mean((row, left) in group for row in range(top, bottom + 1))
        right_ratio = mean((row, right) in group for row in range(top, bottom + 1))
        bottom_ratio = mean((bottom, col) in group for col in range(left, right + 1))
        top_filled_ratio = mean(cell in group for cell in top_inner)
        interior = {
            (row, col)
            for row in range(top + 1, bottom)
            for col in range(left + 1, right)
        }
        hollow_ratio = mean(cell not in group for cell in interior) if interior else 0.0
        side_integrity = mean((left_ratio, right_ratio, bottom_ratio))
        score = mean((side_integrity, top_filled_ratio, hollow_ratio, float(one_gap)))
        candidates.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "height": height,
                "width": width,
                "tiles": len(group),
                "left_wall_ratio": left_ratio,
                "right_wall_ratio": right_ratio,
                "bottom_wall_ratio": bottom_ratio,
                "top_filled_ratio": top_filled_ratio,
                "hollow_ratio": hollow_ratio,
                "gate_cells": [(top, col) for col in gap_columns],
                "one_contiguous_gate": one_gap,
                "courtyard_score": score,
            }
        )
    return max(candidates, key=lambda item: (item["courtyard_score"], item["tiles"]), default=None)


def zelda_gated_courtyard_metrics(level: Sequence[str]) -> dict[str, Any]:
    courtyard = _best_gated_courtyard(level)
    player = _single_position(level, "P")
    key = _single_position(level, "K")
    door = _single_position(level, "D")
    enemies = positions(level, "E")
    gate_cells = set(courtyard["gate_cells"]) if courtyard else set()
    wall_shape = bool(
        courtyard
        and courtyard["left_wall_ratio"] >= 0.9
        and courtyard["right_wall_ratio"] >= 0.9
        and courtyard["bottom_wall_ratio"] >= 0.9
        and courtyard["top_filled_ratio"] >= 0.65
        and courtyard["hollow_ratio"] >= 0.95
    )
    key_inside = bool(
        courtyard
        and key
        and courtyard["top"] < key[0] < courtyard["bottom"]
        and courtyard["left"] < key[1] < courtyard["right"]
    )
    door_at_gate = bool(courtyard and door and door in gate_cells)
    guards_outside = bool(
        courtyard
        and len(enemies) == 3
        and all(
            row < courtyard["top"] and courtyard["left"] <= col <= courtyard["right"]
            for row, col in enemies
        )
    )
    player_outside = bool(
        courtyard
        and player
        and not (
            courtyard["top"] < player[0] < courtyard["bottom"]
            and courtyard["left"] < player[1] < courtyard["right"]
        )
    )
    atoms = {
        "large_hollow_walled_courtyard": wall_shape,
        "one_contiguous_passable_gate": bool(courtyard and courtyard["one_contiguous_gate"]),
        "key_inside_courtyard": key_inside,
        "door_at_gate": door_at_gate,
        "three_guards_outside_gate": guards_outside,
        "player_outside_courtyard": player_outside,
    }
    return {
        "best_gated_courtyard": courtyard,
        "player": player,
        "key": key,
        "door": door,
        "enemies": sorted(enemies),
        "atoms": atoms,
    }


EVALUATORS = {
    "t1_hollow_rectangle": t1_metrics,
    "t2_spell_smb": t2_metrics,
    "t3_staircase_entities": t3_metrics,
    "t4_rapunzel_koopa_tower": t4_metrics,
    "z_t1_u_courtyard": zelda_t1_metrics,
    "z_t2_wall_letter_z": zelda_t2_metrics,
    "z_t3_left_to_right_guard_line": zelda_t3_metrics,
    "z_t4_guarded_key_vault": zelda_t4_metrics,
    "z_gated_courtyard": zelda_gated_courtyard_metrics,
}


def _load_level(path: Path, height: int, width: int) -> list[str]:
    rows = path.read_text().splitlines()
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"expected {height}x{width} level at {path}")
    return rows


def _final_valid(trace_path: Path) -> bool | None:
    if not trace_path.is_file():
        return None
    result: bool | None = None
    for line in trace_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "final_level":
            result = bool(event.get("valid"))
    return result


def _functional_result(domain: str, final_metrics: Mapping[str, Any], valid: bool | None) -> dict[str, Any]:
    if domain == "smb":
        complete = float(final_metrics.get("complete", 0.0))
        return {"valid": valid, "complete": complete, "success": valid is True and complete >= 1.0}
    if domain == "zelda":
        tracked = {
            name: final_metrics.get(name)
            for name in ("players", "keys", "doors", "enemies", "player_key", "key_door", "regions")
        }
        success = (
            valid is True
            and tracked["players"] == 1
            and tracked["keys"] == 1
            and tracked["doors"] == 1
            and tracked["enemies"] == 3
            and float(tracked["player_key"] or 0) > 0
            and float(tracked["key_door"] or 0) > 0
            and tracked["regions"] == 1
        )
        return {"valid": valid, "success": success, "metrics": tracked}
    raise ValueError(f"unsupported free-form evaluation domain: {domain}")


def evaluate_cell(cell: Mapping[str, Any], case: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    cell_dir = Path(cell["run_dir"])
    level_path = cell_dir / "final_level.txt"
    summary_path = cell_dir / "summary.json"
    evaluator_name = str(case.get("evaluator", case["case_id"]))
    evaluator = EVALUATORS.get(evaluator_name)
    if evaluator is None:
        raise ValueError(f"no evaluator registered for {evaluator_name}")
    programmatic_atoms = list(case["programmatic_atoms"])
    if not level_path.is_file() or not summary_path.is_file():
        atoms = {atom: False for atom in programmatic_atoms}
        return {
            "cell_id": cell["cell_id"],
            "condition": cell["condition"],
            "case_id": case["case_id"],
            "tier": case["tier"],
            "seed": cell["seed"],
            "status": "missing_or_failed",
            "programmatic_atoms": atoms,
            "programmatic_score": 0.0,
            "programmatic_strict_success": False,
            "functional": {"valid": False, "success": False},
            "run_dir": str(cell_dir),
        }
    level = _load_level(level_path, int(environment["height"]), int(environment["width"]))
    details = evaluator(level)
    atoms = details.pop("atoms")
    if set(atoms) != set(programmatic_atoms):
        raise ValueError(f"{case['case_id']}: evaluator atom set differs from manifest")
    summary = json.loads(summary_path.read_text())
    final_metrics = summary.get("final_metrics") or {}
    valid = _final_valid(cell_dir / "trace.jsonl")
    functional = _functional_result(str(environment["domain"]), final_metrics, valid)
    return {
        "cell_id": cell["cell_id"],
        "condition": cell["condition"],
        "case_id": case["case_id"],
        "tier": case["tier"],
        "seed": cell["seed"],
        "status": "complete",
        "programmatic_atoms": atoms,
        "programmatic_score": mean(float(value) for value in atoms.values()),
        "programmatic_strict_success": all(atoms.values()),
        "programmatic_details": details,
        "judgment_atoms_pending": list(case.get("judgment_atoms", [])),
        "functional": functional,
        "run_dir": str(cell_dir),
    }


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    """Two-sided exact paired randomization p-value (descriptive at n=3)."""
    if not differences:
        return 1.0
    observed = abs(mean(differences))
    exceed = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        if abs(mean(sign * value for sign, value in zip(signs, differences, strict=True))) >= observed - 1e-12:
            exceed += 1
    return exceed / total


def summarize_records(suite: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(row["condition"], row["case_id"], int(row["seed"])): row for row in records}
    case_summaries: list[dict[str, Any]] = []
    all_deltas: list[float] = []
    for case in suite["cases"]:
        pairs: list[dict[str, Any]] = []
        for seed_value in suite["environment"]["replicate_seeds"]:
            seed = int(seed_value)
            matched = lookup[("matched", case["case_id"], seed)]
            control = lookup[("null", case["case_id"], seed)]
            delta = float(matched["programmatic_score"]) - float(control["programmatic_score"])
            all_deltas.append(delta)
            pairs.append(
                {
                    "seed": seed,
                    "matched_score": matched["programmatic_score"],
                    "null_score": control["programmatic_score"],
                    "paired_delta": delta,
                    "matched_strict": matched["programmatic_strict_success"],
                    "null_strict": control["programmatic_strict_success"],
                    "matched_functional": matched["functional"]["success"],
                    "null_functional": control["functional"]["success"],
                }
            )
        differences = [row["paired_delta"] for row in pairs]
        case_summaries.append(
            {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "programmatic_atom_coverage": len(case["programmatic_atoms"]),
                "judgment_atoms_pending": list(case.get("judgment_atoms", [])),
                "pairs": pairs,
                "mean_matched_score": mean(row["matched_score"] for row in pairs),
                "mean_null_score": mean(row["null_score"] for row in pairs),
                "mean_paired_delta": mean(differences),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences),
                "matched_strict_successes": sum(bool(row["matched_strict"]) for row in pairs),
                "null_strict_successes": sum(bool(row["null_strict"]) for row in pairs),
                "matched_functional_successes": sum(bool(row["matched_functional"]) for row in pairs),
                "null_functional_successes": sum(bool(row["null_functional"]) for row in pairs),
            }
        )
    matched_records = [row for row in records if row["condition"] == "matched"]
    null_records = [
        row for row in records if row["condition"] == "null"
    ]
    replicate_count = len(suite["environment"]["replicate_seeds"])
    minimum_two_sided_p = min(1.0, 2.0 ** (1 - replicate_count))
    return {
        "suite_id": suite["suite_id"],
        "claim_boundary": suite["claim_boundary"],
        "attempts": {
            "matched_expected": len(matched_records),
            "matched_complete": sum(row["status"] == "complete" for row in matched_records),
            "shared_null_expected": len(suite["environment"]["replicate_seeds"]),
            "shared_null_complete": len(
                {
                    row["cell_id"] for row in null_records if row["status"] == "complete"
                }
            ),
        },
        "overall": {
            "paired_comparisons": len(all_deltas),
            "mean_programmatic_paired_delta": mean(all_deltas),
            "positive_delta_pairs": sum(delta > 0 for delta in all_deltas),
            "zero_delta_pairs": sum(delta == 0 for delta in all_deltas),
            "negative_delta_pairs": sum(delta < 0 for delta in all_deltas),
            "matched_functional_successes": sum(row["functional"]["success"] for row in matched_records),
            "matched_functional_total": len(matched_records),
            "narrative_judgment_pending": any(case.get("judgment_atoms") for case in suite["cases"]),
        },
        "cases": case_summaries,
        "inference_note": (
            f"{replicate_count} replicate seed(s) provide a paired descriptive pilot. Exact "
            f"sign-flip p-values are reported, but the minimum attainable two-sided p-value "
            f"at n={replicate_count} is {minimum_two_sided_p:g}."
        ),
    }


def evaluate_plan(suite: Mapping[str, Any], plan: Mapping[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    null_cells = {int(cell["seed"]): cell for cell in plan["cells"] if cell["condition"] == "null"}
    records: list[dict[str, Any]] = []
    for case in suite["cases"]:
        for seed_value in suite["environment"]["replicate_seeds"]:
            seed = int(seed_value)
            matched = next(
                cell
                for cell in plan["cells"]
                if cell["condition"] == "matched" and cell["case_id"] == case["case_id"] and int(cell["seed"]) == seed
            )
            records.append(evaluate_cell(matched, case, suite["environment"]))
            records.append(evaluate_cell(null_cells[seed], case, suite["environment"]))
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite["suite_id"],
        "suite_sha256": plan["suite_sha256"],
        "failed_and_missing_attempts_remain_in_denominator": True,
        "records": records,
    }
    summary = summarize_records(suite, records)
    write_json(output_dir / "evaluation.json", evaluation)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_summary_markdown(summary))
    return evaluation, summary


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        f"# {summary['suite_id']}",
        "",
        str(summary["claim_boundary"]),
        "",
        "## Paired programmatic readout",
        "",
        "| Tier | Case | Matched | Null | Delta | Strict M/N | Functional M/N |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        n = len(case["pairs"])
        lines.append(
            f"| {case['tier']} | {case['case_id']} | {case['mean_matched_score']:.3f} | "
            f"{case['mean_null_score']:.3f} | {case['mean_paired_delta']:+.3f} | "
            f"{case['matched_strict_successes']}/{case['null_strict_successes']} of {n} | "
            f"{case['matched_functional_successes']}/{case['null_functional_successes']} of {n} |"
        )
    lines.extend(
        [
            "",
            f"Overall mean paired programmatic delta: `{overall['mean_programmatic_paired_delta']:+.3f}` "
            f"across `{overall['paired_comparisons']}` case-seed pairs.",
            "",
            str(summary["inference_note"]),
            "",
            "Narrative/theme atoms remain pending until blinded human or VLM judgments are added; "
            "they are not silently replaced by the tile-native score.",
            "",
        ]
    )
    return "\n".join(lines)


def export_blind_human_items(
    suite: Mapping[str, Any], plan: Mapping[str, Any], output_dir: Path, *, seed: int = 260826
) -> dict[str, Any]:
    rng = random.Random(seed)
    cells = {(cell["condition"], cell["case_id"], int(cell["seed"])): cell for cell in plan["cells"]}
    nulls = {int(cell["seed"]): cell for cell in plan["cells"] if cell["condition"] == "null"}
    items: list[dict[str, Any]] = []
    keys: dict[str, Any] = {}
    for case in suite["cases"]:
        for replicate in suite["environment"]["replicate_seeds"]:
            for condition in ("matched", "null"):
                seed_value = int(replicate)
                cell = cells[("matched", case["case_id"], seed_value)] if condition == "matched" else nulls[seed_value]
                image = Path(cell["run_dir"]) / "final_level.png"
                if not image.is_file():
                    continue
                item_id = hashlib.sha256(
                    f"{suite['suite_id']}\0{case['case_id']}\0{condition}\0{seed_value}".encode()
                ).hexdigest()[:16]
                candidates = [str(case["instruction"]), *map(str, case["hard_negatives"])]
                rng.shuffle(candidates)
                items.append(
                    {
                        "item_id": item_id,
                        "image": str(image.resolve()),
                        "question": case["human_question"],
                        "visible_criteria": case["required_atoms"],
                        "rating_scale": suite["rating_scale"],
                        "retrieval_candidates": candidates,
                    }
                )
                keys[item_id] = {
                    "condition": condition,
                    "case_id": case["case_id"],
                    "tier": case["tier"],
                    "seed": seed_value,
                    "intended_candidate_index": candidates.index(str(case["instruction"])),
                }
    rng.shuffle(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_human_eval.jsonl"
    blind_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))
    key_payload = {"blinding_seed": seed, "items": keys}
    write_json(output_dir / "blind_key.json", key_payload)
    return {"items": len(items), "blind_file": str(blind_path), "key_file": str(output_dir / "blind_key.json")}


def _load_frozen(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    suite_path = run_dir / "frozen_suite.yaml"
    plan_path = run_dir / "run_plan.json"
    if not suite_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("run directory needs frozen_suite.yaml and run_plan.json; run plan first")
    suite = load_suite(suite_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("suite_sha256") != sha256_json(suite):
        raise ValueError("frozen suite hash does not match run plan")
    return suite, plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--suite", type=Path, default=ROOT / "configs/high_level_freeform_zelda_t1_t4_32x32.yaml")
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--config", type=Path)
        command.add_argument("--provider")
        command.add_argument("--model")
        command.add_argument("--base-url")
        command.add_argument("--max-tokens", type=int)
        command.add_argument("--temperature", type=float)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    export = subparsers.add_parser("export-human")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--blinding-seed", type=int, default=260826)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--suite", type=Path, default=ROOT / "configs/high_level_freeform_zelda_t1_t4_32x32.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        suite = load_suite(args.suite.resolve())
        print(json.dumps({"suite_id": suite["suite_id"], "suite_sha256": sha256_json(suite)}, indent=2))
        return 0
    if args.command in {"plan", "run"}:
        suite_path = args.suite.resolve()
        suite = load_suite(suite_path)
        run_dir = args.run_dir.resolve()
        plan = build_plan(
            suite,
            run_dir,
            config_override=args.config,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        freeze_plan(suite_path, suite, plan, run_dir)
        result: dict[str, Any] = {
            "run_dir": str(run_dir),
            "attempts": plan["expected_generation_attempts"],
            "paired_comparisons": plan["expected_paired_comparisons"],
        }
        if args.command == "run":
            result.update(execute_plan(plan))
        print(json.dumps(result, indent=2))
        return 0
    suite, plan = _load_frozen(args.run_dir.resolve())
    if args.command == "evaluate":
        _, summary = evaluate_plan(suite, plan, args.run_dir.resolve())
        print(json.dumps(summary, indent=2))
        return 0
    result = export_blind_human_items(
        suite, plan, args.run_dir.resolve(), seed=args.blinding_seed
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

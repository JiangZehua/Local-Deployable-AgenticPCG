"""Readout for the Zelda open-prompt BSP tool-choice pilot."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Sequence


Cell = tuple[int, int]


def _positions(level: Sequence[str], tile: str) -> set[Cell]:
    return {
        (row, col)
        for row, line in enumerate(level)
        for col, value in enumerate(line)
        if value == tile
    }


def _components(mask: set[Cell]) -> list[set[Cell]]:
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


def detect_room_boxes(level: Sequence[str]) -> list[dict[str, int]]:
    """Detect open room bodies while excluding one-tile BSP corridors.

    A core cell has a fully passable 3x3 neighborhood. Rectangular BSP rooms
    retain connected cores after this erosion; one-tile corridors disappear.
    Expanding each core bounding box by one recovers the room interior.
    """
    height = len(level)
    width = len(level[0]) if level else 0
    passable = {
        (row, col)
        for row, line in enumerate(level)
        for col, tile in enumerate(line)
        if tile != "#"
    }
    core = {
        (row, col)
        for row, col in passable
        if 0 < row < height - 1
        and 0 < col < width - 1
        and all(
            (row + dy, col + dx) in passable
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
    }
    boxes: list[dict[str, int]] = []
    for group in _components(core):
        rows = [row for row, _ in group]
        cols = [col for _, col in group]
        boxes.append(
            {
                "top": max(1, min(rows) - 1),
                "bottom": min(height - 2, max(rows) + 1),
                "left": max(1, min(cols) - 1),
                "right": min(width - 2, max(cols) + 1),
                "core_tiles": len(group),
            }
        )
    return sorted(boxes, key=lambda box: (box["top"], box["left"]))


def room_distribution(level: Sequence[str]) -> dict[str, Any]:
    boxes = detect_room_boxes(level)
    enemies = _positions(level, "E")
    keys = _positions(level, "K")

    def inside(cell: Cell, box: dict[str, int]) -> bool:
        row, col = cell
        return box["top"] <= row <= box["bottom"] and box["left"] <= col <= box["right"]

    enemy_counts = [sum(inside(enemy, box) for enemy in enemies) for box in boxes]
    key_rooms = [index for index, box in enumerate(boxes) if any(inside(key, box) for key in keys)]
    return {
        "room_count": len(boxes),
        "room_boxes": boxes,
        "enemy_counts_by_room": enemy_counts,
        "rooms_with_exactly_one_enemy": sum(count == 1 for count in enemy_counts),
        "all_rooms_have_exactly_one_enemy": bool(boxes) and all(count == 1 for count in enemy_counts),
        "key_room_indices": key_rooms,
        "key_is_in_a_room": len(key_rooms) == 1,
        "enemy_count": len(enemies),
    }


def trace_tool_usage(trace_path: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    accepted_bsp_steps: list[int] = []
    for raw in trace_path.read_text().splitlines():
        event = json.loads(raw)
        if event.get("event") != "step":
            continue
        parsed = event.get("parsed_message") or {}
        step_calls = parsed.get("tool_calls") or []
        for index, call in enumerate(step_calls):
            calls.append(
                {
                    "step": event.get("step_num"),
                    "index": index,
                    "tool_name": call.get("tool_name"),
                    "parameters": call.get("parameters"),
                    "step_accepted": bool(event.get("accepted")),
                }
            )
        if event.get("accepted") and any(
            call.get("tool_name") == "generate_zelda_bsp" for call in step_calls
        ):
            accepted_bsp_steps.append(int(event["step_num"]))
    bsp_calls = [call for call in calls if call["tool_name"] == "generate_zelda_bsp"]
    manual_calls = [call for call in calls if call["tool_name"] == "place_tile"]
    return {
        "first_tool": calls[0]["tool_name"] if calls else None,
        "total_tool_calls": len(calls),
        "bsp_call_count": len(bsp_calls),
        "bsp_calls": bsp_calls,
        "accepted_bsp_steps": accepted_bsp_steps,
        "place_tile_call_count": len(manual_calls),
        "used_bsp_proactively": bool(bsp_calls),
    }


def evaluate_run(label: str, run_dir: Path) -> dict[str, Any]:
    level = (run_dir / "final_level.txt").read_text().splitlines()
    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = summary.get("final_metrics") or {}
    functional = (
        metrics.get("players") == 1
        and metrics.get("keys") == 1
        and metrics.get("doors") == 1
        and metrics.get("enemies") == 8
        and metrics.get("regions") == 1
        and float(metrics.get("player_key") or 0) > 0
        and float(metrics.get("key_door") or 0) > 0
    )
    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "tool_usage": trace_tool_usage(run_dir / "trace.jsonl"),
        "rooms": room_distribution(level),
        "functional_valid": functional,
        "final_metrics": metrics,
        "accepted_steps": summary.get("accepted_steps"),
        "rejected_steps": summary.get("rejected_steps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsp-run", type=Path, required=True)
    parser.add_argument("--manual-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "zelda_bsp_open_prompt_eval.v1",
        "arms": [
            evaluate_run("bsp_available", args.bsp_run),
            evaluate_run("manual_only", args.manual_run),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

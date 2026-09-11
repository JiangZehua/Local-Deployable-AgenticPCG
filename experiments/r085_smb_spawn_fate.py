"""R085-F2 conservative per-spawn fate attribution for SMB candidates."""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
from typing import Any, Mapping

from experiments.r080_smb_astar_probe import (
    TARGET_FIRST_ACCEPTANCE,
    TargetFirstSmbOptimizer,
)
from experiments.r083_smb_baseline import (
    DEFAULT_MANIFEST,
    preflight as baseline_preflight,
    run as baseline_run,
    smoke_case_ids,
)
from toolusepcg.smb_contract import (
    ALL_KILL_EVENTS,
    COUNTED_KILL_METHODS,
    FALL_KILL_EVENT,
    SMB_COIN_EVENT_PARAMS,
    SMB_ENGINE_PADDING_TILES,
    SMB_SPRITE_NAMES,
    COLLECT_EVENT,
)


TILE_TO_SPRITE = {7: 2, 8: 6, 9: 8}


def _engine_spawn_position(spawn: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(spawn["x"]) + SMB_ENGINE_PADDING_TILES) * 16.0 + 8.0,
        float(spawn["y"]) * 16.0 + 15.0,
    )


def _tile_position(x: float, y: float, *, width: int, height: int) -> dict[str, int]:
    return {
        "x": max(0, min(width - 1, int(x / 16) - SMB_ENGINE_PADDING_TILES)),
        "y": max(0, min(height - 1, int(y / 16))),
    }


def attribute_enemy_spawn_fates(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Track enemy observations and conservatively attach kill/fall events.

    The engine exposes anonymous ``[sprite_type, x, y]`` observations per
    frame. Tracks therefore use deterministic nearest-neighbor association and
    expose that method; unmatched disappearances are never mislabeled kills.
    """
    width = int(info.get("width", 32))
    height = int(info.get("height", 16))
    spawns = list((info.get("spawn_positions") or {}).get("enemies") or [])
    tracks = []
    for index, spawn in enumerate(spawns):
        sprite_type = TILE_TO_SPRITE.get(int(spawn.get("tile", -1)))
        tracks.append(
            {
                "spawn_id": f"enemy-{index}",
                "spawn_position": {"x": int(spawn["x"]), "y": int(spawn["y"])},
                "tile": int(spawn.get("tile", -1)),
                "sprite_type": sprite_type,
                "sprite_name": SMB_SPRITE_NAMES.get(sprite_type, "unknown"),
                "observations": [],
            }
        )

    for frame_index, frame in enumerate(info.get("enemy_frames") or []):
        observations = [
            (int(item[0]), float(item[1]), float(item[2]))
            for item in (frame or [])
            if len(item) >= 3
        ]
        unmatched = set(range(len(tracks)))
        for sprite_type, x, y in observations:
            candidates = [
                index
                for index in unmatched
                if tracks[index]["sprite_type"] == sprite_type
            ]
            if not candidates:
                continue
            def distance(index: int) -> float:
                prior = tracks[index]["observations"]
                anchor = (
                    (prior[-1]["x"], prior[-1]["y"])
                    if prior else _engine_spawn_position(spawns[index])
                )
                return hypot(x - anchor[0], y - anchor[1])
            matched = min(candidates, key=lambda index: (distance(index), index))
            tracks[matched]["observations"].append(
                {"frame": frame_index, "x": x, "y": y}
            )
            unmatched.remove(matched)

    locations = list(info.get("locations") or [])
    for track in tracks:
        closest = None
        for observation in track["observations"]:
            frame = int(observation["frame"])
            if frame >= len(locations):
                continue
            mario_x, mario_y = map(float, locations[frame][:2])
            distance_tiles = hypot(
                float(observation["x"]) - mario_x,
                float(observation["y"]) - mario_y,
            ) / 16.0
            candidate = (distance_tiles, frame, mario_x, mario_y, observation)
            if closest is None or candidate[:2] < closest[:2]:
                closest = candidate
        track["closest_approach_tiles"] = (
            round(float(closest[0]), 3) if closest is not None else None
        )
        track["closest_approach_frame"] = (
            int(closest[1]) if closest is not None else None
        )
        track["observed_frames"] = len(track["observations"])
        track["first_observed_frame"] = (
            track["observations"][0]["frame"] if track["observations"] else None
        )
        track["last_observed_frame"] = (
            track["observations"][-1]["frame"] if track["observations"] else None
        )
        track["last_observed_position"] = (
            _tile_position(
                track["observations"][-1]["x"],
                track["observations"][-1]["y"],
                width=width,
                height=height,
            )
            if track["observations"] else None
        )
        track["fate"] = None
        track["fate_event_position"] = None

    for event in info.get("game_events") or []:
        event_type = int(event.get("event_type", event.get("type", -1)))
        if event_type not in ALL_KILL_EVENTS:
            continue
        sprite_type = int(event.get("event_param", event.get("param", -1)))
        event_time = int(event.get("time", 0))
        candidates = [
            track for track in tracks
            if track["fate"] is None
            and track["sprite_type"] == sprite_type
            and track["observations"]
        ]
        if not candidates:
            continue
        def event_distance(track: dict[str, Any]) -> tuple[float, float, str]:
            nearest = min(
                track["observations"],
                key=lambda item: abs(int(item["frame"]) - event_time),
            )
            temporal = abs(int(nearest["frame"]) - event_time)
            spatial = hypot(
                float(nearest["x"]) - float(event.get("mario_x", 0.0)),
                float(nearest["y"]) - float(event.get("mario_y", 0.0)),
            )
            return temporal, spatial, str(track["spawn_id"])
        matched = min(candidates, key=event_distance)
        matched["fate"] = (
            "fell_excluded" if event_type == FALL_KILL_EVENT
            else COUNTED_KILL_METHODS.get(event_type, "killed")
        )
        matched["fate_event_position"] = dict(
            event.get("mario_event_position") or {}
        )

    mario_max_x = -1
    for location in locations:
        mario_max_x = max(
            mario_max_x,
            int(float(location[0]) / 16) - SMB_ENGINE_PADDING_TILES,
        )
    for track in tracks:
        if track["fate"] is None:
            if not track["observations"] and track["spawn_position"]["x"] > mario_max_x + 2:
                track["fate"] = "unreached"
            elif track["observations"]:
                track["fate"] = "avoided_or_survived"
            else:
                track["fate"] = "unobserved_near_trajectory"
        track["attribution_method"] = "anonymous-frame-nearest-neighbor-v1"
        del track["observations"]
    return tracks


def attribute_coin_spawn_fates(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Match coin pickup events one-to-one to nearest remaining spawn tile."""
    spawns = list((info.get("spawn_positions") or {}).get("coins") or [])
    rows = [
        {
            "spawn_id": f"coin-{index}",
            "spawn_position": {"x": int(spawn["x"]), "y": int(spawn["y"])},
            "fate": "uncollected",
            "pickup_mario_position": None,
            "attribution_method": "nearest-unmatched-spawn-v1",
        }
        for index, spawn in enumerate(spawns)
    ]
    available = set(range(len(rows)))
    events = [
        event for event in (info.get("game_events") or [])
        if int(event.get("event_type", event.get("type", -1))) == COLLECT_EVENT
        and int(event.get("event_param", event.get("param", -1)))
        in SMB_COIN_EVENT_PARAMS
    ]
    for event in events:
        if not available:
            break
        position = event.get("mario_event_position") or {}
        x, y = int(position.get("x", 0)), int(position.get("y", 0))
        matched = min(
            available,
            key=lambda index: (
                abs(rows[index]["spawn_position"]["x"] - x)
                + abs(rows[index]["spawn_position"]["y"] - y),
                index,
            ),
        )
        rows[matched]["fate"] = "collected"
        rows[matched]["pickup_mario_position"] = {"x": x, "y": y}
        available.remove(matched)
    return rows


class SpawnFateSmbOptimizer(TargetFirstSmbOptimizer):
    """R082 feedback plus explicit, conservative per-spawn fate records."""

    def _build_environment_step_feedback(self, new_level, new_result, tool_results):
        base = super()._build_environment_step_feedback(
            new_level, new_result, tool_results
        )
        info = new_result.info or {}
        lines = [base, "Enemy spawn fates (candidate-local):"]
        enemies = attribute_enemy_spawn_fates(info)
        if not enemies:
            lines.append("- none")
        for item in enemies:
            spawn = item["spawn_position"]
            closest = item["closest_approach_tiles"]
            closest_text = "unknown" if closest is None else f"{closest:.3f} tiles"
            lines.append(
                f"- {item['spawn_id']} {item['sprite_name']} at "
                f"[{spawn['x']},{spawn['y']}]: {item['fate']}; "
                f"closest Mario approach={closest_text}; "
                f"last={item['last_observed_position']}"
            )
        lines.append("Coin spawn fates (candidate-local):")
        coins = attribute_coin_spawn_fates(info)
        if not coins:
            lines.append("- none")
        for item in coins:
            spawn = item["spawn_position"]
            lines.append(
                f"- {item['spawn_id']} at [{spawn['x']},{spawn['y']}]: "
                f"{item['fate']}; pickup Mario position="
                f"{item['pickup_mario_position']}"
            )
        lines.append(
            "- Fate attribution is conservative: avoided_or_survived and "
            "unobserved_near_trajectory are not kills."
        )
        return "\n".join(lines)


def build_spawn_fate_optimizer(
    config: Any,
    *,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
) -> SpawnFateSmbOptimizer:
    if acceptance_policy != TARGET_FIRST_ACCEPTANCE:
        raise ValueError(
            "R085-F2 requires target-first acceptance; "
            f"got {acceptance_policy!r}"
        )
    return SpawnFateSmbOptimizer(config, verbose=False)


def preflight(
    manifest_path: Path,
    output: Path,
    *,
    case_ids: list[str] | None = None,
    seed: int = 83083,
) -> dict[str, Any]:
    return baseline_preflight(
        manifest_path,
        output,
        case_ids=case_ids,
        seed=seed,
        optimizer_builder=build_spawn_fate_optimizer,
        metadata={
            "experiment": "R085-F2",
            "feedback": "conservative-spawn-fate-v1",
        },
    )


def run(
    manifest_path: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    max_protocol_retries: int = 4,
    case_ids: list[str] | None = None,
    seed: int = 83083,
) -> dict[str, Any]:
    return baseline_run(
        manifest_path,
        base_url,
        output,
        model_id=model_id,
        revision=revision,
        max_steps=max_steps,
        max_protocol_retries=max_protocol_retries,
        case_ids=case_ids,
        seed=seed,
        optimizer_builder=build_spawn_fate_optimizer,
        harness_overrides={
            "baseline": "R084-P3 bounded protocol",
            "feedback": "R085-F2 conservative spawn fate",
            "single_changed_factor": "candidate-local spawn fate feedback",
        },
        experiment_name="R085-F2",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=83083)
        command.add_argument("--smoke", action="store_true")
        command.add_argument("--case-id", action="append")
        if name == "run":
            command.add_argument("--base-url", required=True)
            command.add_argument("--model-id", required=True)
            command.add_argument("--revision", required=True)
            command.add_argument("--max-steps", type=int, default=4)
            command.add_argument("--max-protocol-retries", type=int, default=4)
    args = parser.parse_args()
    case_ids = args.case_id
    if args.smoke:
        if case_ids:
            raise ValueError("--smoke and --case-id are mutually exclusive")
        case_ids = list(smoke_case_ids())
    if args.command == "preflight":
        result = preflight(
            args.manifest, args.output, case_ids=case_ids, seed=args.seed
        )
    else:
        result = run(
            args.manifest,
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            max_protocol_retries=args.max_protocol_retries,
            case_ids=case_ids,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Transactional semantic operator for strict mixed SMB behavior episodes."""

from __future__ import annotations

import numpy as np

from ..level import Level, SMBTiles
from .registry import EditTool, ToolResult


SUPPORTED_BEHAVIOR_TARGETS: dict[tuple[int, int, int], dict[str, tuple]] = {
    (1, 0, 1): {
        "enemies": (17,), "gaps": ((14, 17),), "obstacles": (), "coins": (),
    },
    (1, 1, 1): {
        "enemies": (17,), "gaps": ((14, 17),), "obstacles": (),
        "coins": ((2, 14),),
    },
    (1, 1, 2): {
        "enemies": (17,), "gaps": ((14, 17),), "obstacles": (9,),
        "coins": ((2, 14),),
    },
    (1, 2, 3): {
        "enemies": (17,), "gaps": ((14, 17),), "obstacles": (9, 20),
        "coins": ((2, 14), (5, 14)),
    },
    (2, 1, 2): {
        "enemies": (22, 28), "gaps": ((19, 22), (25, 28)),
        "obstacles": (9,), "coins": ((2, 14),),
    },
    (2, 5, 1): {
        "enemies": (22, 28), "gaps": ((19, 22), (25, 28)),
        "obstacles": (),
        "coins": ((2, 14), (5, 14), (9, 14), (13, 14), (16, 14)),
    },
}


class BuildSmbBehaviorEpisodeTool(EditTool):
    """Compile one R102-certified strict mixed behavior episode."""

    @property
    def name(self) -> str:
        return "build_smb_behavior_episode"

    @property
    def description(self) -> str:
        return (
            "Atomically rebuild a 16x32 dynamic SMB level for one certified "
            "mixed (enemy_count, coin_count, jump_count) target. Supported "
            "triples are (1,0,1), (1,1,1), (1,1,2), (1,2,3), (2,1,2), "
            "and (2,5,1). Every accepted episode must pass actual A* exact "
            "verification, stomp-only counted kills with zero fall kills, and "
            "ground-jump causal verification. Use exactly one call."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enemy_count": {"type": "integer", "enum": [1, 2]},
                "coin_count": {"type": "integer", "enum": [0, 1, 2, 5]},
                "jump_count": {"type": "integer", "enum": [1, 2, 3]},
            },
            "required": ["enemy_count", "coin_count", "jump_count"],
        }

    def execute(self, **kwargs) -> ToolResult:
        try:
            key = tuple(
                int(kwargs.get(name))
                for name in ("enemy_count", "coin_count", "jump_count")
            )
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                errors=["enemy_count, coin_count, and jump_count must be integers"],
            )
        if key not in SUPPORTED_BEHAVIOR_TARGETS:
            return ToolResult(
                success=False,
                errors=[
                    f"Unsupported certified behavior target {key}; supported: "
                    f"{sorted(SUPPORTED_BEHAVIOR_TARGETS)}"
                ],
            )
        manager = self._edit_manager
        if manager.problem_type != "smb":
            return ToolResult(
                success=False,
                errors=["build_smb_behavior_episode is only valid for dynamic SMB"],
            )
        if (manager.height, manager.width) != (16, 32):
            return ToolResult(
                success=False,
                errors=["build_smb_behavior_episode requires a 16x32 level"],
            )

        profile = SUPPORTED_BEHAVIOR_TARGETS[key]
        target = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
        target[15, :] = int(SMBTiles.SOLID)
        for start, stop in profile["gaps"]:
            target[15, int(start):int(stop)] = int(SMBTiles.EMPTY)
        for x in profile["enemies"]:
            target[14, int(x)] = int(SMBTiles.GOOMBA)
        for x in profile["obstacles"]:
            target[14, int(x)] = int(SMBTiles.SOLID)
        for x, y in profile["coins"]:
            target[int(y), int(x)] = int(SMBTiles.COIN)

        before = manager.level_data
        changed = np.argwhere(before != target)
        diff = [
            {
                "y": int(y),
                "x": int(x),
                "old": manager.tile_names.get(int(before[y, x]), str(int(before[y, x]))),
                "new": manager.tile_names.get(int(target[y, x]), str(int(target[y, x]))),
            }
            for y, x in changed
        ]
        checkpoint = manager.checkpoint()
        try:
            manager.set_level(Level(target, problem_type="smb"))
        except Exception as exc:
            manager.rollback(checkpoint)
            return ToolResult(
                success=False,
                errors=[f"transaction rolled back: {type(exc).__name__}: {exc}"],
            )
        return ToolResult(
            success=True,
            data={
                "new_level": manager.level.to_string(),
                "applied": True,
                "transactional": True,
                "operator_contract": "r102-certified-strict-mixed-episode-v1",
                "requested_behavior": {
                    "enemies": key[0], "coins": key[1], "jumps": key[2],
                },
                "enemy_x": list(profile["enemies"]),
                "gaps": [list(gap) for gap in profile["gaps"]],
                "obstacle_x": list(profile["obstacles"]),
                "coin_sites": [list(site) for site in profile["coins"]],
                "num_tiles_changed": int(len(changed)),
                "diff": diff,
            },
        )

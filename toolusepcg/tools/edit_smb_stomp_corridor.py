"""Transactional semantic operator for certified SMB stomp corridors."""

from __future__ import annotations

import numpy as np

from ..level import Level, SMBTiles
from .registry import EditTool, ToolResult


SUPPORTED_STOMP_TARGETS: dict[tuple[int, int], dict[str, tuple[int, ...]]] = {
    (1, 0): {"enemies": (10,), "coins": ()},
    (1, 5): {"enemies": (10,), "coins": (2, 5, 9, 14, 17)},
    (2, 4): {"enemies": (10, 20), "coins": (2, 5, 9, 13)},
    (2, 5): {"enemies": (10, 20), "coins": (2, 5, 9, 13, 17)},
    (3, 2): {"enemies": (6, 14, 22), "coins": (2, 5)},
    (3, 3): {"enemies": (6, 14, 22), "coins": (2, 5, 9)},
}


class BuildSmbStompCorridorTool(EditTool):
    """Atomically compile one R100-certified kill/coin/jump-zero layout."""

    @property
    def name(self) -> str:
        return "build_smb_stomp_corridor"

    @property
    def description(self) -> str:
        return (
            "Atomically rebuild a 16x32 dynamic SMB level as a verified "
            "low-ceiling stomp corridor for an exact jump target of 0. "
            "Supported (enemy_count, coin_count) pairs are "
            "(1,0), (1,5), (2,4), (2,5), (3,2), and (3,3). Use exactly "
            "one call and do not combine it with raw tile edits in the same "
            "STEP. The optimizer accepts it only after actual A* stomp-event "
            "verification."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "enemy_count": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "Exact number of enemies that must be stomped.",
                },
                "coin_count": {
                    "type": "integer",
                    "enum": [0, 2, 3, 4, 5],
                    "description": "Exact number of trajectory coins to collect.",
                },
            },
            "required": ["enemy_count", "coin_count"],
        }

    def execute(self, **kwargs) -> ToolResult:
        try:
            enemy_count = int(kwargs.get("enemy_count"))
            coin_count = int(kwargs.get("coin_count"))
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                errors=["enemy_count and coin_count must be integers"],
            )
        key = (enemy_count, coin_count)
        if key not in SUPPORTED_STOMP_TARGETS:
            return ToolResult(
                success=False,
                errors=[
                    f"Unsupported certified stomp target {key}; supported pairs: "
                    f"{sorted(SUPPORTED_STOMP_TARGETS)}"
                ],
            )
        manager = self._edit_manager
        if manager.problem_type != "smb":
            return ToolResult(
                success=False,
                errors=["build_smb_stomp_corridor is only valid for dynamic SMB"],
            )
        if (manager.height, manager.width) != (16, 32):
            return ToolResult(
                success=False,
                errors=[
                    "build_smb_stomp_corridor requires a 16x32 level; got "
                    f"{manager.height}x{manager.width}"
                ],
            )

        profile = SUPPORTED_STOMP_TARGETS[key]
        target = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
        target[12, :] = int(SMBTiles.SOLID)
        target[14:16, :] = int(SMBTiles.SOLID)
        for x in profile["enemies"]:
            target[13, x] = int(SMBTiles.GOOMBA)
        for x in profile["coins"]:
            target[13, x] = int(SMBTiles.COIN)

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
                "operator_contract": "r100-certified-stomp-corridor-v1",
                "requested_behavior": {
                    "enemies": enemy_count,
                    "coins": coin_count,
                    "jumps": 0,
                },
                "enemy_x": list(profile["enemies"]),
                "coin_x": list(profile["coins"]),
                "num_tiles_changed": int(len(changed)),
                "diff": diff,
            },
        )

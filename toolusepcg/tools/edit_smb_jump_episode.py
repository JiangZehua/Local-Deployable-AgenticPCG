"""Transactional semantic operator for certified SMB jump episodes."""

from __future__ import annotations

import numpy as np

from ..level import Level, SMBTiles
from .registry import EditTool, ToolResult


SUPPORTED_JUMP_TARGETS: dict[tuple[int, int], dict[str, tuple]] = {
    (0, 1): {"obstacles": (), "coins": ()},
    (1, 2): {"obstacles": (11,), "coins": ((2, 14),)},
    (2, 3): {"obstacles": (11, 27), "coins": ((2, 14), (5, 14))},
    (3, 4): {
        "obstacles": (7, 19, 27),
        "coins": ((2, 14), (5, 14), (9, 10)),
    },
    (4, 5): {
        "obstacles": (5, 11, 20, 26),
        "coins": ((2, 14), (5, 11), (9, 14), (13, 12)),
    },
    (5, 5): {
        "obstacles": (5, 11, 20, 26),
        "coins": ((2, 14), (5, 11), (9, 14), (13, 12), (17, 14)),
    },
}


class BuildSmbJumpEpisodeTool(EditTool):
    """Compile one cause-labeled, R101-certified jump-only episode."""

    @property
    def name(self) -> str:
        return "build_smb_jump_episode"

    @property
    def description(self) -> str:
        return (
            "Atomically rebuild a 16x32 dynamic SMB level for an exact "
            "enemy_count=0 jump episode. Supported (coin_count, jump_count) "
            "pairs are (0,1), (1,2), (2,3), (3,4), (4,5), and (5,5). "
            "The causal contract is one terminal jump plus jump_count-1 "
            "forced obstacle jumps, with no gap jump, stomp bounce, or kill. "
            "Use exactly one call and do not mix it with raw tile edits."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "coin_count": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3, 4, 5],
                    "description": "Exact number of trajectory coins to collect.",
                },
                "jump_count": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "Exact number of ground-jump events.",
                },
            },
            "required": ["coin_count", "jump_count"],
        }

    def execute(self, **kwargs) -> ToolResult:
        try:
            coin_count = int(kwargs.get("coin_count"))
            jump_count = int(kwargs.get("jump_count"))
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                errors=["coin_count and jump_count must be integers"],
            )
        key = (coin_count, jump_count)
        if key not in SUPPORTED_JUMP_TARGETS:
            return ToolResult(
                success=False,
                errors=[
                    f"Unsupported certified jump target {key}; supported pairs: "
                    f"{sorted(SUPPORTED_JUMP_TARGETS)}"
                ],
            )
        manager = self._edit_manager
        if manager.problem_type != "smb":
            return ToolResult(
                success=False,
                errors=["build_smb_jump_episode is only valid for dynamic SMB"],
            )
        if (manager.height, manager.width) != (16, 32):
            return ToolResult(
                success=False,
                errors=[
                    "build_smb_jump_episode requires a 16x32 level; got "
                    f"{manager.height}x{manager.width}"
                ],
            )

        profile = SUPPORTED_JUMP_TARGETS[key]
        target = np.full((16, 32), int(SMBTiles.EMPTY), dtype=np.int32)
        target[15, :] = int(SMBTiles.SOLID)
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
                "operator_contract": "r101-certified-jump-episode-v1",
                "requested_behavior": {
                    "enemies": 0,
                    "coins": coin_count,
                    "jumps": jump_count,
                },
                "obstacle_x": list(profile["obstacles"]),
                "coin_sites": [list(site) for site in profile["coins"]],
                "num_tiles_changed": int(len(changed)),
                "diff": diff,
            },
        )

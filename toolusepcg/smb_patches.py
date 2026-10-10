"""Composable 16-row SMB patches for variable-width dynamic levels.

A patch is a parametric column block that shares one boundary contract with
every other patch: solid floor in rows 14-15 at both edges, Mario walking on
row 13, and an optional low ceiling on row 12.  Patches are concatenated left
to right; the assembled level is padded on the left with corridor columns up
to ``MIN_WIDTH``.

Two family sets exist. ``v1`` (PAPER-M8) uses a two-row floor and low
corridor ceilings; a ceiling hides Mario's takeoffs from the engine's JUMP
event. ``v2`` (PAPER-M8b) has no ceilings and a one-row floor (row 15, as in
the r083 atlas), so every takeoff can be audited as a counted ground jump
(``toolusepcg.smb_takeoff_audit``).

Patch families only declare geometry plus an *intended* gameplay role.  Event
counts are never read from a family declaration: they are measured by the A*
evaluator, either during library certification or during search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from toolusepcg.level import Level, SMBTiles


HEIGHT = 16
MIN_WIDTH = 32
FLOOR_TOP_Y = 14
WALK_Y = 13
CEILING_Y = 12

_EMPTY = int(SMBTiles.EMPTY)
_SOLID = int(SMBTiles.SOLID)
_COIN = int(SMBTiles.COIN)
_GOOMBA = int(SMBTiles.GOOMBA)


def _open(width: int) -> np.ndarray:
    data = np.full((HEIGHT, width), _EMPTY, dtype=np.int32)
    data[FLOOR_TOP_Y:, :] = _SOLID
    return data


def _corridor(width: int) -> np.ndarray:
    data = _open(width)
    data[CEILING_Y, :] = _SOLID
    return data


def _render_corridor(p: Mapping[str, int]) -> np.ndarray:
    return _corridor(p["width"])


def _render_open(p: Mapping[str, int]) -> np.ndarray:
    return _open(p["width"])


def _render_stomp(p: Mapping[str, int]) -> np.ndarray:
    data = _corridor(p["width"])
    data[WALK_Y, p["enemy_x"]] = _GOOMBA
    return data


def _render_coins(p: Mapping[str, int]) -> np.ndarray:
    data = _corridor(p["spacing"] * p["count"] + 2)
    for index in range(p["count"]):
        data[WALK_Y, 1 + p["spacing"] * index] = _COIN
    return data


def _render_obstacle(p: Mapping[str, int]) -> np.ndarray:
    data = _open(p["width"])
    data[FLOOR_TOP_Y - p["height"] : FLOOR_TOP_Y, p["x"]] = _SOLID
    return data


def _render_gap(p: Mapping[str, int]) -> np.ndarray:
    data = _open(p["width"])
    data[FLOOR_TOP_Y:, p["x"] : p["x"] + p["gap"]] = _EMPTY
    return data


def _render_open_goomba(p: Mapping[str, int]) -> np.ndarray:
    data = _open(p["width"])
    data[WALK_Y, p["enemy_x"]] = _GOOMBA
    return data


def _render_pit_goomba(p: Mapping[str, int]) -> np.ndarray:
    data = _open(p["width"])
    data[FLOOR_TOP_Y:, p["pit_x"] : p["pit_x"] + p["pit"]] = _EMPTY
    data[WALK_Y, p["enemy_x"]] = _GOOMBA
    return data


FLOOR_Y = 15  # v2: one-row floor


def _flat(width: int) -> np.ndarray:
    data = np.full((HEIGHT, width), _EMPTY, dtype=np.int32)
    data[FLOOR_Y, :] = _SOLID
    return data


def _render_flat(p: Mapping[str, int]) -> np.ndarray:
    return _flat(p["width"])


def _render_obstacle15(p: Mapping[str, int]) -> np.ndarray:
    data = _flat(p["width"])
    data[FLOOR_Y - p["height"] : FLOOR_Y, p["x"]] = _SOLID
    return data


def _render_gap15(p: Mapping[str, int]) -> np.ndarray:
    data = _flat(p["width"])
    data[FLOOR_Y, p["x"] : p["x"] + p["gap"]] = _EMPTY
    return data


def _render_coins15(p: Mapping[str, int]) -> np.ndarray:
    data = _flat(p["spacing"] * p["count"] + 2)
    for index in range(p["count"]):
        data[p["row"], 1 + p["spacing"] * index] = _COIN
    return data


def _render_stomp_cluster(p: Mapping[str, int]) -> np.ndarray:
    """``runway`` flat tiles, then ``count`` units of [pit][goomba on floor][gap floor]."""
    unit = p["pit"] + 1 + p["gap"]
    data = _flat(p["runway"] + p["count"] * unit + 1)
    for index in range(p["count"]):
        x = p["runway"] + index * unit
        data[FLOOR_Y, x : x + p["pit"]] = _EMPTY
        data[FLOOR_Y - 1, x + p["pit"]] = _GOOMBA
    return data


def _render_stomp_pillar(p: Mapping[str, int]) -> np.ndarray:
    """``runway`` flat, pit, a raised pillar carrying a goomba, pit, landing."""
    w = p["runway"] + p["pit"] + p["pillar"] + p["pit"] + 2
    data = _flat(w)
    x = p["runway"]
    data[FLOOR_Y, x : x + p["pit"]] = _EMPTY
    px = x + p["pit"]
    data[FLOOR_Y - p["height"] : FLOOR_Y, px : px + p["pillar"]] = _SOLID
    data[FLOOR_Y - p["height"] - 1, px] = _GOOMBA
    data[FLOOR_Y, px + p["pillar"] : px + p["pillar"] + p["pit"]] = _EMPTY
    return data


def _render_open_coins(p: Mapping[str, int]) -> np.ndarray:
    data = _open(p["spacing"] * p["count"] + 2)
    for index in range(p["count"]):
        data[p["row"], 1 + p["spacing"] * index] = _COIN
    return data


@dataclass(frozen=True)
class Family:
    name: str
    role: str
    render: Callable[[Mapping[str, int]], np.ndarray]
    # Inclusive parameter ranges used by search mutation; constraints between
    # parameters are enforced by ``valid_params``.
    ranges: Mapping[str, tuple[int, int]]
    valid_params: Callable[[Mapping[str, int]], bool]


FAMILIES: dict[str, Family] = {
    family.name: family
    for family in (
        Family("corridor", "filler", _render_corridor, {"width": (2, 12)},
               lambda p: True),
        Family("open", "filler", _render_open, {"width": (2, 12)},
               lambda p: True),
        Family("stomp", "enemies", _render_stomp,
               {"width": (6, 16), "enemy_x": (2, 14)},
               lambda p: 2 <= p["enemy_x"] <= p["width"] - 2),
        Family("coins", "coins", _render_coins,
               {"count": (1, 5), "spacing": (1, 3)}, lambda p: True),
        Family("obstacle", "jumps", _render_obstacle,
               {"width": (4, 12), "height": (1, 3), "x": (1, 11)},
               lambda p: 1 <= p["x"] <= p["width"] - 1),
        Family("gap", "jumps", _render_gap,
               {"width": (4, 12), "gap": (1, 4), "x": (1, 10)},
               lambda p: 1 <= p["x"] and p["x"] + p["gap"] <= p["width"] - 1),
        Family("open_goomba", "enemies", _render_open_goomba,
               {"width": (6, 16), "enemy_x": (2, 14)},
               lambda p: 2 <= p["enemy_x"] <= p["width"] - 2),
        Family("pit_goomba", "enemies", _render_pit_goomba,
               {"width": (8, 16), "pit": (1, 4), "pit_x": (1, 8), "enemy_x": (3, 14)},
               lambda p: (1 <= p["pit_x"] and p["pit_x"] + p["pit"] < p["enemy_x"]
                          <= p["width"] - 2)),
        Family("open_coins", "coins", _render_open_coins,
               {"count": (1, 5), "spacing": (1, 3), "row": (9, 13)}, lambda p: True),
        # v2 (ceiling-free, one-row floor)
        Family("flat", "filler", _render_flat, {"width": (2, 24)}, lambda p: True),
        Family("obstacle15", "jumps", _render_obstacle15,
               {"width": (4, 12), "height": (1, 3), "x": (1, 11)},
               lambda p: 1 <= p["x"] <= p["width"] - 1),
        Family("gap15", "jumps", _render_gap15,
               {"width": (4, 12), "gap": (1, 4), "x": (1, 10)},
               lambda p: 1 <= p["x"] and p["x"] + p["gap"] <= p["width"] - 1),
        Family("coins15", "coins", _render_coins15,
               {"count": (1, 5), "spacing": (1, 3), "row": (11, 14)}, lambda p: True),
        Family("stomp_cluster", "enemies", _render_stomp_cluster,
               {"count": (1, 4), "pit": (2, 4), "gap": (1, 3), "runway": (2, 16)}, lambda p: True),
        Family("stomp_pillar", "enemies", _render_stomp_pillar,
               {"runway": (2, 12), "pit": (1, 3), "pillar": (1, 3), "height": (1, 2)}, lambda p: True),
    )
}
FAMILY_SETS = {
    "v1": ("corridor", "open", "stomp", "coins", "obstacle", "gap", "open_goomba", "pit_goomba", "open_coins"),
    "v2": ("flat", "obstacle15", "gap15", "coins15", "stomp_cluster", "stomp_pillar"),
}


@dataclass(frozen=True)
class Patch:
    family: str
    params: tuple[tuple[str, int], ...]

    @classmethod
    def make(cls, family: str, **params: int) -> "Patch":
        patch = cls(family, tuple(sorted((k, int(v)) for k, v in params.items())))
        patch.validate()
        return patch

    @property
    def p(self) -> dict[str, int]:
        return dict(self.params)

    @property
    def role(self) -> str:
        return FAMILIES[self.family].role

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown patch family: {self.family!r}")
        family = FAMILIES[self.family]
        params = self.p
        if set(params) != set(family.ranges):
            raise ValueError(f"{self.family} expects params {sorted(family.ranges)}")
        for name, (lower, upper) in family.ranges.items():
            if not lower <= params[name] <= upper:
                raise ValueError(f"{self.family}.{name}={params[name]} outside [{lower},{upper}]")
        if not family.valid_params(params):
            raise ValueError(f"inconsistent {self.family} params: {params}")

    def render(self) -> np.ndarray:
        return FAMILIES[self.family].render(self.p)

    @property
    def width(self) -> int:
        return int(self.render().shape[1])

    def token(self) -> str:
        return self.family + "(" + ",".join(f"{k}={v}" for k, v in self.params) + ")"

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "params": self.p}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Patch":
        return cls.make(str(payload["family"]), **dict(payload["params"]))


def assemble(patches: Sequence[Patch], *, min_width: int = MIN_WIDTH, pad: str = "corridor") -> Level:
    """Concatenate patches left to right; left-pad to min_width (v1 corridor, v2 flat)."""
    blocks = [patch.render() for patch in patches]
    data = np.concatenate(blocks, axis=1) if blocks else np.zeros((HEIGHT, 0), np.int32)
    if data.shape[1] < min_width:
        filler = {"corridor": _corridor, "flat": _flat}[pad]
        data = np.concatenate([filler(min_width - data.shape[1]), data], axis=1)
    return Level(data.astype(np.int32), problem_type="smb")


def plan_payload(patches: Iterable[Patch]) -> list[dict[str, Any]]:
    return [patch.as_dict() for patch in patches]


def plan_sha256(patches: Iterable[Patch]) -> str:
    return hashlib.sha256(
        json.dumps(plan_payload(patches), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def level_sha256(level: Level) -> str:
    return hashlib.sha256(level.to_string().encode()).hexdigest()


class WidthEvaluatorPool:
    """One frozen A* StatsEvaluator per level width (engine timer scales with W)."""

    def __init__(self) -> None:
        self._evaluators: dict[int, Any] = {}

    def get(self, width: int):
        if width not in self._evaluators:
            from toolusepcg.config import load_config
            from toolusepcg.evaluator import StatsEvaluator
            from toolusepcg.factory import LevelFactory

            config = load_config(
                Path(__file__).resolve().parents[1] / "configs/smb.yaml",
                {
                    "env": {"height": HEIGHT, "width": int(width), "smb_solver": "astar"},
                    "targets": {
                        "complete": {"mode": "target", "value": 1.0},
                        "enemies": {"mode": "target", "value": 0},
                        "coins": {"mode": "target", "value": 0},
                        "jumps": {"mode": "target", "value": 0},
                    },
                },
            )
            self._evaluators[width] = StatsEvaluator(config, LevelFactory(config).env)
        return self._evaluators[width]

    def evaluate(self, level: Level):
        if level.height != HEIGHT:
            raise ValueError(f"dynamic SMB levels must have {HEIGHT} rows")
        return self.get(level.width).evaluate(level)

    def evaluate_audited(self, level: Level):
        """Evaluate and attach ``info['takeoff_audit']`` (physical takeoffs vs JUMP events)."""
        from toolusepcg.smb_takeoff_audit import evaluate_with_takeoff_audit

        return evaluate_with_takeoff_audit(self.evaluate, level)

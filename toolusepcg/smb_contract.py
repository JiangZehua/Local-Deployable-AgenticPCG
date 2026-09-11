"""Frozen SMB mastery contract and semantic event adapter.

The bundled Mario engine reports event coordinates in padded engine pixels and
uses integer event codes.  This module is the single adapter boundary between
that engine representation and the 16x32 level-space contract used by the
mastery harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np

from pcg_benchmark.probs.smbtile.engine.core import MarioForwardModel
from pcg_benchmark.probs.smbtile.engine.helper import EventType, SpriteType


SMB_EVENT_ADAPTER_VERSION = "smb-event-v1"
SMB_ENGINE_PADDING_TILES = 3
SMB_REQUIRED_AGENT = "astar"
# The engine emits the raw level block id (15) for a directly collected tile
# coin, while one legacy query path compares against OBS_COIN (31).  The
# adapter recognizes both and keeps this inconsistency out of harness code.
SMB_COIN_EVENT_PARAM = 15
SMB_COIN_EVENT_PARAMS = frozenset(
    {SMB_COIN_EVENT_PARAM, int(MarioForwardModel.OBS_COIN)}
)

SMB_EVENT_ENUM = {
    name: int(member.value) for name, member in EventType.__members__.items()
}
SMB_EVENT_NAMES = {value: name for name, value in SMB_EVENT_ENUM.items()}
SMB_SPRITE_NAMES = {
    int(member.value): name.lower()
    for name, member in SpriteType.__members__.items()
    if isinstance(member.value, int)
}

STOMP_KILL_EVENT = SMB_EVENT_ENUM["STOMP_KILL"]
FIRE_KILL_EVENT = SMB_EVENT_ENUM["FIRE_KILL"]
SHELL_KILL_EVENT = SMB_EVENT_ENUM["SHELL_KILL"]
FALL_KILL_EVENT = SMB_EVENT_ENUM["FALL_KILL"]
JUMP_EVENT = SMB_EVENT_ENUM["JUMP"]
LAND_EVENT = SMB_EVENT_ENUM["LAND"]
COLLECT_EVENT = SMB_EVENT_ENUM["COLLECT"]

# Gameplay kill metric deliberately excludes FALL_KILL: only kills caused by
# Mario mechanics count toward the mastery target.
COUNTED_KILL_EVENTS = frozenset(
    {STOMP_KILL_EVENT, FIRE_KILL_EVENT, SHELL_KILL_EVENT}
)
ALL_KILL_EVENTS = frozenset({*COUNTED_KILL_EVENTS, FALL_KILL_EVENT})
COUNTED_KILL_METHODS = {
    STOMP_KILL_EVENT: "stomped",
    FIRE_KILL_EVENT: "fire-killed",
    SHELL_KILL_EVENT: "shell-killed",
}

SMB_TARGET_METRICS = ("enemies", "coins", "jumps")
SMB_TARGET_ENVELOPES: dict[str, dict[str, tuple[int, int]]] = {
    "native": {
        "enemies": (0, 4),
        "coins": (0, 3),
        "jumps": (0, 6),
    },
    "expanded": {
        "enemies": (0, 4),
        "coins": (0, 5),
        "jumps": (0, 6),
    },
}


@dataclass(frozen=True)
class SmbTargetError:
    """Exact target error under the mastery contract."""

    e_k: float
    e_c: float
    e_j: float

    @property
    def e_inf(self) -> float:
        return max(self.e_k, self.e_c, self.e_j)

    @property
    def e_sum(self) -> float:
        return self.e_k + self.e_c + self.e_j

    @property
    def exact(self) -> bool:
        return self.e_inf == 0

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "e_k": self.e_k,
            "e_c": self.e_c,
            "e_j": self.e_j,
            "e_inf": self.e_inf,
            "e_sum": self.e_sum,
            "exact": self.exact,
        }


def smb_target_error(
    metrics: Mapping[str, int | float],
    targets: Mapping[str, int | float],
) -> SmbTargetError:
    """Return exact per-axis errors; no tolerance or approximate semantics."""
    return SmbTargetError(
        e_k=abs(float(metrics.get("enemies", 0)) - float(targets["enemies"])),
        e_c=abs(float(metrics.get("coins", 0)) - float(targets["coins"])),
        e_j=abs(float(metrics.get("jumps", 0)) - float(targets["jumps"])),
    )


def get_smb_target_envelope(name: str = "native") -> dict[str, tuple[int, int]]:
    """Return a copy of a named, explicit SMB mastery target envelope."""
    try:
        return dict(SMB_TARGET_ENVELOPES[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown SMB target envelope {name!r}; expected one of "
            f"{sorted(SMB_TARGET_ENVELOPES)}"
        ) from exc


def iter_smb_targets(name: str = "native") -> Iterable[dict[str, int]]:
    """Enumerate every target in an envelope in deterministic axis order."""
    envelope = get_smb_target_envelope(name)
    ranges = {
        metric: range(bounds[0], bounds[1] + 1)
        for metric, bounds in envelope.items()
    }
    for enemies, coins, jumps in product(
        ranges["enemies"], ranges["coins"], ranges["jumps"]
    ):
        yield {"enemies": enemies, "coins": coins, "jumps": jumps}


def validate_smb_target_value(
    metric: str,
    value: Any,
    *,
    envelope: str = "expanded",
) -> int | float:
    """Fail closed for exact SMB targets outside the selected envelope."""
    if metric == "complete":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("SMB exact target 'complete' must be numeric 1.0")
        if not isfinite(float(value)) or float(value) != 1.0:
            raise ValueError(
                "SMB exact target 'complete' is out_of_contract; expected 1.0"
            )
        return float(value)
    if metric not in SMB_TARGET_METRICS:
        raise ValueError(f"Unknown SMB mastery metric {metric!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SMB exact target '{metric}' must be an integer")
    numeric = float(value)
    if not isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"SMB exact target '{metric}' must be a finite integer")
    lower, upper = get_smb_target_envelope(envelope)[metric]
    integer = int(numeric)
    if not lower <= integer <= upper:
        raise ValueError(
            f"SMB exact target '{metric}'={integer} is out_of_contract for "
            f"the {envelope} envelope [{lower}, {upper}]"
        )
    return integer


def _bounded_tile_position(
    mario_x: Any,
    mario_y: Any,
    *,
    width: int,
    height: int,
) -> dict[str, int]:
    """Convert padded engine pixels to a bounded level-space Mario position."""
    x = int(float(mario_x) / 16) - SMB_ENGINE_PADDING_TILES
    y = int(float(mario_y) / 16)
    return {
        "x": max(0, min(width - 1, x)),
        "y": max(0, min(height - 1, y)),
    }


def normalize_smb_event(
    event: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Name an engine event and attach its bounded Mario event position."""
    event_type = int(event.get("type", -1))
    event_param = int(event.get("param", 0))
    position = _bounded_tile_position(
        event.get("mario_x", 0.0),
        event.get("mario_y", 0.0),
        width=width,
        height=height,
    )
    normalized = dict(event)
    normalized.update(
        {
            "event_type": event_type,
            "event_type_name": SMB_EVENT_NAMES.get(event_type, "UNKNOWN"),
            "event_param": event_param,
            "mario_event_position": position,
            "counted_enemy_kill": event_type in COUNTED_KILL_EVENTS,
            "fall_kill_excluded": event_type == FALL_KILL_EVENT,
            "coin_pickup": (
                event_type == COLLECT_EVENT and event_param in SMB_COIN_EVENT_PARAMS
            ),
            "ground_jump": event_type == JUMP_EVENT,
            "stomp_bounce": event_type == STOMP_KILL_EVENT,
        }
    )
    return normalized


def normalized_smb_event_trace(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the compact, stable semantic trace used by determinism audits."""
    trace = []
    for event in info.get("game_events", []):
        trace.append(
            {
                "event_type": int(event.get("event_type", event.get("type", -1))),
                "event_type_name": event.get("event_type_name", "UNKNOWN"),
                "event_param": int(event.get("event_param", event.get("param", 0))),
                "mario_event_position": dict(event["mario_event_position"]),
                "time": int(event.get("time", 0)),
            }
        )
    return trace


def adapt_smb_info(
    raw_info: Mapping[str, Any],
    level_data: np.ndarray,
    *,
    problem: Any,
) -> dict[str, Any]:
    """Expose engine gates/config and normalize events without mutating raw info."""
    info = dict(raw_info)
    height, width = (int(level_data.shape[0]), int(level_data.shape[1]))
    tube_issues = int(info.get("tube", 0))
    empty_ratio = float(info.get("empty", 0.0))
    floating_ratio = float(info.get("fenemies", 0.0))
    empty_threshold = float(getattr(problem, "_empty", 0.5))
    floating_threshold = float(getattr(problem, "_fenemies", 0.1))

    gate_status = {
        "tube": {
            "value": tube_issues,
            "required": 0,
            "passed": tube_issues == 0,
        },
        "empty_ratio": {
            "value": empty_ratio,
            "operator": ">",
            "threshold": empty_threshold,
            "passed": empty_ratio > empty_threshold,
        },
        "floating_enemy_ratio": {
            "value": floating_ratio,
            "operator": "<",
            "threshold": floating_threshold,
            "passed": floating_ratio < floating_threshold,
        },
    }
    skip_reasons = [name for name, gate in gate_status.items() if not gate["passed"]]
    engine_executed = not skip_reasons

    events = [
        normalize_smb_event(event, width=width, height=height)
        for event in info.get("game_events", [])
    ]
    coin_spawns = []
    enemy_spawns = []
    for y, row in enumerate(level_data):
        for x, tile in enumerate(row):
            value = int(tile)
            if value == 6:
                coin_spawns.append({"x": x, "y": y})
            elif value in {7, 8, 9}:
                enemy_spawns.append({"x": x, "y": y, "tile": value})

    info.update(
        {
            "width": width,
            "height": height,
            "game_events": events,
            "normalized_event_trace": normalized_smb_event_trace(
                {"game_events": events}
            ),
            "engine_executed": engine_executed,
            "engine_skip_reason": ",".join(skip_reasons) if skip_reasons else None,
            "engine_skip_reasons": skip_reasons,
            "engine_gate_status": gate_status,
            "floating_enemy_ratio": floating_ratio,
            "spawn_positions": {
                "coins": coin_spawns,
                "enemies": enemy_spawns,
            },
            "event_coordinate_semantics": "mario_event_position",
            "event_adapter_version": SMB_EVENT_ADAPTER_VERSION,
            "engine_contract": {
                "agent": getattr(problem, "_agent", None),
                "timer": getattr(problem, "_timer", None),
                "solver_expansions": getattr(problem, "_solver", None),
                "padding_tiles": SMB_ENGINE_PADDING_TILES,
                "event_enum": dict(SMB_EVENT_ENUM),
            },
        }
    )
    return info


def smb_metrics_are_finite(metrics: Mapping[str, Any]) -> bool:
    """Check all mastery metrics came back as finite numeric values."""
    for name in ("complete", *SMB_TARGET_METRICS):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not isfinite(float(value)):
            return False
    return True

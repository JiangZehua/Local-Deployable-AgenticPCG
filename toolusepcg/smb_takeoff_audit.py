"""Audit physical ground takeoffs against the engine's JUMP events.

The engine emits ``JUMP`` only when a ground takeoff has clear headroom
(``Mario.update``: ``elif self.onGround and self.mayJump`` sets ``jumpTime`` and
then checks for blocking tiles above Mario's head). A takeoff under a low
ceiling therefore leaves the ground without being counted. This module
observes the live game world (never the A* search clones) and records every
takeoff, so a level can be required to have ``hidden_takeoffs == 0``: each
takeoff is a counted ground jump. It changes no physics and no actions.
"""

from __future__ import annotations

from typing import Any, Callable

from pcg_benchmark.probs.smbtile.engine.core import MarioGame
from pcg_benchmark.probs.smbtile.engine.helper import MarioActions
from pcg_benchmark.probs.smbtile.engine.sprites import Mario

from toolusepcg.smb_contract import JUMP_EVENT

AUDIT_VERSION = "smb-takeoff-audit-v1"
_JUMP = MarioActions.JUMP.value


def evaluate_with_takeoff_audit(evaluate: Callable[[Any], Any], level: Any) -> Any:
    """Run ``evaluate(level)`` and attach ``info['takeoff_audit']``."""
    state: dict[str, Any] = {"game": None}
    takeoffs: list[dict[str, Any]] = []
    original_run, original_update = MarioGame.runGame, Mario.update

    def observed_run(game, *args, **kwargs):
        state["game"] = game
        return original_run(game, *args, **kwargs)

    def observed_update(mario, *args, **kwargs):
        game = state["game"]
        live = game is not None and mario.world is getattr(game, "_world", None)
        # Same branch condition as the engine's takeoff, read before the update.
        takeoff = bool(live and mario.alive and mario.actions[_JUMP] and mario.jumpTime >= 0
                       and mario.onGround and mario.mayJump)
        if takeoff:
            takeoffs.append({"tick": int(mario.world.currentTick), "x": float(mario.x), "y": float(mario.y)})
        return original_update(mario, *args, **kwargs)

    MarioGame.runGame, Mario.update = observed_run, observed_update
    try:
        result = evaluate(level)
    finally:
        MarioGame.runGame, Mario.update = original_run, original_update
    info = result.info if result.info is not None else {}
    jump_ticks = sorted(int(e.get("time", -1)) for e in info.get("game_events", [])
                        if int(e.get("event_type", e.get("type", -1))) == JUMP_EVENT)
    remaining = list(jump_ticks)
    hidden = []
    for takeoff in takeoffs:
        if takeoff["tick"] in remaining:
            remaining.remove(takeoff["tick"])
        else:
            hidden.append(takeoff)
    info["takeoff_audit"] = {
        "version": AUDIT_VERSION,
        "takeoffs": len(takeoffs),
        "jump_events": len(jump_ticks),
        "hidden_takeoffs": len(hidden),
        "unmatched_jump_events": len(remaining),
        "hidden": hidden,
    }
    result.info = info
    return result


def takeoffs_clean(result: Any) -> bool:
    audit = (result.info or {}).get("takeoff_audit")
    return bool(audit) and audit["hidden_takeoffs"] == 0 and audit["unmatched_jump_events"] == 0

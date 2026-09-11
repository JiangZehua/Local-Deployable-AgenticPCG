"""Interaction protocol version selection."""

from __future__ import annotations

import os


INTERACTION_PROTOCOL_ENV = "RPE_PCG_INTERACTION_PROTOCOL"


def effective_interaction_protocol() -> str:
    """Return the effective interaction protocol for the current environment."""
    return "v2" if os.environ.get(INTERACTION_PROTOCOL_ENV) == "v2" else "v1"


def interaction_protocol_v2_enabled() -> bool:
    """Return whether interaction protocol v2 is enabled."""
    return effective_interaction_protocol() == "v2"

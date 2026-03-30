from __future__ import annotations

"""Action vocabulary shared by the deterministic engine and learned models."""

from enum import StrEnum


class Action(StrEnum):
    """Canonical control actions used throughout the project."""

    NONE = "none"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

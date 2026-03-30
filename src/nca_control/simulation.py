from __future__ import annotations

"""Small text-mode helpers for deterministic scripted rollouts."""

from collections.abc import Iterable

from .actions import Action
from .grid import GridState, step_grid


def parse_actions(raw_actions: str) -> list[Action]:
    """Parse a comma-separated action list used by the CLI and tests."""

    if not raw_actions.strip():
        return []
    return [Action(part.strip().lower()) for part in raw_actions.split(",")]


def rollout_states(initial_state: GridState, actions: Iterable[Action]) -> list[GridState]:
    """Replay the deterministic reference dynamics and keep every intermediate state."""

    states = [initial_state]
    current = initial_state
    for action in actions:
        current = step_grid(current, action)
        states.append(current)
    return states

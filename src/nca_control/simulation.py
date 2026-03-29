from __future__ import annotations

from collections.abc import Iterable

from .actions import Action
from .grid import GridState, step_grid


def parse_actions(raw_actions: str) -> list[Action]:
    if not raw_actions.strip():
        return []
    return [Action(part.strip().lower()) for part in raw_actions.split(",")]


def rollout_states(initial_state: GridState, actions: Iterable[Action]) -> list[GridState]:
    states = [initial_state]
    current = initial_state
    for action in actions:
        current = step_grid(current, action)
        states.append(current)
    return states


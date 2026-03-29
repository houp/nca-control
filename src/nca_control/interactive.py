from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock

import torch

from .actions import Action
from .grid import GridState, step_grid
from .inference import predict_next_state


def action_from_keysym(keysym: str) -> Action | None:
    normalized = keysym.lower()
    if normalized in {"up", "arrowup"}:
        return Action.UP
    if normalized in {"down", "arrowdown"}:
        return Action.DOWN
    if normalized in {"left", "arrowleft"}:
        return Action.LEFT
    if normalized in {"right", "arrowright"}:
        return Action.RIGHT
    if normalized in {"space", " "}:
        return Action.NONE
    return None

def prediction_to_grid_state(
    prediction: torch.Tensor,
    *,
    value: float = 1.0,
    blocked: frozenset[tuple[int, int]] = frozenset(),
    exit_cell: tuple[int, int] | None = None,
    exit_fill: frozenset[tuple[int, int]] | None = None,
    terminated: bool = False,
) -> GridState:
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError("prediction must have shape [1, height, width]")
    height, width = prediction.shape[1], prediction.shape[2]
    flat_index = int(torch.argmax(prediction[0]).item())
    row = flat_index // width
    col = flat_index % width
    return GridState(
        height=height,
        width=width,
        row=row,
        col=col,
        value=value,
        blocked=blocked,
        exit_cell=exit_cell,
        exit_fill=exit_fill,
        terminated=terminated,
    )


@dataclass(slots=True)
class InteractiveCompareSession:
    checkpoint_path: str
    initial_state: GridState
    device: str = "auto"
    reset_factory: Callable[[], GridState] | None = field(default=None, repr=False)
    reference_state: GridState = field(init=False)
    model_state: GridState = field(init=False)
    last_action: Action = field(init=False)
    version: int = field(init=False)
    _lock: RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = RLock()
        self.reference_state = self.initial_state
        self.model_state = self.initial_state
        self.last_action = Action.NONE
        self.version = 0

    def reset(self) -> dict[str, object]:
        with self._lock:
            if self.reset_factory is not None:
                self.initial_state = self.reset_factory()
            self.reference_state = self.initial_state
            self.model_state = self.initial_state
            self.last_action = Action.NONE
            self.version += 1
            return self._snapshot_unlocked()

    def apply_action(self, action: Action) -> dict[str, object]:
        with self._lock:
            self.last_action = action
            self.reference_state = step_grid(self.reference_state, action)
            if self.model_state.terminated:
                self.model_state = step_grid(self.model_state, action)
            else:
                prediction = predict_next_state(
                    self.checkpoint_path,
                    self.model_state,
                    action,
                    device=self.device,
                    hard_decode=True,
                )
                predicted_state = prediction_to_grid_state(
                    prediction,
                    value=self.model_state.value,
                    blocked=self.model_state.blocked,
                    exit_cell=self.model_state.exit_cell,
                    exit_fill=self.model_state.exit_fill,
                )
                if predicted_state.exit_cell is not None and (predicted_state.row, predicted_state.col) == predicted_state.exit_cell:
                    predicted_state = GridState(
                        height=predicted_state.height,
                        width=predicted_state.width,
                        row=predicted_state.row,
                        col=predicted_state.col,
                        value=predicted_state.value,
                        blocked=predicted_state.blocked,
                        exit_cell=predicted_state.exit_cell,
                        exit_fill=predicted_state.exit_fill,
                        terminated=True,
                    )
                self.model_state = predicted_state
            self.version += 1
            return self._snapshot_unlocked()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, object]:
        mismatch = not _states_match(self.reference_state, self.model_state)
        return {
            "version": self.version,
            "last_action": self.last_action.value,
            "reference": serialize_grid_state(self.reference_state),
            "model": serialize_grid_state(self.model_state),
            "match": not mismatch,
        }


def serialize_grid_state(state: GridState) -> dict[str, object]:
    return {
        "height": state.height,
        "width": state.width,
        "row": state.row,
        "col": state.col,
        "value": state.value,
        "blocked": [[row, col] for row, col in sorted(state.blocked)],
        "exit_fill": [[row, col] for row, col in sorted(state.exit_fill or frozenset())],
        "exit_cell": list(state.exit_cell) if state.exit_cell is not None else None,
        "terminated": state.terminated,
    }


def _states_match(reference: GridState, model: GridState) -> bool:
    return (
        reference.row == model.row
        and reference.col == model.col
        and reference.terminated == model.terminated
        and reference.exit_fill == model.exit_fill
        and reference.exit_cell == model.exit_cell
        and reference.blocked == model.blocked
    )

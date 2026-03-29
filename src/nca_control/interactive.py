from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .actions import Action
from .grid import GridState
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


def prediction_to_grid_state(prediction: torch.Tensor, *, value: float = 1.0) -> GridState:
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError("prediction must have shape [1, height, width]")
    height, width = prediction.shape[1], prediction.shape[2]
    flat_index = int(torch.argmax(prediction[0]).item())
    row = flat_index // width
    col = flat_index % width
    return GridState(height=height, width=width, row=row, col=col, value=value)


@dataclass(slots=True)
class InteractiveCompareSession:
    checkpoint_path: str
    initial_state: GridState
    device: str = "auto"
    reference_state: GridState = field(init=False)
    model_state: GridState = field(init=False)
    last_action: Action = field(init=False)

    def __post_init__(self) -> None:
        self.reference_state = self.initial_state
        self.model_state = self.initial_state
        self.last_action = Action.NONE

    def reset(self) -> dict[str, object]:
        self.reference_state = self.initial_state
        self.model_state = self.initial_state
        self.last_action = Action.NONE
        return self.snapshot()

    def apply_action(self, action: Action) -> dict[str, object]:
        from .grid import step_grid

        self.last_action = action
        self.reference_state = step_grid(self.reference_state, action)
        prediction = predict_next_state(
            self.checkpoint_path,
            self.model_state,
            action,
            device=self.device,
            hard_decode=True,
        )
        self.model_state = prediction_to_grid_state(prediction, value=self.model_state.value)
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        mismatch = (self.reference_state.row, self.reference_state.col) != (
            self.model_state.row,
            self.model_state.col,
        )
        return {
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
    }

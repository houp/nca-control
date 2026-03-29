from __future__ import annotations

import torch

from .actions import Action
from .grid import GridState


def action_from_keysym(keysym: str) -> Action | None:
    normalized = keysym.lower()
    if normalized == "up":
        return Action.UP
    if normalized == "down":
        return Action.DOWN
    if normalized == "left":
        return Action.LEFT
    if normalized == "right":
        return Action.RIGHT
    if normalized == "space":
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

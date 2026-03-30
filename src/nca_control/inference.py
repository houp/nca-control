from __future__ import annotations

"""Checkpoint loading and hard-decoding helpers for interactive use."""

from pathlib import Path

import torch

from .actions import Action
from .dataset import encode_control_input
from .device import resolve_device
from .grid import GridState, step_grid
from .model import ControllableNCAModel


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[ControllableNCAModel, dict[str, object], torch.device]:
    """Restore a PyTorch checkpoint together with its saved configuration."""

    resolved_device = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = dict(payload["config"])
    input_channels = int(payload["model_state_dict"]["perception.weight"].shape[1])
    model = ControllableNCAModel(
        input_channels=input_channels,
        state_channels=int(config.get("state_channels", 1)),
        hidden_channels=int(config["hidden_channels"]),
        perception_kernel_size=int(config.get("perception_kernel_size", 3)),
        update_kernel_size=int(config.get("update_kernel_size", 1)),
        cell_value_max=float(config["value"]),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, resolved_device


def predict_next_state(
    checkpoint_path: str | Path,
    state: GridState,
    action: Action,
    device: str = "auto",
    hard_decode: bool = True,
) -> torch.Tensor:
    """Run one learned transition step for a deterministic state/action pair."""

    model, config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    include_exit_dynamics = int(config.get("state_channels", 1)) > 1
    model_input = encode_control_input(
        state,
        action,
        device=resolved_device,
        include_exit_dynamics=include_exit_dynamics,
    ).unsqueeze(0)
    with torch.no_grad():
        prediction = model(model_input)
    prediction = prediction.squeeze(0).cpu()
    if hard_decode:
        if prediction.shape[0] == 2:
            return hard_decode_exit_prediction(prediction)
        return hard_decode_grid(prediction)
    return prediction


def hard_decode_grid(grid: torch.Tensor) -> torch.Tensor:
    """Project a plain single-channel prediction back to one active cell."""

    if grid.ndim != 3 or grid.shape[0] != 1:
        raise ValueError("grid must have shape [1, height, width]")
    value = float(grid.sum().item())
    decoded = torch.zeros_like(grid)
    flat_index = int(torch.argmax(grid[0]).item())
    width = grid.shape[-1]
    row = flat_index // width
    col = flat_index % width
    decoded[0, row, col] = value
    return decoded


def hard_decode_exit_prediction(prediction: torch.Tensor, active_threshold: float = 0.5) -> torch.Tensor:
    """Project exit-aware predictions back to legal active/exit-fill channels."""

    if prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError("prediction must have shape [2, height, width]")
    decoded = torch.zeros_like(prediction)
    active = prediction[0]
    exit_fill = prediction[1]
    active_max = float(active.max().item())
    if active_max >= active_threshold:
        flat_index = int(torch.argmax(active).item())
        width = prediction.shape[-1]
        row = flat_index // width
        col = flat_index % width
        decoded[0, row, col] = 1.0
    decoded[1] = (exit_fill >= active_threshold).to(decoded.dtype)
    return decoded


def decode_prediction_state(
    prediction: torch.Tensor,
    previous_state: GridState,
    *,
    active_threshold: float = 0.5,
) -> GridState:
    """Convert a model prediction back into a valid ``GridState``."""

    if prediction.ndim != 3:
        raise ValueError("prediction must have shape [channels, height, width]")

    if prediction.shape[0] == 1:
        flat_index = int(torch.argmax(prediction[0]).item())
        row = flat_index // previous_state.width
        col = flat_index % previous_state.width
        return GridState(
            height=previous_state.height,
            width=previous_state.width,
            row=row,
            col=col,
            value=previous_state.value,
            blocked=previous_state.blocked,
            exit_cell=previous_state.exit_cell,
            exit_fill=previous_state.exit_fill,
            terminated=False,
        )

    if prediction.shape[0] != 2:
        raise ValueError("prediction must have either 1 or 2 channels")

    if previous_state.terminated:
        # Keep terminal dynamics exact by delegating to the deterministic engine.
        return step_grid(previous_state, Action.NONE)

    decoded = hard_decode_exit_prediction(prediction, active_threshold=active_threshold)
    exit_fill_cells = {
        (int(row), int(col))
        for row, col in torch.nonzero(decoded[1], as_tuple=False).tolist()
    }
    if previous_state.exit_cell is not None:
        exit_fill_cells.add(previous_state.exit_cell)

    if float(decoded[0].sum().item()) == 0.0:
        row, col = previous_state.exit_cell or (previous_state.row, previous_state.col)
        terminated = True
    else:
        flat_index = int(torch.argmax(decoded[0]).item())
        row = flat_index // previous_state.width
        col = flat_index % previous_state.width
        terminated = False
        if previous_state.exit_cell is not None:
            exit_fill_cells = {previous_state.exit_cell}

    return GridState(
        height=previous_state.height,
        width=previous_state.width,
        row=row,
        col=col,
        value=previous_state.value,
        blocked=previous_state.blocked,
        exit_cell=previous_state.exit_cell,
        exit_fill=frozenset(exit_fill_cells),
        terminated=terminated,
    )

from __future__ import annotations

from pathlib import Path

import torch

from .actions import Action
from .dataset import encode_control_input
from .device import resolve_device
from .grid import GridState
from .model import ControllableNCAModel


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[ControllableNCAModel, dict[str, object], torch.device]:
    resolved_device = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = dict(payload["config"])
    model = ControllableNCAModel(
        hidden_channels=int(config["hidden_channels"]),
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
    model, _config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    model_input = encode_control_input(state, action, device=resolved_device).unsqueeze(0)
    with torch.no_grad():
        prediction = model(model_input)
    prediction = prediction.squeeze(0).cpu()
    if hard_decode:
        return hard_decode_grid(prediction)
    return prediction


def hard_decode_grid(grid: torch.Tensor) -> torch.Tensor:
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

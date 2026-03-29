from __future__ import annotations

from pathlib import Path

import torch

from .dataset import build_transition_dataset
from .inference import load_checkpoint


def decode_argmax_positions(grids: torch.Tensor) -> torch.Tensor:
    if grids.ndim != 4 or grids.shape[1] != 1:
        raise ValueError("grids must have shape [batch, 1, height, width]")
    batch, _channels, _height, width = grids.shape
    flat_indices = torch.argmax(grids[:, 0].reshape(batch, -1), dim=1)
    rows = torch.div(flat_indices, width, rounding_mode="floor")
    cols = flat_indices % width
    return torch.stack([rows, cols], dim=1)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
) -> dict[str, float | int | str]:
    model, config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    dataset = build_transition_dataset(
        height=int(config["height"]),
        width=int(config["width"]),
        value=float(config["value"]),
        device="cpu",
    )
    with torch.no_grad():
        predictions = model(dataset.inputs.to(resolved_device)).cpu()

    target_positions = decode_argmax_positions(dataset.targets)
    predicted_positions = decode_argmax_positions(predictions)
    argmax_accuracy = (
        (predicted_positions == target_positions).all(dim=1).to(torch.float32).mean().item()
    )
    mse = torch.nn.functional.mse_loss(predictions, dataset.targets).item()
    predicted_max = predictions.amax(dim=(1, 2, 3)).mean().item()

    return {
        "device": str(resolved_device),
        "num_samples": int(dataset.inputs.shape[0]),
        "argmax_accuracy": argmax_accuracy,
        "mse": mse,
        "mean_predicted_max": predicted_max,
    }

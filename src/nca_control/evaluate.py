from __future__ import annotations

from pathlib import Path

import torch

from .dataset import ACTION_ORDER
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


def evaluate_rollout_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_sequences: int,
    steps_per_sequence: int,
    device: str = "auto",
    seed: int = 0,
    max_reported_failures: int = 5,
) -> dict[str, object]:
    model, config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    height = int(config["height"])
    width = int(config["width"])
    value = float(config["value"])
    up_index = ACTION_ORDER.index(ACTION_ORDER[1])
    down_index = ACTION_ORDER.index(ACTION_ORDER[2])
    left_index = ACTION_ORDER.index(ACTION_ORDER[3])
    right_index = ACTION_ORDER.index(ACTION_ORDER[4])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    ref_rows = torch.randint(0, height, (num_sequences,), generator=generator, device="cpu").to(resolved_device)
    ref_cols = torch.randint(0, width, (num_sequences,), generator=generator, device="cpu").to(resolved_device)
    model_rows = ref_rows.clone()
    model_cols = ref_cols.clone()

    failed = torch.zeros(num_sequences, dtype=torch.bool, device=resolved_device)
    first_failure_step = torch.full((num_sequences,), -1, dtype=torch.long, device=resolved_device)
    first_failure_action = torch.full((num_sequences,), -1, dtype=torch.long, device=resolved_device)
    first_failure_ref = torch.full((num_sequences, 2), -1, dtype=torch.long, device=resolved_device)
    first_failure_model = torch.full((num_sequences, 2), -1, dtype=torch.long, device=resolved_device)

    batch_index = torch.arange(num_sequences, device=resolved_device)

    for step_index in range(steps_per_sequence):
        action_indices = torch.randint(
            0,
            len(ACTION_ORDER),
            (num_sequences,),
            generator=generator,
            device="cpu",
        )
        action_indices = action_indices.to(resolved_device)
        ref_rows = torch.where(
            action_indices == up_index,
            (ref_rows - 1) % height,
            torch.where(action_indices == down_index, (ref_rows + 1) % height, ref_rows),
        )
        ref_cols = torch.where(
            action_indices == left_index,
            (ref_cols - 1) % width,
            torch.where(action_indices == right_index, (ref_cols + 1) % width, ref_cols),
        )

        inputs = torch.zeros((num_sequences, 1 + len(ACTION_ORDER), height, width), device=resolved_device)
        inputs[batch_index, 0, model_rows, model_cols] = value
        inputs[batch_index, 1 + action_indices, :, :] = 1.0

        with torch.no_grad():
            predictions = model(inputs)

        flat_indices = torch.argmax(predictions[:, 0].reshape(num_sequences, -1), dim=1)
        model_rows = torch.div(flat_indices, width, rounding_mode="floor")
        model_cols = flat_indices % width

        mismatch = ((ref_rows != model_rows) | (ref_cols != model_cols)) & (~failed)
        first_failure_step[mismatch] = step_index + 1
        first_failure_action[mismatch] = action_indices[mismatch]
        first_failure_ref[mismatch, 0] = ref_rows[mismatch]
        first_failure_ref[mismatch, 1] = ref_cols[mismatch]
        first_failure_model[mismatch, 0] = model_rows[mismatch]
        first_failure_model[mismatch, 1] = model_cols[mismatch]
        failed = failed | mismatch

    num_failed = int(failed.sum().item())
    rollout_accuracy = 1.0 - (num_failed / num_sequences)
    failed_indices = torch.nonzero(failed, as_tuple=False).flatten().tolist()[:max_reported_failures]
    first_failures: list[dict[str, object]] = []
    for index in failed_indices:
        first_failures.append(
            {
                "sequence": int(index),
                "step": int(first_failure_step[index].item()),
                "action": ACTION_ORDER[int(first_failure_action[index].item())].value,
                "reference": [
                    int(first_failure_ref[index, 0].item()),
                    int(first_failure_ref[index, 1].item()),
                ],
                "model": [
                    int(first_failure_model[index, 0].item()),
                    int(first_failure_model[index, 1].item()),
                ],
            }
        )

    return {
        "device": str(resolved_device),
        "num_sequences": num_sequences,
        "steps_per_sequence": steps_per_sequence,
        "total_rollout_steps": num_sequences * steps_per_sequence,
        "num_failed_sequences": num_failed,
        "exact_rollout_rate": rollout_accuracy,
        "first_failures": first_failures,
    }

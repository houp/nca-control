from __future__ import annotations

from pathlib import Path

import torch

from .dataset import ACTION_ORDER
from .dataset import MazeTransitionDataset, TransitionDataset, build_maze_transition_dataset, build_transition_dataset
from .inference import load_checkpoint
from .maze import generate_maze


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
    dataset = _build_evaluation_dataset(config)
    predictions, targets = _predict_dataset(model, dataset, resolved_device)
    target_positions = decode_argmax_positions(targets)
    predicted_positions = decode_argmax_positions(predictions)
    argmax_accuracy = (
        (predicted_positions == target_positions).all(dim=1).to(torch.float32).mean().item()
    )
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    predicted_max = predictions.amax(dim=(1, 2, 3)).mean().item()

    return {
        "device": str(resolved_device),
        "num_samples": int(targets.shape[0]),
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
    task = str(config.get("task", "plain"))
    up_index = ACTION_ORDER.index(ACTION_ORDER[1])
    down_index = ACTION_ORDER.index(ACTION_ORDER[2])
    left_index = ACTION_ORDER.index(ACTION_ORDER[3])
    right_index = ACTION_ORDER.index(ACTION_ORDER[4])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    blocked_tensor = torch.zeros((num_sequences, height, width), dtype=torch.bool, device=resolved_device)
    if task == "maze":
        maze_seed = int(config.get("maze_seed", 0))
        start_rows: list[int] = []
        start_cols: list[int] = []
        for index in range(num_sequences):
            layout = generate_maze(height=height, width=width, seed=maze_seed + seed + index)
            open_cells = layout.open_cells()
            sampled_index = int(torch.randint(0, len(open_cells), (1,), generator=generator, device="cpu").item())
            row, col = open_cells[sampled_index]
            start_rows.append(row)
            start_cols.append(col)
            for blocked_row, blocked_col in layout.blocked:
                blocked_tensor[index, blocked_row, blocked_col] = True
        ref_rows = torch.tensor(start_rows, dtype=torch.long, device=resolved_device)
        ref_cols = torch.tensor(start_cols, dtype=torch.long, device=resolved_device)
    else:
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
        proposed_ref_rows = torch.where(
            action_indices == up_index,
            (ref_rows - 1) % height,
            torch.where(action_indices == down_index, (ref_rows + 1) % height, ref_rows),
        )
        proposed_ref_cols = torch.where(
            action_indices == left_index,
            (ref_cols - 1) % width,
            torch.where(action_indices == right_index, (ref_cols + 1) % width, ref_cols),
        )
        blocked_reference = blocked_tensor[batch_index, proposed_ref_rows, proposed_ref_cols]
        ref_rows = torch.where(blocked_reference, ref_rows, proposed_ref_rows)
        ref_cols = torch.where(blocked_reference, ref_cols, proposed_ref_cols)

        inputs = torch.zeros((num_sequences, 2 + len(ACTION_ORDER), height, width), device=resolved_device)
        inputs[batch_index, 0, model_rows, model_cols] = value
        inputs[:, 1, :, :] = blocked_tensor.to(dtype=torch.float32)
        inputs[batch_index, 2 + action_indices, :, :] = 1.0

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


def _build_evaluation_dataset(config: dict[str, object]) -> TransitionDataset | MazeTransitionDataset:
    if config.get("task", "plain") == "maze":
        return build_maze_transition_dataset(
            height=int(config["height"]),
            width=int(config["width"]),
            num_mazes=int(config.get("eval_num_mazes", config.get("num_mazes", 8))),
            seed=int(config.get("maze_seed", 0)) + int(config.get("eval_seed_offset", 10_000)),
            value=float(config["value"]),
        )
    return build_transition_dataset(
        height=int(config["height"]),
        width=int(config["width"]),
        value=float(config["value"]),
        device="cpu",
    )


def _predict_dataset(
    model: torch.nn.Module,
    dataset: TransitionDataset | MazeTransitionDataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, TransitionDataset):
        with torch.no_grad():
            predictions = model(dataset.inputs.to(device)).cpu()
        return predictions, dataset.targets.cpu()

    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    index_range = torch.arange(len(dataset), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_indices = index_range[start : start + batch_size]
            batch_inputs, batch_targets = dataset.materialize_batch(batch_indices, device=device)
            prediction_batches.append(model(batch_inputs).cpu())
            target_batches.append(batch_targets.cpu())
    return torch.cat(prediction_batches, dim=0), torch.cat(target_batches, dim=0)

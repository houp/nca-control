from __future__ import annotations

"""Evaluation utilities for one-step accuracy and long-horizon rollout exactness."""

from pathlib import Path

import torch

from .actions import Action
from .dataset import ACTION_ORDER
from .dataset import (
    MazeExitTransitionDataset,
    MazeTransitionDataset,
    TransitionDataset,
    build_maze_exit_transition_dataset,
    build_maze_transition_dataset,
    build_transition_dataset,
    encode_control_input,
    propose_action_positions_torch,
)
from .grid import GridState, step_grid
from .inference import decode_prediction_state, load_checkpoint
from .maze import generate_maze


def decode_argmax_positions(grids: torch.Tensor) -> torch.Tensor:
    """Convert single-channel heatmaps into integer grid positions."""

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
    """Run the standard one-step evaluation for a saved PyTorch checkpoint."""

    model, config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    dataset = _build_evaluation_dataset(config)
    predictions, targets = _predict_dataset(model, dataset, resolved_device)
    if predictions.shape[1] == 2:
        if not isinstance(dataset, MazeExitTransitionDataset):
            raise TypeError("maze_exit evaluation requires MazeExitTransitionDataset")
        return _evaluate_exit_predictions(predictions, targets, dataset, resolved_device)
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
    height: int | None = None,
    width: int | None = None,
) -> dict[str, object]:
    """Compare learned rollouts against the deterministic reference dynamics."""

    model, config, resolved_device = load_checkpoint(checkpoint_path, device=device)
    if str(config.get("task", "plain")) == "maze_exit":
        return _evaluate_exit_rollout(
            model,
            config,
            resolved_device,
            num_sequences=num_sequences,
            steps_per_sequence=steps_per_sequence,
            seed=seed,
            max_reported_failures=max_reported_failures,
            height_override=height,
            width_override=width,
        )
    rollout_height = int(config["height"]) if height is None else int(height)
    rollout_width = int(config["width"]) if width is None else int(width)
    value = float(config["value"])
    task = str(config.get("task", "plain"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    blocked_tensor = torch.zeros((num_sequences, rollout_height, rollout_width), dtype=torch.bool, device=resolved_device)
    if task == "maze":
        maze_seed = int(config.get("maze_seed", 0))
        start_rows: list[int] = []
        start_cols: list[int] = []
        for index in range(num_sequences):
            layout = generate_maze(height=rollout_height, width=rollout_width, seed=maze_seed + seed + index)
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
        ref_rows = torch.randint(0, rollout_height, (num_sequences,), generator=generator, device="cpu").to(resolved_device)
        ref_cols = torch.randint(0, rollout_width, (num_sequences,), generator=generator, device="cpu").to(resolved_device)
    model_rows = ref_rows.clone()
    model_cols = ref_cols.clone()

    failed = torch.zeros(num_sequences, dtype=torch.bool, device=resolved_device)
    first_failure_step = torch.full((num_sequences,), -1, dtype=torch.long, device=resolved_device)
    first_failure_action = torch.full((num_sequences,), -1, dtype=torch.long, device=resolved_device)
    first_failure_ref = torch.full((num_sequences, 2), -1, dtype=torch.long, device=resolved_device)
    first_failure_model = torch.full((num_sequences, 2), -1, dtype=torch.long, device=resolved_device)

    batch_index = torch.arange(num_sequences, device=resolved_device)
    blocked_channel = blocked_tensor.to(dtype=torch.float32)

    for step_index in range(steps_per_sequence):
        action_indices = torch.randint(
            0,
            len(ACTION_ORDER),
            (num_sequences,),
            generator=generator,
            device="cpu",
        )
        action_indices = action_indices.to(resolved_device)
        proposed_ref_rows, proposed_ref_cols = propose_action_positions_torch(
            ref_rows,
            ref_cols,
            action_indices,
            height=rollout_height,
            width=rollout_width,
        )
        blocked_reference = blocked_tensor[batch_index, proposed_ref_rows, proposed_ref_cols]
        ref_rows = torch.where(blocked_reference, ref_rows, proposed_ref_rows)
        ref_cols = torch.where(blocked_reference, ref_cols, proposed_ref_cols)

        inputs = torch.zeros((num_sequences, 2 + len(ACTION_ORDER), rollout_height, rollout_width), device=resolved_device)
        inputs[batch_index, 0, model_rows, model_cols] = value
        inputs[:, 1, :, :] = blocked_channel
        inputs[batch_index, 2 + action_indices, :, :] = 1.0

        with torch.no_grad():
            predictions = model(inputs)

        flat_indices = torch.argmax(predictions[:, 0].reshape(num_sequences, -1), dim=1)
        model_rows = torch.div(flat_indices, rollout_width, rounding_mode="floor")
        model_cols = flat_indices % rollout_width

        # Only keep the first mismatch per sequence so reports stay compact and actionable.
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
        "rollout_height": rollout_height,
        "rollout_width": rollout_width,
        "num_sequences": num_sequences,
        "steps_per_sequence": steps_per_sequence,
        "total_rollout_steps": num_sequences * steps_per_sequence,
        "num_failed_sequences": num_failed,
        "exact_rollout_rate": rollout_accuracy,
        "first_failures": first_failures,
    }


def _build_evaluation_dataset(config: dict[str, object]) -> TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset:
    if config.get("task", "plain") == "maze_exit":
        return build_maze_exit_transition_dataset(
            height=int(config["height"]),
            width=int(config["width"]),
            num_mazes=int(config.get("eval_num_mazes", config.get("num_mazes", 8))),
            seed=int(config.get("maze_seed", 0)) + int(config.get("eval_seed_offset", 10_000)),
            value=float(config["value"]),
        )
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
    dataset: TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset,
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


def _evaluate_exit_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dataset: MazeExitTransitionDataset,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Score exit-aware predictions after decoding them back into legal game states."""

    predicted_active_present = predictions[:, 0, :, :].amax(dim=(1, 2)) >= 0.5
    target_active_present = targets[:, 0, :, :].amax(dim=(1, 2)) >= 0.5
    active_presence_accuracy = (
        (predicted_active_present == target_active_present).to(torch.float32).mean().item()
    )

    active_position_accuracy = 1.0
    active_mask = target_active_present & predicted_active_present
    if active_mask.any():
        predicted_positions = decode_argmax_positions(predictions[active_mask, :1, :, :])
        target_positions = decode_argmax_positions(targets[active_mask, :1, :, :])
        active_position_accuracy = (
            (predicted_positions == target_positions).all(dim=1).to(torch.float32).mean().item()
        )

    predicted_exit = predictions[:, 1, :, :] >= 0.5
    target_exit = targets[:, 1, :, :] >= 0.5
    decoded_matches: list[bool] = []
    decoded_exit_matches: list[bool] = []
    decoded_termination_matches: list[bool] = []
    for index in range(len(dataset)):
        # Decode both tensors through the same state machine used by the browser
        # visualizer so accuracy reflects gameplay semantics, not raw logits alone.
        previous_state = dataset.state_for_index(index)
        predicted_state = decode_prediction_state(predictions[index], previous_state)
        target_state = decode_prediction_state(targets[index], previous_state)
        decoded_matches.append(predicted_state == target_state)
        decoded_exit_matches.append(predicted_state.exit_fill == target_state.exit_fill)
        decoded_termination_matches.append(predicted_state.terminated == target_state.terminated)

    exit_fill_exact_accuracy = sum(decoded_exit_matches) / len(decoded_exit_matches)
    full_state_accuracy = sum(decoded_matches) / len(decoded_matches)
    termination_accuracy = sum(decoded_termination_matches) / len(decoded_termination_matches)

    return {
        "device": str(device),
        "num_samples": int(targets.shape[0]),
        "active_presence_accuracy": active_presence_accuracy,
        "active_position_accuracy": active_position_accuracy,
        "termination_accuracy": termination_accuracy,
        "exit_fill_exact_accuracy": exit_fill_exact_accuracy,
        "full_state_accuracy": full_state_accuracy,
        "mse": torch.nn.functional.mse_loss(predictions, targets).item(),
    }


def _evaluate_exit_rollout(
    model: torch.nn.Module,
    config: dict[str, object],
    device: torch.device,
    *,
    num_sequences: int,
    steps_per_sequence: int,
    seed: int,
    max_reported_failures: int,
    height_override: int | None = None,
    width_override: int | None = None,
) -> dict[str, object]:
    """Roll out the exit-aware task by comparing decoded model states to the reference engine."""

    height = int(config["height"]) if height_override is None else int(height_override)
    width = int(config["width"]) if width_override is None else int(width_override)
    value = float(config["value"])
    maze_seed = int(config.get("maze_seed", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    reference_states: list[GridState] = []
    model_states: list[GridState] = []
    for index in range(num_sequences):
        layout = generate_maze(height=height, width=width, seed=maze_seed + seed + index)
        open_cells = [cell for cell in layout.open_cells() if cell != layout.exit_cell]
        sampled_index = int(torch.randint(0, len(open_cells), (1,), generator=generator, device="cpu").item())
        row, col = open_cells[sampled_index]
        initial_state = layout.to_grid_state(row=row, col=col, value=value)
        reference_states.append(initial_state)
        model_states.append(initial_state)

    failures: list[dict[str, object]] = []
    failed_sequences: set[int] = set()
    with torch.no_grad():
        for step_index in range(steps_per_sequence):
            action_indices = torch.randint(
                0,
                len(ACTION_ORDER),
                (num_sequences,),
                generator=generator,
                device="cpu",
            )
            actions = [ACTION_ORDER[int(index)] for index in action_indices.tolist()]
            reference_states = [step_grid(state, action) for state, action in zip(reference_states, actions, strict=True)]
            batch_inputs = torch.stack(
                [
                    encode_control_input(state, action, device=device, include_exit_dynamics=True)
                    for state, action in zip(model_states, actions, strict=True)
                ],
                dim=0,
            )
            predictions = model(batch_inputs).cpu()
            model_states = [
                decode_prediction_state(predictions[index], model_states[index])
                for index in range(num_sequences)
            ]

            for sequence_index, (reference_state, model_state, action) in enumerate(
                zip(reference_states, model_states, actions, strict=True)
            ):
                if sequence_index in failed_sequences:
                    continue
                if reference_state != model_state:
                    failed_sequences.add(sequence_index)
                    if len(failures) < max_reported_failures:
                        failures.append(
                            {
                                "sequence": sequence_index,
                                "step": step_index + 1,
                                "action": action.value,
                                "reference": _serialize_rollout_state(reference_state),
                                "model": _serialize_rollout_state(model_state),
                            }
                        )

    return {
        "device": str(device),
        "rollout_height": height,
        "rollout_width": width,
        "num_sequences": num_sequences,
        "steps_per_sequence": steps_per_sequence,
        "total_rollout_steps": num_sequences * steps_per_sequence,
        "num_failed_sequences": len(failed_sequences),
        "exact_rollout_rate": 1.0 - (len(failed_sequences) / num_sequences),
        "first_failures": failures,
    }


def _serialize_rollout_state(state: GridState) -> dict[str, object]:
    return {
        "row": state.row,
        "col": state.col,
        "terminated": state.terminated,
        "exit_fill_size": len(state.exit_fill or frozenset()),
    }

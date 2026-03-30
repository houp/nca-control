from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch

from .actions import Action
from .dataset import (
    ACTION_ORDER,
    DOWN_ACTION_INDEX,
    LEFT_ACTION_INDEX,
    MazeExitTransitionDataset,
    MazeTransitionDataset,
    RIGHT_ACTION_INDEX,
    TransitionDataset,
    UP_ACTION_INDEX,
    build_maze_exit_transition_dataset,
    build_maze_transition_dataset,
    build_transition_dataset,
    encode_control_input,
)
from .evaluate import decode_argmax_positions
from .grid import GridState, step_grid
from .inference import decode_prediction_state
from .maze import generate_maze
from .train import TrainConfig


class MLXControllableNCAModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 7,
        state_channels: int = 1,
        hidden_channels: int = 32,
        perception_kernel_size: int = 3,
        update_kernel_size: int = 1,
        cell_value_max: float = 1.0,
    ) -> None:
        super().__init__()
        if perception_kernel_size <= 0 or perception_kernel_size % 2 == 0:
            raise ValueError("perception_kernel_size must be a positive odd integer")
        if update_kernel_size <= 0 or update_kernel_size % 2 == 0:
            raise ValueError("update_kernel_size must be a positive odd integer")
        self.state_channels = state_channels
        self.cell_value_max = cell_value_max
        self.perception_kernel_size = perception_kernel_size
        self.update_kernel_size = update_kernel_size
        self.perception = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=perception_kernel_size,
            bias=True,
        )
        self.update_hidden = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=update_kernel_size,
            bias=True,
        )
        self.output = nn.Conv2d(hidden_channels, state_channels, kernel_size=1, bias=True)

    def forward_logits(self, inputs: mx.array) -> mx.array:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, height, width, channels]")
        if inputs.shape[-1] < self.state_channels:
            raise ValueError("inputs do not contain enough state channels")
        features = self.perception(_circular_pad_nhwc(inputs, self.perception_kernel_size))
        features = nn.relu(features)
        hidden = self.update_hidden(_circular_pad_nhwc(features, self.update_kernel_size))
        hidden = nn.relu(hidden)
        return self.output(hidden)

    def __call__(self, inputs: mx.array) -> mx.array:
        logits = self.forward_logits(inputs)
        if self.state_channels == 1:
            batch, height, width, channels = logits.shape
            flattened = mx.reshape(logits, (batch, height * width, channels))
            normalized = mx.softmax(flattened, axis=1)
            return mx.reshape(normalized, (batch, height, width, channels)) * self.cell_value_max
        return mx.sigmoid(logits) * self.cell_value_max


def train_one_step_mlx(
    config: TrainConfig,
    output_dir: str | Path,
    progress_printer: Callable[[str], None] | None = None,
) -> dict[str, object]:
    mx.random.seed(config.seed)
    np.random.seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.jsonl"
    latest_status_path = output_path / "latest_status.json"
    checkpoint_path = output_path / "checkpoint_mlx"
    weights_path = output_path / "weights.npz"
    config_path = output_path / "config.json"
    metrics_path = output_path / "metrics.json"
    progress_path.write_text("", encoding="utf-8")

    dataset, num_samples, input_channels = _build_training_dataset(config)
    adapter = _MLXDatasetAdapter(dataset)
    state_channels = _task_state_channels(config.task)
    model = MLXControllableNCAModel(
        input_channels=input_channels,
        state_channels=state_channels,
        hidden_channels=config.hidden_channels,
        perception_kernel_size=config.perception_kernel_size,
        update_kernel_size=config.update_kernel_size,
        cell_value_max=config.value,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    losses: list[float] = []
    epoch_times_sec: list[float] = []
    config_payload = {
        **asdict(config),
        "backend": "mlx",
        "input_channels": input_channels,
        "state_channels": state_channels,
    }
    _write_json_file(
        latest_status_path,
        {
            "status": "running",
            "device": "mlx",
            "epoch": 0,
            "epochs_total": config.epochs,
            "num_samples": num_samples,
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "progress_path": str(progress_path),
        },
    )
    training_start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    for epoch_index in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        sample_count = 0
        permutation = rng.permutation(num_samples)
        for batch_indices in _iter_numpy_batches(permutation, config.batch_size):
            batch_inputs, batch_targets = adapter.materialize_batch(batch_indices)
            loss, grads = loss_and_grad(model, batch_inputs, batch_targets, config.task)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            batch_size = int(batch_inputs.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

        epoch_loss = epoch_loss_sum / sample_count
        epoch_time_sec = time.perf_counter() - epoch_start
        elapsed_sec = time.perf_counter() - training_start
        losses.append(epoch_loss)
        epoch_times_sec.append(epoch_time_sec)
        progress_record = {
            "epoch": epoch_index,
            "epochs_total": config.epochs,
            "device": "mlx",
            "loss": epoch_loss,
            "epoch_time_sec": epoch_time_sec,
            "elapsed_sec": elapsed_sec,
            "epoch_samples_per_second": sample_count / epoch_time_sec,
            "running_samples_per_second": (num_samples * epoch_index) / elapsed_sec,
            "num_samples": num_samples,
        }
        _append_jsonl_record(progress_path, progress_record)
        _write_json_file(
            latest_status_path,
            {
                "status": "running",
                **progress_record,
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "progress_path": str(progress_path),
            },
        )
        if progress_printer is not None:
            progress_printer(
                " ".join(
                    [
                        f"epoch={epoch_index}/{config.epochs}",
                        f"loss={epoch_loss:.6f}",
                        f"epoch_time_sec={epoch_time_sec:.3f}",
                        f"samples_per_second={sample_count / epoch_time_sec:.2f}",
                    ]
                )
            )

    total_train_time_sec = time.perf_counter() - training_start
    model.save_weights(str(weights_path))
    _write_json_file(config_path, config_payload)
    checkpoint_path.write_text(json.dumps({"weights": str(weights_path), "config": str(config_path)}, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "device": "mlx",
        "backend": "mlx",
        "final_loss": losses[-1],
        "loss_history": losses,
        "epoch_times_sec": epoch_times_sec,
        "total_train_time_sec": total_train_time_sec,
        "samples_per_second": (num_samples * config.epochs) / total_train_time_sec,
        "num_samples": num_samples,
    }
    _write_json_file(metrics_path, metrics)
    _write_json_file(
        latest_status_path,
        {
            "status": "completed",
            "device": "mlx",
            "epoch": config.epochs,
            "epochs_total": config.epochs,
            "final_loss": losses[-1],
            "total_train_time_sec": total_train_time_sec,
            "samples_per_second": metrics["samples_per_second"],
            "num_samples": num_samples,
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
            "progress_path": str(progress_path),
        },
    )
    return {
        "checkpoint_path": checkpoint_path,
        "weights_path": weights_path,
        "config_path": config_path,
        "metrics_path": metrics_path,
        "progress_path": progress_path,
        "latest_status_path": latest_status_path,
        "metrics": metrics,
    }


def evaluate_mlx_checkpoint(
    checkpoint_path: str | Path,
    batch_size: int = 256,
) -> dict[str, object]:
    model, config = load_mlx_checkpoint(checkpoint_path)
    dataset = _build_evaluation_dataset(config)
    predictions, targets = _predict_dataset(model, dataset, batch_size=batch_size)
    if str(config.get("task", "plain")) == "maze_exit":
        if not isinstance(dataset, MazeExitTransitionDataset):
            raise TypeError("maze_exit evaluation requires MazeExitTransitionDataset")
        return _evaluate_exit_predictions(predictions, targets, dataset)

    argmax_accuracy = _argmax_accuracy(predictions, targets)
    return {
        "device": "mlx",
        "backend": "mlx",
        "num_samples": int(targets.shape[0]),
        "argmax_accuracy": argmax_accuracy,
        "mse": torch.nn.functional.mse_loss(predictions, targets).item(),
        "mean_predicted_max": float(predictions.amax(dim=(1, 2, 3)).mean().item()),
    }


def evaluate_rollout_mlx_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_sequences: int,
    steps_per_sequence: int,
    seed: int,
    max_reported_failures: int = 8,
    height_override: int | None = None,
    width_override: int | None = None,
) -> dict[str, object]:
    model, config = load_mlx_checkpoint(checkpoint_path)
    task = str(config.get("task", "plain"))
    if task != "maze_exit":
        raise ValueError("MLX rollout evaluation currently supports maze_exit checkpoints only")

    height = int(config["height"]) if height_override is None else int(height_override)
    width = int(config["width"]) if width_override is None else int(width_override)
    value = float(config["value"])
    maze_seed = int(config.get("maze_seed", 0))
    generator = np.random.default_rng(seed)

    reference_states: list[GridState] = []
    model_states: list[GridState] = []
    for index in range(num_sequences):
        layout = generate_maze(height=height, width=width, seed=maze_seed + seed + index)
        open_cells = [cell for cell in layout.open_cells() if cell != layout.exit_cell]
        sampled_index = int(generator.integers(0, len(open_cells)))
        row, col = open_cells[sampled_index]
        initial_state = layout.to_grid_state(row=row, col=col, value=value)
        reference_states.append(initial_state)
        model_states.append(initial_state)

    failures: list[dict[str, object]] = []
    failed_sequences: set[int] = set()
    for step_index in range(steps_per_sequence):
        action_indices = generator.integers(0, len(ACTION_ORDER), size=num_sequences)
        actions = [ACTION_ORDER[int(index)] for index in action_indices.tolist()]
        reference_states = [step_grid(state, action) for state, action in zip(reference_states, actions, strict=True)]
        batch_inputs = np.stack(
            [
                _torch_nchw_to_mlx_nhwc(
                    encode_control_input(state, action, device="cpu", include_exit_dynamics=True).unsqueeze(0).numpy()
                )[0]
                for state, action in zip(model_states, actions, strict=True)
            ],
            axis=0,
        )
        predictions = model(mx.array(batch_inputs))
        mx.eval(predictions)
        predictions_torch = torch.from_numpy(_mlx_nhwc_to_torch_nchw(np.array(predictions)))
        model_states = [
            decode_prediction_state(predictions_torch[index], model_states[index])
            for index in range(num_sequences)
        ]

        for sequence_index, (reference_state, model_state, action) in enumerate(
            zip(reference_states, model_states, actions, strict=True)
        ):
            if reference_state == model_state:
                continue
            failed_sequences.add(sequence_index)
            if len(failures) < max_reported_failures:
                failures.append(
                    {
                        "sequence_index": sequence_index,
                        "step_index": step_index,
                        "action": action.value,
                        "reference_state": _state_to_summary(reference_state),
                        "model_state": _state_to_summary(model_state),
                    }
                )

    total_steps = num_sequences * steps_per_sequence
    return {
        "device": "mlx",
        "backend": "mlx",
        "num_sequences": num_sequences,
        "steps_per_sequence": steps_per_sequence,
        "total_rollout_steps": total_steps,
        "failed_sequences": len(failed_sequences),
        "exact_rollout_rate": 1.0 - (len(failed_sequences) / num_sequences),
        "failures": failures,
    }


def load_mlx_checkpoint(checkpoint_path: str | Path) -> tuple[MLXControllableNCAModel, dict[str, object]]:
    checkpoint_file = Path(checkpoint_path)
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    config = json.loads(Path(payload["config"]).read_text(encoding="utf-8"))
    model = MLXControllableNCAModel(
        input_channels=int(config["input_channels"]),
        state_channels=int(config.get("state_channels", 1)),
        hidden_channels=int(config["hidden_channels"]),
        perception_kernel_size=int(config.get("perception_kernel_size", 3)),
        update_kernel_size=int(config.get("update_kernel_size", 1)),
        cell_value_max=float(config["value"]),
    )
    model.load_weights(str(payload["weights"]))
    mx.eval(model.parameters())
    return model, config


def predict_next_state_mlx(
    checkpoint_path: str | Path,
    state: GridState,
    action: Action,
    hard_decode: bool = True,
) -> torch.Tensor:
    model, config = load_mlx_checkpoint(checkpoint_path)
    include_exit_dynamics = int(config.get("state_channels", 1)) > 1
    model_input = encode_control_input(
        state,
        action,
        device="cpu",
        include_exit_dynamics=include_exit_dynamics,
    ).unsqueeze(0)
    predictions = model(mx.array(_torch_nchw_to_mlx_nhwc(model_input.numpy())))
    mx.eval(predictions)
    prediction = torch.from_numpy(_mlx_nhwc_to_torch_nchw(np.array(predictions)))[0]
    if hard_decode:
        if prediction.shape[0] == 2:
            from .inference import hard_decode_exit_prediction

            return hard_decode_exit_prediction(prediction)
        from .inference import hard_decode_grid

        return hard_decode_grid(prediction)
    return prediction


def convert_torch_checkpoint_to_mlx(
    torch_checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = torch.load(torch_checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    input_channels = int(payload["model_state_dict"]["perception.weight"].shape[1])
    model = MLXControllableNCAModel(
        input_channels=input_channels,
        state_channels=int(config.get("state_channels", 1)),
        hidden_channels=int(config["hidden_channels"]),
        perception_kernel_size=int(config.get("perception_kernel_size", 3)),
        update_kernel_size=int(config.get("update_kernel_size", 1)),
        cell_value_max=float(config["value"]),
    )
    apply_torch_state_dict_to_mlx_model(model, payload["model_state_dict"])
    weights_path = output_path / "weights.npz"
    config_path = output_path / "config.json"
    checkpoint_path = output_path / "checkpoint_mlx"
    model.save_weights(str(weights_path))
    _write_json_file(
        config_path,
        {
            **config,
            "backend": "mlx",
            "source_checkpoint": str(torch_checkpoint_path),
            "input_channels": input_channels,
        },
    )
    checkpoint_path.write_text(json.dumps({"weights": str(weights_path), "config": str(config_path)}, indent=2) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": checkpoint_path,
        "weights_path": weights_path,
        "config_path": config_path,
    }


def _propose_action_positions_numpy(
    rows: np.ndarray,
    cols: np.ndarray,
    action_indices: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    proposed_rows = np.where(
        action_indices == UP_ACTION_INDEX,
        (rows - 1) % height,
        np.where(action_indices == DOWN_ACTION_INDEX, (rows + 1) % height, rows),
    )
    proposed_cols = np.where(
        action_indices == LEFT_ACTION_INDEX,
        (cols - 1) % width,
        np.where(action_indices == RIGHT_ACTION_INDEX, (cols + 1) % width, cols),
    )
    return proposed_rows, proposed_cols


def apply_torch_state_dict_to_mlx_model(
    model: MLXControllableNCAModel,
    state_dict: dict[str, torch.Tensor],
) -> None:
    model.perception.weight = mx.array(_torch_conv_weight_to_mlx(state_dict["perception.weight"]))
    model.perception.bias = mx.array(state_dict["perception.bias"].detach().cpu().numpy())
    model.update_hidden.weight = mx.array(_torch_conv_weight_to_mlx(state_dict["update.1.weight"]))
    model.update_hidden.bias = mx.array(state_dict["update.1.bias"].detach().cpu().numpy())
    model.output.weight = mx.array(_torch_conv_weight_to_mlx(state_dict["update.3.weight"]))
    model.output.bias = mx.array(state_dict["update.3.bias"].detach().cpu().numpy())
    mx.eval(model.parameters())


def _loss_fn(
    model: MLXControllableNCAModel,
    inputs: mx.array,
    targets: mx.array,
    task: str,
) -> mx.array:
    logits = model.forward_logits(inputs)
    return _compute_loss(logits, targets, task)


def _compute_loss(logits: mx.array, targets: mx.array, task: str) -> mx.array:
    if task != "maze_exit":
        batch = logits.shape[0]
        logits_flat = mx.reshape(logits, (batch, -1))
        target_indices = mx.argmax(mx.reshape(targets, (batch, -1)), axis=1)
        selected = logits_flat[mx.arange(batch), target_indices]
        return mx.mean(mx.logsumexp(logits_flat, axis=1) - selected)

    active_logits = logits[:, :, :, 0]
    exit_logits = logits[:, :, :, 1]
    active_targets = targets[:, :, :, 0]
    exit_targets = targets[:, :, :, 1]
    spatial_size = float(active_logits.shape[1] * active_logits.shape[2])
    active_loss = _binary_cross_entropy_with_logits(active_logits, active_targets, pos_weight=spatial_size)
    exit_loss = _binary_cross_entropy_with_logits(exit_logits, exit_targets, pos_weight=4.0)
    active_counts = mx.sum(mx.sigmoid(active_logits), axis=(1, 2))
    target_counts = mx.sum(active_targets, axis=(1, 2))
    count_loss = mx.mean(mx.square(active_counts - target_counts))
    return active_loss + exit_loss + count_loss


def _binary_cross_entropy_with_logits(logits: mx.array, targets: mx.array, pos_weight: float) -> mx.array:
    log_weight = 1.0 + (pos_weight - 1.0) * targets
    base = mx.maximum(-logits, 0.0) + mx.log1p(mx.exp(-mx.abs(logits)))
    return mx.mean((1.0 - targets) * logits + log_weight * base)


def _build_training_dataset(config: TrainConfig) -> tuple[TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset, int, int]:
    if config.task == "maze_exit":
        dataset = build_maze_exit_transition_dataset(
            height=config.height,
            width=config.width,
            num_mazes=config.num_mazes,
            seed=config.maze_seed,
            value=config.value,
        )
        sample_input, _sample_target = dataset[0]
        return dataset, len(dataset), int(sample_input.shape[0])
    elif config.task == "maze":
        dataset = build_maze_transition_dataset(
            height=config.height,
            width=config.width,
            num_mazes=config.num_mazes,
            seed=config.maze_seed,
            value=config.value,
        )
        sample_input, _sample_target = dataset[0]
        return dataset, len(dataset), int(sample_input.shape[0])
    elif config.task == "plain":
        dataset = build_transition_dataset(
            height=config.height,
            width=config.width,
            value=config.value,
            device="cpu",
        )
        return dataset, int(dataset.inputs.shape[0]), int(dataset.inputs.shape[1])
    else:
        raise ValueError(f"unsupported task: {config.task}")


def _build_evaluation_dataset(config: dict[str, object]) -> TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset:
    task = str(config.get("task", "plain"))
    if task == "maze_exit":
        return build_maze_exit_transition_dataset(
            height=int(config["height"]),
            width=int(config["width"]),
            num_mazes=int(config.get("eval_num_mazes", 8)),
            seed=int(config.get("maze_seed", 0)) + int(config.get("eval_seed_offset", 10_000)),
            value=float(config["value"]),
        )
    if task == "maze":
        return build_maze_transition_dataset(
            height=int(config["height"]),
            width=int(config["width"]),
            num_mazes=int(config.get("eval_num_mazes", 8)),
            seed=int(config.get("maze_seed", 0)) + int(config.get("eval_seed_offset", 10_000)),
            value=float(config["value"]),
        )
    return build_transition_dataset(
        height=int(config["height"]),
        width=int(config["width"]),
        value=float(config["value"]),
        device="cpu",
    )


class _MLXDatasetAdapter:
    def __init__(self, dataset: TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset) -> None:
        self.dataset = dataset
        self._init_numpy_banks()

    def _init_numpy_banks(self) -> None:
        if isinstance(self.dataset, TransitionDataset):
            self.inputs = self.dataset.inputs.detach().cpu().numpy().astype(np.float32, copy=False)
            self.targets = self.dataset.targets.detach().cpu().numpy().astype(np.float32, copy=False)
            return

        if isinstance(self.dataset, MazeTransitionDataset):
            self.examples = self.dataset.examples.detach().cpu().numpy()
            self.blocked_grids = self.dataset.blocked_grids.detach().cpu().numpy().astype(np.float32, copy=False)
            self.action_grids = self.dataset.action_grids.detach().cpu().numpy().astype(np.float32, copy=False)
            self.height = self.dataset.height
            self.width = self.dataset.width
            self.value = float(self.dataset.value)
            return

        if isinstance(self.dataset, MazeExitTransitionDataset):
            self.examples = self.dataset.examples.detach().cpu().numpy()
            self.blocked_grids = self.dataset.blocked_grids.detach().cpu().numpy().astype(np.float32, copy=False)
            self.exit_fill_grids = self.dataset.exit_fill_grids.detach().cpu().numpy().astype(np.float32, copy=False)
            self.fill_stage_lengths = self.dataset.fill_stage_lengths.detach().cpu().numpy()
            self.exit_cells = self.dataset.exit_cells.detach().cpu().numpy()
            self.action_grids = self.dataset.action_grids.detach().cpu().numpy().astype(np.float32, copy=False)
            self.height = self.dataset.height
            self.width = self.dataset.width
            self.value = float(self.dataset.value)
            return

        raise TypeError(f"unsupported dataset type: {type(self.dataset)!r}")

    def materialize_batch(self, batch_indices: np.ndarray) -> tuple[mx.array, mx.array]:
        if isinstance(self.dataset, TransitionDataset):
            return (
                mx.array(_torch_nchw_to_mlx_nhwc(self.inputs[batch_indices])),
                mx.array(_torch_nchw_to_mlx_nhwc(self.targets[batch_indices])),
            )
        if isinstance(self.dataset, MazeTransitionDataset):
            return self._materialize_maze_batch(batch_indices)
        if isinstance(self.dataset, MazeExitTransitionDataset):
            return self._materialize_maze_exit_batch(batch_indices)
        raise TypeError(f"unsupported dataset type: {type(self.dataset)!r}")

    def _materialize_maze_batch(self, batch_indices: np.ndarray) -> tuple[mx.array, mx.array]:
        metadata = self.examples[batch_indices]
        layout_indices = metadata[:, 0]
        rows = metadata[:, 1]
        cols = metadata[:, 2]
        action_indices = metadata[:, 3]

        batch_size = int(metadata.shape[0])
        batch_axis = np.arange(batch_size)
        blocked = self.blocked_grids[layout_indices]
        action_channels = self.action_grids[action_indices]

        state_channel = np.zeros((batch_size, 1, self.height, self.width), dtype=np.float32)
        state_channel[batch_axis, 0, rows, cols] = self.value
        inputs = np.concatenate([state_channel, blocked[:, None, :, :], action_channels], axis=1)

        proposed_rows, proposed_cols = _propose_action_positions_numpy(
            rows,
            cols,
            action_indices,
            height=self.height,
            width=self.width,
        )
        blocked_targets = blocked[batch_axis, proposed_rows, proposed_cols] > 0.5
        target_rows = np.where(blocked_targets, rows, proposed_rows)
        target_cols = np.where(blocked_targets, cols, proposed_cols)

        targets = np.zeros((batch_size, 1, self.height, self.width), dtype=np.float32)
        targets[batch_axis, 0, target_rows, target_cols] = self.value
        return mx.array(_torch_nchw_to_mlx_nhwc(inputs)), mx.array(_torch_nchw_to_mlx_nhwc(targets))

    def _materialize_maze_exit_batch(self, batch_indices: np.ndarray) -> tuple[mx.array, mx.array]:
        metadata = self.examples[batch_indices]
        layout_indices = metadata[:, 0]
        rows = metadata[:, 1]
        cols = metadata[:, 2]
        action_indices = metadata[:, 3]
        terminated = metadata[:, 4].astype(bool)
        fill_stage_indices = metadata[:, 5]

        batch_size = int(metadata.shape[0])
        batch_axis = np.arange(batch_size)
        blocked = self.blocked_grids[layout_indices]
        exit_fill = self.exit_fill_grids[layout_indices, fill_stage_indices]
        action_channels = self.action_grids[action_indices]
        exit_cells = self.exit_cells[layout_indices]
        exit_rows = exit_cells[:, 0]
        exit_cols = exit_cells[:, 1]

        active_channel = np.zeros((batch_size, 1, self.height, self.width), dtype=np.float32)
        active_indices = batch_axis[~terminated]
        if active_indices.size > 0:
            active_channel[active_indices, 0, rows[~terminated], cols[~terminated]] = self.value
        inputs = np.concatenate([active_channel, exit_fill[:, None, :, :], blocked[:, None, :, :], action_channels], axis=1)

        targets = np.zeros((batch_size, 2, self.height, self.width), dtype=np.float32)
        targets[:, 1, :, :] = exit_fill

        active_examples = ~terminated
        if np.any(active_examples):
            active_rows = rows[active_examples]
            active_cols = cols[active_examples]
            active_actions = action_indices[active_examples]
            active_layout_rows = exit_rows[active_examples]
            active_layout_cols = exit_cols[active_examples]
            active_batch_indices = batch_axis[active_examples]

            proposed_rows, proposed_cols = _propose_action_positions_numpy(
                active_rows,
                active_cols,
                active_actions,
                height=self.height,
                width=self.width,
            )
            blocked_targets = blocked[active_batch_indices, proposed_rows, proposed_cols] > 0.5
            target_rows = np.where(blocked_targets, active_rows, proposed_rows)
            target_cols = np.where(blocked_targets, active_cols, proposed_cols)
            reached_exit = (target_rows == active_layout_rows) & (target_cols == active_layout_cols)
            surviving = ~reached_exit
            if np.any(surviving):
                targets[active_batch_indices[surviving], 0, target_rows[surviving], target_cols[surviving]] = self.value

        if np.any(terminated):
            terminated_indices = batch_axis[terminated]
            max_stage_indices = self.fill_stage_lengths[layout_indices[terminated]] - 1
            next_stage_indices = np.minimum(fill_stage_indices[terminated] + 1, max_stage_indices)
            next_exit_fill = self.exit_fill_grids[layout_indices[terminated], next_stage_indices]
            targets[terminated_indices, 1, :, :] = next_exit_fill

        return mx.array(_torch_nchw_to_mlx_nhwc(inputs)), mx.array(_torch_nchw_to_mlx_nhwc(targets))


def _predict_dataset(
    model: MLXControllableNCAModel,
    dataset: TransitionDataset | MazeTransitionDataset | MazeExitTransitionDataset,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapter = _MLXDatasetAdapter(dataset)
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for batch_indices in _iter_numpy_batches(np.arange(len(dataset), dtype=np.int64), batch_size):
        batch_inputs, batch_targets = adapter.materialize_batch(batch_indices)
        batch_predictions = model(batch_inputs)
        mx.eval(batch_predictions, batch_targets)
        prediction_batches.append(torch.from_numpy(_mlx_nhwc_to_torch_nchw(np.array(batch_predictions))))
        target_batches.append(torch.from_numpy(_mlx_nhwc_to_torch_nchw(np.array(batch_targets))))
    return torch.cat(prediction_batches, dim=0), torch.cat(target_batches, dim=0)


def _evaluate_exit_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dataset: MazeExitTransitionDataset,
) -> dict[str, object]:
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

    decoded_matches: list[bool] = []
    decoded_exit_matches: list[bool] = []
    decoded_termination_matches: list[bool] = []
    for index in range(len(dataset)):
        previous_state = dataset.state_for_index(index)
        predicted_state = decode_prediction_state(predictions[index], previous_state)
        target_state = decode_prediction_state(targets[index], previous_state)
        decoded_matches.append(predicted_state == target_state)
        decoded_exit_matches.append(predicted_state.exit_fill == target_state.exit_fill)
        decoded_termination_matches.append(predicted_state.terminated == target_state.terminated)

    return {
        "device": "mlx",
        "backend": "mlx",
        "num_samples": int(targets.shape[0]),
        "active_presence_accuracy": active_presence_accuracy,
        "active_position_accuracy": active_position_accuracy,
        "termination_accuracy": sum(decoded_termination_matches) / len(decoded_termination_matches),
        "exit_fill_exact_accuracy": sum(decoded_exit_matches) / len(decoded_exit_matches),
        "full_state_accuracy": sum(decoded_matches) / len(decoded_matches),
        "mse": torch.nn.functional.mse_loss(predictions, targets).item(),
    }


def _argmax_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predicted_positions = decode_argmax_positions(predictions)
    target_positions = decode_argmax_positions(targets)
    return (
        (predicted_positions == target_positions).all(dim=1).to(torch.float32).mean().item()
    )


def _circular_pad_nhwc(inputs: mx.array, kernel_size: int) -> mx.array:
    pad = kernel_size // 2
    if pad == 0:
        return inputs
    padded = mx.concatenate([inputs[:, -pad:, :, :], inputs, inputs[:, :pad, :, :]], axis=1)
    return mx.concatenate([padded[:, :, -pad:, :], padded, padded[:, :, :pad, :]], axis=2)


def _torch_conv_weight_to_mlx(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32, copy=False)


def _torch_nchw_to_mlx_nhwc(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (0, 2, 3, 1))


def _mlx_nhwc_to_torch_nchw(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (0, 3, 1, 2))


def _iter_numpy_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, int(indices.shape[0]), batch_size):
        yield indices[start : start + batch_size]


def _task_state_channels(task: str) -> int:
    if task == "maze_exit":
        return 2
    return 1


def _append_jsonl_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _state_to_summary(state: GridState) -> dict[str, object]:
    return {
        "row": state.row,
        "col": state.col,
        "terminated": state.terminated,
        "exit_fill_size": len(state.exit_fill or frozenset()),
    }

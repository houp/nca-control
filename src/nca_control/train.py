from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset

from .dataset import (
    MazeExitTransitionDataset,
    MazeTransitionDataset,
    TransitionDataset,
    build_maze_exit_transition_dataset,
    build_maze_transition_dataset,
    build_transition_dataset,
)
from .device import resolve_device
from .model import ControllableNCAModel


@dataclass(frozen=True, slots=True)
class TrainConfig:
    task: str = "plain"
    height: int = 6
    width: int = 6
    value: float = 1.0
    num_mazes: int = 32
    maze_seed: int = 0
    eval_num_mazes: int = 8
    eval_seed_offset: int = 10_000
    hidden_channels: int = 32
    perception_kernel_size: int = 3
    update_kernel_size: int = 1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 0


def train_one_step(
    config: TrainConfig,
    output_dir: str | Path,
    progress_printer: Callable[[str], None] | None = None,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    progress_path = output_path / "progress.jsonl"
    latest_status_path = output_path / "latest_status.json"
    progress_path.write_text("", encoding="utf-8")

    device = resolve_device(config.device)
    dataset, num_samples, input_channels = _build_training_dataset(config)
    state_channels = _task_state_channels(config.task)
    model = ControllableNCAModel(
        input_channels=input_channels,
        state_channels=state_channels,
        hidden_channels=config.hidden_channels,
        perception_kernel_size=config.perception_kernel_size,
        update_kernel_size=config.update_kernel_size,
        cell_value_max=config.value,
    ).to(device)
    if device.type in {"mps", "cuda"}:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    epoch_times_sec: list[float] = []
    training_start = time.perf_counter()
    _write_json_file(
        latest_status_path,
        {
            "status": "running",
            "device": str(device),
            "epoch": 0,
            "epochs_total": config.epochs,
            "num_samples": num_samples,
            "checkpoint_path": str(output_path / "checkpoint.pt"),
            "metrics_path": str(output_path / "metrics.json"),
            "progress_path": str(progress_path),
        },
    )
    for epoch_index in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in _iterate_training_batches(dataset, config.batch_size, device, config.seed + len(losses)):
            if device.type in {"mps", "cuda"}:
                batch_inputs = batch_inputs.contiguous(memory_format=torch.channels_last)
                batch_targets = batch_targets.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(batch_inputs)
            loss = _compute_loss(logits, batch_targets, config.task)
            loss.backward()
            optimizer.step()

            batch_size = batch_inputs.shape[0]
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        epoch_loss_mean = epoch_loss / sample_count
        epoch_time_sec = time.perf_counter() - epoch_start
        elapsed_sec = time.perf_counter() - training_start
        losses.append(epoch_loss_mean)
        epoch_times_sec.append(epoch_time_sec)
        progress_record = {
            "epoch": epoch_index,
            "epochs_total": config.epochs,
            "device": str(device),
            "loss": epoch_loss_mean,
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
                "checkpoint_path": str(output_path / "checkpoint.pt"),
                "metrics_path": str(output_path / "metrics.json"),
                "progress_path": str(progress_path),
            },
        )
        if progress_printer is not None:
            progress_printer(
                " ".join(
                    [
                        f"epoch={epoch_index}/{config.epochs}",
                        f"loss={epoch_loss_mean:.6f}",
                        f"epoch_time_sec={epoch_time_sec:.3f}",
                        f"samples_per_second={sample_count / epoch_time_sec:.2f}",
                    ]
                )
            )

    total_train_time_sec = time.perf_counter() - training_start

    checkpoint_path = output_path / "checkpoint.pt"
    metrics_path = output_path / "metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                **asdict(config),
                "input_channels": input_channels,
                "state_channels": state_channels,
            },
            "final_loss": losses[-1],
            "loss_history": losses,
        },
        checkpoint_path,
    )
    metrics = {
        "device": str(device),
        "final_loss": losses[-1],
        "loss_history": losses,
        "epoch_times_sec": epoch_times_sec,
        "total_train_time_sec": total_train_time_sec,
        "samples_per_second": (num_samples * config.epochs) / total_train_time_sec,
        "num_samples": num_samples,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_json_file(
        latest_status_path,
        {
            "status": "completed",
            "device": str(device),
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
        "metrics_path": metrics_path,
        "progress_path": progress_path,
        "latest_status_path": latest_status_path,
        "metrics": metrics,
    }


def _build_training_dataset(config: TrainConfig) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], int, int]:
    if config.task == "maze_exit":
        maze_dataset = build_maze_exit_transition_dataset(
            height=config.height,
            width=config.width,
            num_mazes=config.num_mazes,
            seed=config.maze_seed,
            value=config.value,
        )
        sample_input, _sample_target = maze_dataset[0]
        return maze_dataset, len(maze_dataset), int(sample_input.shape[0])
    if config.task == "maze":
        maze_dataset = build_maze_transition_dataset(
            height=config.height,
            width=config.width,
            num_mazes=config.num_mazes,
            seed=config.maze_seed,
            value=config.value,
        )
        sample_input, _sample_target = maze_dataset[0]
        return maze_dataset, len(maze_dataset), int(sample_input.shape[0])
    if config.task != "plain":
        raise ValueError(f"unsupported task: {config.task}")
    tensor_data = build_transition_dataset(
        height=config.height,
        width=config.width,
        value=config.value,
        device="cpu",
    )
    dataset = TensorDataset(tensor_data.inputs, tensor_data.targets)
    return dataset, int(tensor_data.inputs.shape[0]), int(tensor_data.inputs.shape[1])


def _iterate_training_batches(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(dataset, MazeTransitionDataset | MazeExitTransitionDataset):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(len(dataset), generator=generator)
        for start in range(0, len(dataset), batch_size):
            batch_indices = permutation[start : start + batch_size]
            yield dataset.materialize_batch(batch_indices, device=device)
        return

    if isinstance(dataset, TensorDataset):
        inputs, targets = dataset.tensors
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        for start in range(0, inputs.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            yield inputs[batch_indices].to(device), targets[batch_indices].to(device)
        return

    raise TypeError(f"unsupported dataset type: {type(dataset)!r}")


def _task_state_channels(task: str) -> int:
    if task == "maze_exit":
        return 2
    return 1


def _append_jsonl_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    if task != "maze_exit":
        target_indices = torch.argmax(targets.view(targets.shape[0], -1), dim=1)
        return torch.nn.functional.cross_entropy(logits.view(logits.shape[0], -1), target_indices)

    active_logits = logits[:, 0, :, :]
    exit_logits = logits[:, 1, :, :]
    active_targets = targets[:, 0, :, :]
    exit_targets = targets[:, 1, :, :]
    spatial_size = active_logits.shape[-2] * active_logits.shape[-1]
    active_pos_weight = torch.tensor(float(spatial_size), device=logits.device)
    exit_pos_weight = torch.tensor(4.0, device=logits.device)

    active_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        active_logits,
        active_targets,
        pos_weight=active_pos_weight,
    )
    exit_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        exit_logits,
        exit_targets,
        pos_weight=exit_pos_weight,
    )
    active_counts = torch.sigmoid(active_logits).sum(dim=(1, 2))
    target_counts = active_targets.sum(dim=(1, 2))
    count_loss = torch.nn.functional.mse_loss(active_counts, target_counts)
    return active_loss + exit_loss + count_loss

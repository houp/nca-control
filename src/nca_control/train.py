from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset

from .dataset import MazeTransitionDataset, TransitionDataset, build_maze_transition_dataset, build_transition_dataset
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
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    device: str = "auto"
    seed: int = 0


def train_one_step(config: TrainConfig, output_dir: str | Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.device)
    dataset, num_samples, input_channels = _build_training_dataset(config)
    model = ControllableNCAModel(
        input_channels=input_channels,
        hidden_channels=config.hidden_channels,
        cell_value_max=config.value,
    ).to(device)
    if device.type in {"mps", "cuda"}:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    epoch_times_sec: list[float] = []
    training_start = time.perf_counter()
    for _ in range(config.epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in _iterate_training_batches(dataset, config.batch_size, device, config.seed + len(losses)):
            if device.type in {"mps", "cuda"}:
                batch_inputs = batch_inputs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(batch_inputs)
            target_indices = torch.argmax(batch_targets.view(batch_targets.shape[0], -1), dim=1)
            loss = torch.nn.functional.cross_entropy(logits.view(logits.shape[0], -1), target_indices)
            loss.backward()
            optimizer.step()

            batch_size = batch_inputs.shape[0]
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        losses.append(epoch_loss / sample_count)
        epoch_times_sec.append(time.perf_counter() - epoch_start)

    total_train_time_sec = time.perf_counter() - training_start

    checkpoint_path = output_path / "checkpoint.pt"
    metrics_path = output_path / "metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {**asdict(config), "input_channels": input_channels},
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
    return {
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def _build_training_dataset(config: TrainConfig) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], int, int]:
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
    if isinstance(dataset, MazeTransitionDataset):
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

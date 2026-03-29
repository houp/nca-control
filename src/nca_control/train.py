from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import TransitionDataset, build_transition_dataset
from .device import resolve_device
from .model import ControllableNCAModel


@dataclass(frozen=True, slots=True)
class TrainConfig:
    height: int = 6
    width: int = 6
    value: float = 1.0
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
    dataset = build_transition_dataset(
        height=config.height,
        width=config.width,
        value=config.value,
        device="cpu",
    )
    model = ControllableNCAModel(
        hidden_channels=config.hidden_channels,
        cell_value_max=config.value,
    ).to(device)
    dataloader = _make_dataloader(dataset, config.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    losses: list[float] = []
    for _ in range(config.epochs):
        epoch_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = batch_inputs.shape[0]
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size

        losses.append(epoch_loss / sample_count)

    checkpoint_path = output_path / "checkpoint.pt"
    metrics_path = output_path / "metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "final_loss": losses[-1],
            "loss_history": losses,
        },
        checkpoint_path,
    )
    metrics = {
        "device": str(device),
        "final_loss": losses[-1],
        "loss_history": losses,
        "num_samples": int(dataset.inputs.shape[0]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def _make_dataloader(dataset: TransitionDataset, batch_size: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    tensor_dataset = TensorDataset(dataset.inputs, dataset.targets)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

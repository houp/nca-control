from __future__ import annotations

import json

import torch

from nca_control.actions import Action
from nca_control.device import resolve_device
from nca_control.grid import GridState
from nca_control.inference import load_checkpoint, predict_next_state
from nca_control.train import TrainConfig, train_one_step


def test_resolve_device_returns_torch_device() -> None:
    device = resolve_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "mps"}


def test_train_one_step_creates_checkpoint_and_metrics(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(height=3, width=3, epochs=2, batch_size=8, hidden_channels=8, device="cpu"),
        output_dir=tmp_path,
    )

    checkpoint_path = result["checkpoint_path"]
    metrics_path = result["metrics_path"]
    metrics = result["metrics"]

    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert metrics["device"] == "cpu"
    assert metrics["num_samples"] == 45
    assert metrics["final_loss"] >= 0.0

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["num_samples"] == 45


def test_checkpoint_round_trip_supports_prediction(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(height=2, width=2, epochs=1, batch_size=4, hidden_channels=8, device="cpu"),
        output_dir=tmp_path,
    )
    checkpoint_path = result["checkpoint_path"]

    model, config, device = load_checkpoint(checkpoint_path, device="cpu")
    prediction = predict_next_state(
        checkpoint_path,
        GridState(height=2, width=2, row=0, col=0),
        Action.RIGHT,
        device="cpu",
    )

    assert config["height"] == 2
    assert device.type == "cpu"
    assert prediction.shape == (1, 2, 2)
    assert torch.isfinite(prediction).all()
    assert model.training is False

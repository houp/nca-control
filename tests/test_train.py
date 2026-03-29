from __future__ import annotations

import json

import torch

from nca_control.actions import Action
from nca_control.device import resolve_device
from nca_control.grid import GridState
from nca_control.inference import decode_prediction_state, hard_decode_exit_prediction, hard_decode_grid, load_checkpoint, predict_next_state
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


def test_train_one_step_creates_maze_checkpoint_and_metrics(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(
            task="maze",
            height=9,
            width=9,
            num_mazes=2,
            eval_num_mazes=1,
            epochs=1,
            batch_size=16,
            hidden_channels=8,
            device="cpu",
        ),
        output_dir=tmp_path,
    )

    metrics = result["metrics"]

    assert result["checkpoint_path"].exists()
    assert metrics["device"] == "cpu"
    assert metrics["num_samples"] > 0


def test_train_one_step_creates_maze_exit_checkpoint_and_metrics(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(
            task="maze_exit",
            height=9,
            width=9,
            num_mazes=2,
            eval_num_mazes=1,
            epochs=1,
            batch_size=16,
            hidden_channels=8,
            device="cpu",
        ),
        output_dir=tmp_path,
    )

    metrics = result["metrics"]

    assert result["checkpoint_path"].exists()
    assert metrics["device"] == "cpu"
    assert metrics["num_samples"] > 0


def test_checkpoint_round_trip_supports_prediction(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(
            height=2,
            width=2,
            epochs=1,
            batch_size=4,
            hidden_channels=8,
            perception_kernel_size=5,
            update_kernel_size=3,
            device="cpu",
        ),
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
    assert config["perception_kernel_size"] == 5
    assert config["update_kernel_size"] == 3
    assert device.type == "cpu"
    assert prediction.shape == (1, 2, 2)
    assert torch.isfinite(prediction).all()
    assert model.training is False
    assert model.perception.kernel_size == (5, 5)
    assert model.update[1].kernel_size == (3, 3)


def test_hard_decode_grid_returns_one_exact_active_cell() -> None:
    grid = torch.tensor([[[0.2, 0.7], [0.05, 0.05]]], dtype=torch.float32)

    decoded = hard_decode_grid(grid)

    assert decoded.tolist() == [[[0.0, 1.0], [0.0, 0.0]]]


def test_hard_decode_exit_prediction_supports_terminal_state() -> None:
    prediction = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.8], [0.6, 0.1]],
        ],
        dtype=torch.float32,
    )

    decoded = hard_decode_exit_prediction(prediction)

    assert decoded[0].sum().item() == 0.0
    assert decoded[1].tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_decode_prediction_state_marks_terminated_when_no_active_cell() -> None:
    prediction = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    state = decode_prediction_state(
        prediction,
        GridState(height=2, width=2, row=0, col=0, exit_cell=(0, 1)),
    )

    assert state.terminated is True
    assert state.exit_fill == frozenset({(0, 1)})


def test_decode_prediction_state_clamps_exit_fill_before_termination() -> None:
    prediction = torch.tensor(
        [
            [[0.1, 0.9], [0.2, 0.1]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    state = decode_prediction_state(
        prediction,
        GridState(height=2, width=2, row=0, col=0, exit_cell=(1, 1)),
    )

    assert state.terminated is False
    assert state.exit_fill == frozenset({(1, 1)})


def test_decode_prediction_state_expands_fill_deterministically_after_termination() -> None:
    prediction = torch.zeros((2, 3, 3), dtype=torch.float32)
    state = decode_prediction_state(
        prediction,
        GridState(height=3, width=3, row=1, col=1, exit_cell=(1, 1), terminated=True),
    )

    assert state.terminated is True
    assert state.exit_fill == frozenset({(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)})

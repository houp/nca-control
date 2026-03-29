from __future__ import annotations

import torch

from nca_control.evaluate import decode_argmax_positions, evaluate_checkpoint
from nca_control.train import TrainConfig, train_one_step


def test_decode_argmax_positions_returns_row_col_pairs() -> None:
    grids = torch.tensor(
        [
            [[[0.1, 0.2], [0.9, 0.3]]],
            [[[0.4, 0.8], [0.2, 0.1]]],
        ]
    )

    positions = decode_argmax_positions(grids)

    assert positions.tolist() == [[1, 0], [0, 1]]


def test_evaluate_checkpoint_returns_expected_metric_keys(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(height=2, width=2, epochs=1, batch_size=4, hidden_channels=8, device="cpu"),
        output_dir=tmp_path,
    )

    metrics = evaluate_checkpoint(result["checkpoint_path"], device="cpu")

    assert metrics["device"] == "cpu"
    assert metrics["num_samples"] == 20
    assert 0.0 <= metrics["argmax_accuracy"] <= 1.0
    assert metrics["mse"] >= 0.0


def test_evaluate_checkpoint_supports_maze_task(tmp_path) -> None:
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

    metrics = evaluate_checkpoint(result["checkpoint_path"], device="cpu")

    assert metrics["num_samples"] > 0
    assert 0.0 <= metrics["argmax_accuracy"] <= 1.0

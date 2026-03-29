from __future__ import annotations

from nca_control.evaluate import evaluate_rollout_checkpoint
from nca_control.train import TrainConfig, train_one_step


def test_evaluate_rollout_checkpoint_returns_expected_metrics(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(height=2, width=2, epochs=3, batch_size=4, hidden_channels=8, device="cpu"),
        output_dir=tmp_path,
    )

    metrics = evaluate_rollout_checkpoint(
        result["checkpoint_path"],
        num_sequences=8,
        steps_per_sequence=20,
        device="cpu",
        seed=0,
    )

    assert metrics["device"] == "cpu"
    assert metrics["num_sequences"] == 8
    assert metrics["steps_per_sequence"] == 20
    assert metrics["total_rollout_steps"] == 160
    assert 0.0 <= metrics["exact_rollout_rate"] <= 1.0
    assert metrics["num_failed_sequences"] >= 0
    assert isinstance(metrics["first_failures"], list)


def test_evaluate_rollout_checkpoint_supports_maze_task(tmp_path) -> None:
    result = train_one_step(
        TrainConfig(
            task="maze",
            height=9,
            width=9,
            num_mazes=2,
            eval_num_mazes=1,
            epochs=2,
            batch_size=16,
            hidden_channels=8,
            device="cpu",
        ),
        output_dir=tmp_path,
    )

    metrics = evaluate_rollout_checkpoint(
        result["checkpoint_path"],
        num_sequences=4,
        steps_per_sequence=10,
        device="cpu",
        seed=0,
    )

    assert metrics["device"] == "cpu"
    assert metrics["num_sequences"] == 4
    assert metrics["steps_per_sequence"] == 10
    assert 0.0 <= metrics["exact_rollout_rate"] <= 1.0


def test_evaluate_rollout_checkpoint_supports_maze_exit_task(tmp_path) -> None:
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

    metrics = evaluate_rollout_checkpoint(
        result["checkpoint_path"],
        num_sequences=4,
        steps_per_sequence=10,
        device="cpu",
        seed=0,
    )

    assert metrics["device"] == "cpu"
    assert metrics["num_sequences"] == 4
    assert metrics["steps_per_sequence"] == 10
    assert 0.0 <= metrics["exact_rollout_rate"] <= 1.0


def test_evaluate_rollout_checkpoint_accepts_grid_size_overrides(tmp_path) -> None:
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

    metrics = evaluate_rollout_checkpoint(
        result["checkpoint_path"],
        num_sequences=2,
        steps_per_sequence=5,
        device="cpu",
        seed=0,
        height=15,
        width=15,
    )

    assert metrics["rollout_height"] == 15
    assert metrics["rollout_width"] == 15
    assert 0.0 <= metrics["exact_rollout_rate"] <= 1.0

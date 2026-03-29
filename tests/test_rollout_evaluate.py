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

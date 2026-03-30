from __future__ import annotations

from pathlib import Path

from nca_control.seed_stability import (
    build_seed_run_record,
    render_seed_stability_report,
    summarize_seed_runs,
)


def test_build_seed_run_record_marks_exact_run() -> None:
    record = build_seed_run_record(
        seed=4,
        checkpoint_path=Path("checkpoint_mlx"),
        progress_path=Path("progress.jsonl"),
        metrics_path=Path("metrics.json"),
        train_metrics={"final_loss": 0.02, "loss_history": [0.2, 0.1, 0.02]},
        one_step={"full_state_accuracy": 1.0, "termination_accuracy": 1.0},
        rollout_30={"exact_rollout_rate": 1.0},
        rollout_50={"exact_rollout_rate": 1.0},
    )

    assert record["passed"] is True
    assert record["min_loss"] == 0.02


def test_summarize_seed_runs_computes_pass_fraction_and_extrema() -> None:
    seed_runs = [
        build_seed_run_record(
            seed=0,
            checkpoint_path=Path("seed0/checkpoint_mlx"),
            progress_path=Path("seed0/progress.jsonl"),
            metrics_path=Path("seed0/metrics.json"),
            train_metrics={"final_loss": 0.02, "loss_history": [0.3, 0.02]},
            one_step={"full_state_accuracy": 1.0, "termination_accuracy": 1.0},
            rollout_30={"exact_rollout_rate": 1.0},
            rollout_50={"exact_rollout_rate": 1.0},
        ),
        build_seed_run_record(
            seed=4,
            checkpoint_path=Path("seed4/checkpoint_mlx"),
            progress_path=Path("seed4/progress.jsonl"),
            metrics_path=Path("seed4/metrics.json"),
            train_metrics={"final_loss": 0.15, "loss_history": [0.3, 0.2, 0.15]},
            one_step={"full_state_accuracy": 0.9, "termination_accuracy": 0.8},
            rollout_30={"exact_rollout_rate": 0.0},
            rollout_50={"exact_rollout_rate": 0.0},
        ),
    ]

    summary = summarize_seed_runs(seed_runs)

    assert summary["num_runs"] == 2
    assert summary["pass_count"] == 1
    assert summary["pass_fraction"] == 0.5
    assert summary["pass_seeds"] == [0]
    assert summary["fail_seeds"] == [4]
    assert summary["best_seed"] == 0
    assert summary["worst_seed"] == 4
    assert summary["final_loss"]["median"] == 0.08499999999999999


def test_render_seed_stability_report_includes_pass_fraction_and_table() -> None:
    seed_runs = [
        build_seed_run_record(
            seed=0,
            checkpoint_path=Path("seed0/checkpoint_mlx"),
            progress_path=Path("seed0/progress.jsonl"),
            metrics_path=Path("seed0/metrics.json"),
            train_metrics={"final_loss": 0.02, "loss_history": [0.2, 0.02]},
            one_step={"full_state_accuracy": 1.0, "termination_accuracy": 1.0},
            rollout_30={"exact_rollout_rate": 1.0},
            rollout_50={"exact_rollout_rate": 1.0},
        )
    ]
    payload = {
        "experiment_config": {
            "task": "maze_exit",
            "height": 9,
            "width": 9,
            "num_mazes": 96,
            "hidden_channels": 12,
            "perception_kernel_size": 3,
            "update_kernel_size": 1,
            "batch_size": 128,
            "epochs": 500,
            "seeds": [0],
        },
        "seed_runs": seed_runs,
        "aggregate": summarize_seed_runs(seed_runs),
    }

    report = render_seed_stability_report(payload)

    assert "# MLX Seed Stability Sweep" in report
    assert "exact pass fraction: `1/1 = 1.000`" in report
    assert "| 0 | `0.020000` | `0.020000` | `1.000000` | `1.000000` | `1.000000` | Exact |" in report

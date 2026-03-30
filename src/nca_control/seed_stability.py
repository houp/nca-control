from __future__ import annotations

"""Helpers for summarizing multi-seed training stability experiments."""

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_seed_run_record(
    *,
    seed: int,
    checkpoint_path: Path,
    progress_path: Path,
    metrics_path: Path,
    train_metrics: dict[str, Any],
    one_step: dict[str, Any],
    rollout_30: dict[str, Any],
    rollout_50: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one seed's training and evaluation results into a stable schema."""

    loss_history = [float(value) for value in train_metrics.get("loss_history", [])]
    min_loss = min(loss_history) if loss_history else float(train_metrics["final_loss"])
    passed = (
        float(one_step.get("full_state_accuracy", 0.0)) == 1.0
        and float(one_step.get("termination_accuracy", 0.0)) == 1.0
        and float(rollout_30.get("exact_rollout_rate", 0.0)) == 1.0
        and float(rollout_50.get("exact_rollout_rate", 0.0)) == 1.0
    )
    return {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "progress_path": str(progress_path),
        "metrics_path": str(metrics_path),
        "train_metrics": train_metrics,
        "one_step": one_step,
        "rollout_30": rollout_30,
        "rollout_50": rollout_50,
        "final_loss": float(train_metrics["final_loss"]),
        "min_loss": float(min_loss),
        "passed": passed,
    }


def summarize_seed_runs(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute compact descriptive statistics over a multi-seed experiment."""

    if not seed_runs:
        raise ValueError("seed_runs must not be empty")

    ordered_runs = sorted(seed_runs, key=lambda record: int(record["seed"]))
    final_losses = [float(record["final_loss"]) for record in ordered_runs]
    min_losses = [float(record["min_loss"]) for record in ordered_runs]
    pass_records = [record for record in ordered_runs if bool(record["passed"])]
    fail_records = [record for record in ordered_runs if not bool(record["passed"])]
    best_run = min(ordered_runs, key=lambda record: float(record["final_loss"]))
    worst_run = max(ordered_runs, key=lambda record: float(record["final_loss"]))

    return {
        "num_runs": len(ordered_runs),
        "pass_count": len(pass_records),
        "pass_fraction": len(pass_records) / len(ordered_runs),
        "pass_seeds": [int(record["seed"]) for record in pass_records],
        "fail_seeds": [int(record["seed"]) for record in fail_records],
        "final_loss": _summarize_numeric(final_losses),
        "min_loss": _summarize_numeric(min_losses),
        "best_seed": int(best_run["seed"]),
        "best_final_loss": float(best_run["final_loss"]),
        "worst_seed": int(worst_run["seed"]),
        "worst_final_loss": float(worst_run["final_loss"]),
    }


def render_seed_stability_report(summary: dict[str, Any]) -> str:
    """Render a markdown summary suitable for quick inspection in `runs/`."""

    config = summary["experiment_config"]
    aggregate = summary["aggregate"]
    seed_runs = summary["seed_runs"]

    lines = [
        "# MLX Seed Stability Sweep",
        "",
        "## Experiment Config",
        "",
        f"- task: `{config['task']}`",
        f"- grid: `{config['height']}x{config['width']}`",
        f"- hidden channels: `{config['hidden_channels']}`",
        f"- perception kernel: `{config['perception_kernel_size']}`",
        f"- update kernel: `{config['update_kernel_size']}`",
        f"- mazes: `{config['num_mazes']}`",
        f"- epochs: `{config['epochs']}`",
        f"- batch size: `{config['batch_size']}`",
        f"- seeds: `{config['seeds']}`",
        "",
        "## Aggregate Result",
        "",
        f"- exact pass fraction: `{aggregate['pass_count']}/{aggregate['num_runs']} = {aggregate['pass_fraction']:.3f}`",
        f"- pass seeds: `{aggregate['pass_seeds']}`",
        f"- fail seeds: `{aggregate['fail_seeds']}`",
        f"- final loss mean ± std: `{aggregate['final_loss']['mean']:.6f} ± {aggregate['final_loss']['stdev']:.6f}`",
        f"- final loss median: `{aggregate['final_loss']['median']:.6f}`",
        f"- min loss mean ± std: `{aggregate['min_loss']['mean']:.6f} ± {aggregate['min_loss']['stdev']:.6f}`",
        f"- best seed/final loss: `{aggregate['best_seed']} / {aggregate['best_final_loss']:.6f}`",
        f"- worst seed/final loss: `{aggregate['worst_seed']} / {aggregate['worst_final_loss']:.6f}`",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Final Loss | Min Loss | One-Step | `30x30` | `50x50` | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in seed_runs:
        lines.append(
            "| {seed} | `{final_loss:.6f}` | `{min_loss:.6f}` | `{one_step:.6f}` | `{roll30:.6f}` | `{roll50:.6f}` | {result} |".format(
                seed=int(record["seed"]),
                final_loss=float(record["final_loss"]),
                min_loss=float(record["min_loss"]),
                one_step=float(record["one_step"]["full_state_accuracy"]),
                roll30=float(record["rollout_30"]["exact_rollout_rate"]),
                roll50=float(record["rollout_50"]["exact_rollout_rate"]),
                result="Exact" if bool(record["passed"]) else "Failed",
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_seed_stability_outputs(output_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Persist the machine-readable summary and the human-readable markdown report."""

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_seed_stability_report(summary), encoding="utf-8")
    return summary_path, report_path


def _summarize_numeric(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        stdev = 0.0
    else:
        stdev = float(statistics.stdev(values))
    return {
        "mean": float(statistics.fmean(values)),
        "stdev": stdev,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }

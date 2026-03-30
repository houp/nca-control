from __future__ import annotations

from pathlib import Path

import typer

from nca_control.mlx_backend import evaluate_mlx_checkpoint, evaluate_rollout_mlx_checkpoint, train_one_step_mlx
from nca_control.seed_stability import build_seed_run_record, summarize_seed_runs, write_seed_stability_outputs
from nca_control.train import TrainConfig

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    task: str = typer.Option("maze_exit"),
    height: int = typer.Option(9, min=1),
    width: int = typer.Option(9, min=1),
    num_mazes: int = typer.Option(96, min=1),
    eval_num_mazes: int = typer.Option(8, min=1),
    hidden_channels: int = typer.Option(12, min=1),
    perception_kernel_size: int = typer.Option(3, min=1),
    update_kernel_size: int = typer.Option(1, min=1),
    batch_size: int = typer.Option(128, min=1),
    epochs: int = typer.Option(500, min=1),
    learning_rate: float = typer.Option(1e-3, min=0.0),
    seeds: str = typer.Option("0,1,2,3,4,5,6,7"),
    rollout_num_sequences: int = typer.Option(32, min=1),
    rollout_steps_per_sequence: int = typer.Option(100, min=1),
    rollout_seed_base: int = typer.Option(20_000),
) -> None:
    """Train and evaluate one fixed MLX architecture across multiple random seeds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_seeds = _parse_int_list(seeds)
    experiment_config = {
        "task": task,
        "height": height,
        "width": width,
        "num_mazes": num_mazes,
        "eval_num_mazes": eval_num_mazes,
        "hidden_channels": hidden_channels,
        "perception_kernel_size": perception_kernel_size,
        "update_kernel_size": update_kernel_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seeds": parsed_seeds,
        "rollout_num_sequences": rollout_num_sequences,
        "rollout_steps_per_sequence": rollout_steps_per_sequence,
    }
    seed_runs: list[dict[str, object]] = []

    for seed in parsed_seeds:
        typer.echo(f"seed={seed} status=train")
        run_dir = output_dir / f"seed{seed}"
        train_result = train_one_step_mlx(
            TrainConfig(
                task=task,
                height=height,
                width=width,
                num_mazes=num_mazes,
                eval_num_mazes=eval_num_mazes,
                hidden_channels=hidden_channels,
                perception_kernel_size=perception_kernel_size,
                update_kernel_size=update_kernel_size,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                device="mlx",
                seed=seed,
            ),
            output_dir=run_dir,
            progress_printer=typer.echo,
        )
        checkpoint_path = Path(train_result["checkpoint_path"])
        typer.echo(f"seed={seed} status=eval_one_step")
        one_step = evaluate_mlx_checkpoint(checkpoint_path)
        typer.echo(f"seed={seed} status=eval_rollout_30")
        rollout_30 = evaluate_rollout_mlx_checkpoint(
            checkpoint_path,
            num_sequences=rollout_num_sequences,
            steps_per_sequence=rollout_steps_per_sequence,
            seed=rollout_seed_base + seed,
            height_override=30,
            width_override=30,
        )
        typer.echo(f"seed={seed} status=eval_rollout_50")
        rollout_50 = evaluate_rollout_mlx_checkpoint(
            checkpoint_path,
            num_sequences=rollout_num_sequences,
            steps_per_sequence=rollout_steps_per_sequence,
            seed=rollout_seed_base + 10_000 + seed,
            height_override=50,
            width_override=50,
        )
        seed_runs.append(
            build_seed_run_record(
                seed=seed,
                checkpoint_path=checkpoint_path,
                progress_path=Path(train_result["progress_path"]),
                metrics_path=Path(train_result["metrics_path"]),
                train_metrics=dict(train_result["metrics"]),
                one_step=one_step,
                rollout_30=rollout_30,
                rollout_50=rollout_50,
            )
        )
        summary = {
            "experiment_config": experiment_config,
            "seed_runs": seed_runs,
            "aggregate": summarize_seed_runs(seed_runs),
        }
        summary_path, report_path = write_seed_stability_outputs(output_dir, summary)
        typer.echo(f"seed={seed} passed={seed_runs[-1]['passed']}")
        typer.echo(f"summary={summary_path}")
        typer.echo(f"report={report_path}")

    final_summary = {
        "experiment_config": experiment_config,
        "seed_runs": seed_runs,
        "aggregate": summarize_seed_runs(seed_runs),
    }
    summary_path, report_path = write_seed_stability_outputs(output_dir, final_summary)
    typer.echo(f"summary={summary_path}")
    typer.echo(f"report={report_path}")
    typer.echo(f"pass_fraction={final_summary['aggregate']['pass_fraction']:.6f}")


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    app()

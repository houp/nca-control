from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import typer

from nca_control.mlx_backend import train_one_step_mlx
from nca_control.train import TrainConfig, train_one_step

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    task: str = typer.Option("maze_exit"),
    height: int = typer.Option(9, min=1),
    width: int = typer.Option(9, min=1),
    num_mazes: int = typer.Option(16, min=1),
    eval_num_mazes: int = typer.Option(4, min=1),
    epochs: int = typer.Option(40, min=1),
    batch_size: int = typer.Option(64, min=1),
    hidden_channels: int = typer.Option(32, min=1),
    perception_kernel_size: int = typer.Option(3, min=1),
    update_kernel_size: int = typer.Option(1, min=1),
    learning_rate: float = typer.Option(1e-3, min=0.0),
    seed: int = typer.Option(0),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
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
        device="cpu",
        seed=seed,
    )
    results: dict[str, dict[str, object]] = {}
    torch_cpu = train_one_step(config, output_dir / "torch_cpu")
    results["torch_cpu"] = dict(torch_cpu["metrics"])
    try:
        torch_mps = train_one_step(
            TrainConfig(**{**asdict(config), "device": "mps"}),
            output_dir / "torch_mps",
        )
        results["torch_mps"] = dict(torch_mps["metrics"])
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        results["torch_mps"] = {"error": str(exc)}
    mlx_result = train_one_step_mlx(
        TrainConfig(**{**asdict(config), "device": "mlx"}),
        output_dir / "mlx",
    )
    results["mlx"] = dict(mlx_result["metrics"])

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"summary={summary_path}")
    for backend, metrics in results.items():
        typer.echo(f"[{backend}]")
        for key, value in metrics.items():
            typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()

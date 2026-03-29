from __future__ import annotations

from pathlib import Path

import typer

from nca_control.mlx_backend import evaluate_rollout_mlx_checkpoint

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    height: int = typer.Option(..., min=1),
    width: int = typer.Option(..., min=1),
    num_sequences: int = typer.Option(64, min=1),
    steps_per_sequence: int = typer.Option(200, min=1),
    seed: int = typer.Option(0),
    max_reported_failures: int = typer.Option(8, min=0),
) -> None:
    metrics = evaluate_rollout_mlx_checkpoint(
        checkpoint,
        num_sequences=num_sequences,
        steps_per_sequence=steps_per_sequence,
        seed=seed,
        max_reported_failures=max_reported_failures,
        height_override=height,
        width_override=width,
    )
    for key, value in metrics.items():
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()

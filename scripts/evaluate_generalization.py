from __future__ import annotations

import json
from pathlib import Path

import typer

from nca_control.evaluate import evaluate_rollout_checkpoint

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    height: int = typer.Option(..., min=1),
    width: int = typer.Option(..., min=1),
    num_sequences: int = typer.Option(64, min=1),
    steps_per_sequence: int = typer.Option(200, min=1),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(0),
) -> None:
    metrics = evaluate_rollout_checkpoint(
        checkpoint,
        height=height,
        width=width,
        num_sequences=num_sequences,
        steps_per_sequence=steps_per_sequence,
        device=device,
        seed=seed,
    )
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()

from __future__ import annotations

from pathlib import Path

import typer

from nca_control.mlx_backend import evaluate_mlx_checkpoint

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    batch_size: int = typer.Option(256, min=1),
) -> None:
    metrics = evaluate_mlx_checkpoint(checkpoint, batch_size=batch_size)
    for key, value in metrics.items():
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()

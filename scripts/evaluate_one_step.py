from __future__ import annotations

import json
from pathlib import Path

import typer

from nca_control.evaluate import evaluate_checkpoint

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    device: str = typer.Option("auto"),
) -> None:
    metrics = evaluate_checkpoint(checkpoint, device=device)
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()

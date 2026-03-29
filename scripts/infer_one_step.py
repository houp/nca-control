from __future__ import annotations

from pathlib import Path

import torch
import typer

from nca_control.actions import Action
from nca_control.grid import GridState
from nca_control.inference import predict_next_state

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    height: int = typer.Option(..., min=1),
    width: int = typer.Option(..., min=1),
    row: int = typer.Option(..., min=0),
    col: int = typer.Option(..., min=0),
    action: Action = typer.Option(...),
    value: float = typer.Option(1.0),
    device: str = typer.Option("auto"),
) -> None:
    prediction = predict_next_state(
        checkpoint,
        GridState(height=height, width=width, row=row, col=col, value=value),
        action,
        device=device,
    )
    flat_index = int(torch.argmax(prediction[0]).item())
    pred_row = flat_index // width
    pred_col = flat_index % width

    typer.echo(f"predicted_row={pred_row}")
    typer.echo(f"predicted_col={pred_col}")
    typer.echo(f"predicted_max={prediction[0, pred_row, pred_col].item():.6f}")
    for rendered_row in prediction[0]:
        typer.echo(" ".join(f"{cell.item():0.4f}" for cell in rendered_row))


if __name__ == "__main__":
    app()

from __future__ import annotations

import typer

from .grid import GridState
from .simulation import parse_actions, rollout_states

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Deterministic control and research utilities."""


@app.command()
def simulate(
    height: int = typer.Option(..., min=1),
    width: int = typer.Option(..., min=1),
    row: int = typer.Option(..., min=0),
    col: int = typer.Option(..., min=0),
    value: float = typer.Option(1.0),
    actions: str = typer.Option("", help="Comma-separated actions: up,down,left,right,none"),
) -> None:
    """Render a deterministic scripted rollout in plain text."""
    initial_state = GridState(height=height, width=width, row=row, col=col, value=value)
    frames = rollout_states(initial_state, parse_actions(actions))

    for step_index, frame in enumerate(frames):
        typer.echo(f"step={step_index} row={frame.row} col={frame.col} value={frame.value}")
        typer.echo(frame.as_text())
        if step_index != len(frames) - 1:
            typer.echo("")


if __name__ == "__main__":
    app()

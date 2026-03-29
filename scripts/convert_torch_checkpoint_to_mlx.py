from __future__ import annotations

from pathlib import Path

import typer

from nca_control.mlx_backend import convert_torch_checkpoint_to_mlx

app = typer.Typer(add_completion=False)


@app.command()
def main(
    torch_checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    result = convert_torch_checkpoint_to_mlx(torch_checkpoint, output_dir)
    for key, value in result.items():
        typer.echo(f"{key}={value}")


if __name__ == "__main__":
    app()

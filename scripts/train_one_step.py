from __future__ import annotations

from pathlib import Path

import typer

from nca_control.train import TrainConfig, train_one_step

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    height: int = typer.Option(6, min=1),
    width: int = typer.Option(6, min=1),
    value: float = typer.Option(1.0),
    hidden_channels: int = typer.Option(32, min=1),
    batch_size: int = typer.Option(32, min=1),
    epochs: int = typer.Option(100, min=1),
    learning_rate: float = typer.Option(1e-3, min=0.0),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(0),
) -> None:
    result = train_one_step(
        TrainConfig(
            height=height,
            width=width,
            value=value,
            hidden_channels=hidden_channels,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
        ),
        output_dir=output_dir,
    )
    metrics = result["metrics"]
    typer.echo(f"checkpoint={result['checkpoint_path']}")
    typer.echo(f"metrics={result['metrics_path']}")
    typer.echo(f"device={metrics['device']}")
    typer.echo(f"final_loss={metrics['final_loss']:.6f}")


if __name__ == "__main__":
    app()

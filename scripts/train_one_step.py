from __future__ import annotations

from pathlib import Path

import typer

from nca_control.train import TrainConfig, train_one_step

app = typer.Typer(add_completion=False)


@app.command()
def main(
    task: str = typer.Option("plain"),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    height: int = typer.Option(6, min=1),
    width: int = typer.Option(6, min=1),
    value: float = typer.Option(1.0),
    num_mazes: int = typer.Option(32, min=1),
    maze_seed: int = typer.Option(0),
    eval_num_mazes: int = typer.Option(8, min=1),
    eval_seed_offset: int = typer.Option(10_000),
    hidden_channels: int = typer.Option(32, min=1),
    batch_size: int = typer.Option(32, min=1),
    epochs: int = typer.Option(100, min=1),
    learning_rate: float = typer.Option(1e-3, min=0.0),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(0),
) -> None:
    result = train_one_step(
        TrainConfig(
            task=task,
            height=height,
            width=width,
            value=value,
            num_mazes=num_mazes,
            maze_seed=maze_seed,
            eval_num_mazes=eval_num_mazes,
            eval_seed_offset=eval_seed_offset,
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

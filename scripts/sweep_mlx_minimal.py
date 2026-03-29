from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from nca_control.mlx_backend import evaluate_mlx_checkpoint, evaluate_rollout_mlx_checkpoint, train_one_step_mlx
from nca_control.train import TrainConfig

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
    height: int = typer.Option(9, min=1),
    width: int = typer.Option(9, min=1),
    num_mazes: int = typer.Option(64, min=1),
    eval_num_mazes: int = typer.Option(8, min=1),
    epochs: int = typer.Option(300, min=1),
    batch_size: int = typer.Option(128, min=1),
    learning_rate: float = typer.Option(1e-3, min=0.0),
    perception_kernel_size: int = typer.Option(3, min=1),
    update_kernel_size: int = typer.Option(1, min=1),
    hidden_candidates: str = typer.Option("8,12,16,20,24,28,32"),
    reproducibility_seeds: str = typer.Option("0,1,2,3"),
    rollout_num_sequences: int = typer.Option(16, min=1),
    rollout_steps_per_sequence: int = typer.Option(50, min=1),
    rollout_seed_base: int = typer.Option(10_000),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_list = _parse_int_list(hidden_candidates)
    repro_seeds = _parse_int_list(reproducibility_seeds)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary: dict[str, object] = {
        "search_config": {
            "height": height,
            "width": width,
            "num_mazes": num_mazes,
            "eval_num_mazes": eval_num_mazes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "perception_kernel_size": perception_kernel_size,
            "update_kernel_size": update_kernel_size,
            "hidden_candidates": hidden_list,
            "reproducibility_seeds": repro_seeds,
            "rollout_num_sequences": rollout_num_sequences,
            "rollout_steps_per_sequence": rollout_steps_per_sequence,
        },
        "candidate_results": [],
        "selected_model": None,
    }

    typer.echo(f"search_hidden={hidden_list}")
    for hidden_channels in hidden_list:
        typer.echo(f"candidate_hidden={hidden_channels}")
        screening = _run_candidate(
            output_dir=output_dir / f"h{hidden_channels}_screen_seed0",
            config=TrainConfig(
                task="maze_exit",
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
                device="mlx",
                seed=0,
            ),
            rollout_num_sequences=rollout_num_sequences,
            rollout_steps_per_sequence=rollout_steps_per_sequence,
            rollout_seed_base=rollout_seed_base,
        )
        candidate_record: dict[str, object] = {
            "hidden_channels": hidden_channels,
            "screening": screening,
            "reproducibility": [],
            "selected": False,
        }
        summary["candidate_results"].append(candidate_record)
        _write_outputs(summary_path, report_path, summary)

        if not screening["passed"]:
            typer.echo(f"candidate_hidden={hidden_channels} screening=failed")
            continue

        all_repro_passed = True
        for seed in repro_seeds:
            typer.echo(f"candidate_hidden={hidden_channels} repro_seed={seed}")
            repro_result = _run_candidate(
                output_dir=output_dir / f"h{hidden_channels}_repro_seed{seed}",
                config=TrainConfig(
                    task="maze_exit",
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
                    device="mlx",
                    seed=seed,
                ),
                rollout_num_sequences=rollout_num_sequences,
                rollout_steps_per_sequence=rollout_steps_per_sequence,
                rollout_seed_base=rollout_seed_base,
            )
            candidate_record["reproducibility"].append(repro_result)
            _write_outputs(summary_path, report_path, summary)
            if not repro_result["passed"]:
                all_repro_passed = False
                typer.echo(f"candidate_hidden={hidden_channels} repro_seed={seed} result=failed")
                break

        if all_repro_passed:
            candidate_record["selected"] = True
            summary["selected_model"] = {
                "hidden_channels": hidden_channels,
                "perception_kernel_size": perception_kernel_size,
                "update_kernel_size": update_kernel_size,
                "reproducibility_seeds": repro_seeds,
            }
            _write_outputs(summary_path, report_path, summary)
            typer.echo(f"selected_hidden={hidden_channels}")
            return

    _write_outputs(summary_path, report_path, summary)
    typer.echo("selected_hidden=none")


def _run_candidate(
    *,
    output_dir: Path,
    config: TrainConfig,
    rollout_num_sequences: int,
    rollout_steps_per_sequence: int,
    rollout_seed_base: int,
) -> dict[str, object]:
    train_result = train_one_step_mlx(config, output_dir=output_dir, progress_printer=typer.echo)
    checkpoint_path = train_result["checkpoint_path"]
    one_step = evaluate_mlx_checkpoint(checkpoint_path)
    rollout_30 = evaluate_rollout_mlx_checkpoint(
        checkpoint_path,
        num_sequences=rollout_num_sequences,
        steps_per_sequence=rollout_steps_per_sequence,
        seed=rollout_seed_base + config.seed,
        height_override=30,
        width_override=30,
    )
    rollout_50 = evaluate_rollout_mlx_checkpoint(
        checkpoint_path,
        num_sequences=rollout_num_sequences,
        steps_per_sequence=rollout_steps_per_sequence,
        seed=rollout_seed_base + 1_000 + config.seed,
        height_override=50,
        width_override=50,
    )
    passed = (
        one_step.get("full_state_accuracy") == 1.0
        and one_step.get("termination_accuracy") == 1.0
        and rollout_30.get("exact_rollout_rate") == 1.0
        and rollout_50.get("exact_rollout_rate") == 1.0
    )
    return {
        "config": asdict(config),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(train_result["metrics_path"]),
        "train_metrics": train_result["metrics"],
        "one_step": one_step,
        "rollout_30": rollout_30,
        "rollout_50": rollout_50,
        "passed": passed,
    }


def _write_outputs(summary_path: Path, report_path: Path, summary: dict[str, object]) -> None:
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(summary), encoding="utf-8")


def _render_report(summary: dict[str, object]) -> str:
    search_config = summary["search_config"]
    candidate_results = summary["candidate_results"]
    selected_model = summary["selected_model"]

    lines = [
        "# MLX Minimal Model Sweep",
        "",
        "## Search Config",
        "",
        f"- training grid: `{search_config['height']}x{search_config['width']}`",
        f"- mazes: `{search_config['num_mazes']}`",
        f"- epochs: `{search_config['epochs']}`",
        f"- batch size: `{search_config['batch_size']}`",
        f"- perception kernel: `{search_config['perception_kernel_size']}`",
        f"- update kernel: `{search_config['update_kernel_size']}`",
        f"- hidden candidates: `{search_config['hidden_candidates']}`",
        f"- reproducibility seeds: `{search_config['reproducibility_seeds']}`",
        "",
        "## Candidate Results",
        "",
        "| Hidden | Screen One-Step | Screen `30x30` | Screen `50x50` | Repro Seeds Passed | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in candidate_results:
        repro = result["reproducibility"]
        repro_passes = sum(1 for entry in repro if entry["passed"])
        lines.append(
            "| {hidden} | `{one_step:.6f}` | `{roll30:.6f}` | `{roll50:.6f}` | `{repro_passes}/{repro_total}` | {status} |".format(
                hidden=result["hidden_channels"],
                one_step=result["screening"]["one_step"]["full_state_accuracy"],
                roll30=result["screening"]["rollout_30"]["exact_rollout_rate"],
                roll50=result["screening"]["rollout_50"]["exact_rollout_rate"],
                repro_passes=repro_passes,
                repro_total=len(repro),
                status="Selected" if result["selected"] else ("Failed" if not result["screening"]["passed"] or repro_passes < len(repro) else "Passed"),
            )
        )

    lines.extend(["", "## Selected Model", ""])
    if selected_model is None:
        lines.append("No candidate satisfied the exactness and reproducibility criteria.")
    else:
        lines.extend(
            [
                f"- hidden channels: `{selected_model['hidden_channels']}`",
                f"- perception kernel: `{selected_model['perception_kernel_size']}`",
                f"- update kernel: `{selected_model['update_kernel_size']}`",
                f"- reproducibility seeds: `{selected_model['reproducibility_seeds']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    app()

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DATA_DIR = ROOT / "report" / "data"


def main() -> None:
    REPORT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_backend_benchmark()
    export_minimal_candidates()
    export_seed_stability()


def export_backend_benchmark() -> None:
    summary_path = ROOT / "runs" / "backend-bench-9x9-fixed" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [
        ("PyTorch CPU", float(payload["torch_cpu"]["samples_per_second"])),
        ("PyTorch MPS", float(payload["torch_mps"]["samples_per_second"])),
        ("MLX", float(payload["mlx"]["samples_per_second"])),
    ]
    write_csv(
        REPORT_DATA_DIR / "backend_benchmark.csv",
        ["backend", "samples_per_second"],
        rows,
    )


def export_minimal_candidates() -> None:
    summary_path = ROOT / "runs" / "mlx-minimal-sweep-tight" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[tuple[object, ...]] = []
    for candidate in payload["candidate_results"]:
        screening = candidate["screening"]
        reproducibility = candidate.get("reproducibility", [])
        repro_passes = int(sum(1 for run in reproducibility if bool(run["passed"])))
        selected = bool(candidate.get("selected", False))
        if selected:
            result = "selected"
        elif repro_passes != len(reproducibility):
            result = "failed"
        else:
            result = "screened"
        rows.append(
            (
                int(candidate["hidden_channels"]),
                float(screening["one_step"]["full_state_accuracy"]),
                float(screening["rollout_30"]["exact_rollout_rate"]),
                float(screening["rollout_50"]["exact_rollout_rate"]),
                len(reproducibility),
                repro_passes,
                float(repro_passes / max(1, len(reproducibility))),
                result,
            )
        )
    write_csv(
        REPORT_DATA_DIR / "minimal_candidates.csv",
        [
            "hidden_channels",
            "screen_full_state_accuracy",
            "screen_rollout_30",
            "screen_rollout_50",
            "repro_attempts",
            "repro_passes",
            "repro_pass_fraction",
            "result",
        ],
        rows,
    )


def export_seed_stability() -> None:
    summary_path = ROOT / "runs" / "mlx-seed-stability-96m-500e" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    seed_runs = payload["seed_runs"]

    write_csv(
        REPORT_DATA_DIR / "seed_stability_summary.csv",
        ["metric", "value"],
        [
            ("num_runs", aggregate["num_runs"]),
            ("pass_count", aggregate["pass_count"]),
            ("pass_fraction", aggregate["pass_fraction"]),
            ("final_loss_mean", aggregate["final_loss"]["mean"]),
            ("final_loss_stdev", aggregate["final_loss"]["stdev"]),
            ("final_loss_median", aggregate["final_loss"]["median"]),
            ("min_loss_mean", aggregate["min_loss"]["mean"]),
            ("min_loss_stdev", aggregate["min_loss"]["stdev"]),
            ("best_seed", aggregate["best_seed"]),
            ("best_final_loss", aggregate["best_final_loss"]),
            ("worst_seed", aggregate["worst_seed"]),
            ("worst_final_loss", aggregate["worst_final_loss"]),
        ],
    )

    write_csv(
        REPORT_DATA_DIR / "seed_stability_runs.csv",
        [
            "seed",
            "passed",
            "final_loss",
            "min_loss",
            "one_step_full_state_accuracy",
            "termination_accuracy",
            "rollout_30",
            "rollout_50",
        ],
        [
            (
                int(run["seed"]),
                1 if bool(run["passed"]) else 0,
                float(run["final_loss"]),
                float(run["min_loss"]),
                float(run["one_step"]["full_state_accuracy"]),
                float(run["one_step"]["termination_accuracy"]),
                float(run["rollout_30"]["exact_rollout_rate"]),
                float(run["rollout_50"]["exact_rollout_rate"]),
            )
            for run in seed_runs
        ],
    )

    loss_rows: list[tuple[object, ...]] = []
    for run in seed_runs:
        progress_path = Path(run["progress_path"])
        run_rows: list[tuple[object, ...]] = []
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            row = (
                int(run["seed"]),
                1 if bool(run["passed"]) else 0,
                int(record["epoch"]),
                float(record["loss"]),
            )
            loss_rows.append(row)
            run_rows.append(row[2:])
        write_csv(
            REPORT_DATA_DIR / f"seed_stability_loss_seed{int(run['seed'])}.csv",
            ["epoch", "loss"],
            run_rows,
        )
    write_csv(
        REPORT_DATA_DIR / "seed_stability_loss_curves.csv",
        ["seed", "passed", "epoch", "loss"],
        loss_rows,
    )


def write_csv(path: Path, header: list[str], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()

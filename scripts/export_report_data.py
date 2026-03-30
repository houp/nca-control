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
    export_loss_curves()


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


def export_loss_curves() -> None:
    runs = {
        "seed0_rerun": ROOT / "runs" / "final_h12_p3_u1_seed0_rerun" / "progress.jsonl",
        "seed4_exploratory": ROOT / "runs" / "final_h12_p3_u1_seed4" / "progress.jsonl",
    }
    rows: list[tuple[object, ...]] = []
    for run_label, progress_path in runs.items():
        run_rows: list[tuple[object, ...]] = []
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            row = (run_label, int(record["epoch"]), float(record["loss"]))
            rows.append(row)
            run_rows.append(row[1:])
        write_csv(
            REPORT_DATA_DIR / f"loss_{run_label}.csv",
            ["epoch", "loss"],
            run_rows,
        )
    write_csv(
        REPORT_DATA_DIR / "loss_curves.csv",
        ["run", "epoch", "loss"],
        rows,
    )


def write_csv(path: Path, header: list[str], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()

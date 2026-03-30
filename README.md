# Controllable NCA Research Experiment

This repository contains a stepwise research implementation for a controllable Neural Cellular Automata (NCA) system.

## AI-Generated Provenance

The contents of this project are AI generated. The implementation, documentation, report, and presentation were produced with OpenAI Codex using the GPT-5.4 model with High reasoning effort.

## Goal

Learn an NCA that preserves exactly one active cell and moves it on a 2D periodic grid according to user actions:

- `up`
- `down`
- `left`
- `right`
- `none`

The active cell must:

- never disappear
- never change its value
- move exactly one grid position per step
- wrap around grid boundaries

## Current Status

Step 1 establishes:

- project planning and tracking infrastructure
- a deterministic reference transition engine
- automated baseline tests for movement semantics

Step 2 now establishes:

- deterministic trajectory generation
- a text-mode simulation CLI for scripted control sequences

Step 3 now establishes:

- torch-based supervised transition dataset generation
- explicit control-channel encoding for future NCA training

Step 4 now establishes:

- a minimal trainable controllable NCA model in PyTorch
- verified forward-pass and backpropagation behavior

Step 5 now establishes:

- a checkpointed one-step supervised training script
- a checkpoint-loading inference script

Step 6 now establishes:

- full-dataset checkpoint evaluation
- quantitative evidence that the initial model/loss is not yet exact enough

Step 7 now establishes:

- an exactness-oriented training objective
- periodic-boundary-aware model perception
- a trained 6x6 checkpoint reaching `100%` one-step argmax accuracy in evaluation

Step 8 now establishes:

- a Tk-based interactive comparison app for reference vs learned control
- tested non-UI keyboard/control helpers for local visual verification

Step 12 now establishes:

- deterministic wall-aware movement
- a pure-Python maze generator for static blocked-cell layouts

Step 13 now establishes:

- wall-aware model inputs
- maze-generated supervised training data
- an exact maze-control checkpoint on generated mazes

Step 16 now establishes:

- deterministic maze exit semantics
- guaranteed solvable mazes with explicit start and exit cells

Step 17 now establishes:

- exit-aware browser/session rendering
- maze regeneration on every visualizer reset

Step 18 now establishes:

- an exit-aware maze training task with terminal lockout
- one-step and rollout evaluation for the learned exit-aware checkpoint
- exact decoded gameplay semantics on the verified smoke checkpoint

Step 22 now establishes:

- a fresh minimal-architecture sweep after clearing `runs/`
- a verified smaller exact model at `hidden_channels=32`, `perception_kernel_size=3`, `update_kernel_size=1`

Step 24 now establishes:

- per-epoch training progress artifacts in each run directory
- non-silent CLI training output with per-epoch loss and speed
- an MPS-vs-CPU throughput investigation showing that MPS loses on tiny `9x9` workloads but reaches parity on larger `30x30` runs

Step 25 now establishes:

- a parallel MLX backend for Apple Silicon without removing the PyTorch implementation
- torch-to-MLX parity checks plus native MLX training/evaluation scripts
- a measured three-way runtime comparison across PyTorch CPU, PyTorch MPS, and MLX

Step 27 now establishes:

- a stronger MLX-based revisit of the minimal-model problem
- a new smallest exact and reproducible `maze_exit` model at `hidden_channels=12`, `perception_kernel_size=3`, `update_kernel_size=1`

Step 28 now establishes:

- explicit boundary checks on `9`, `10`, and `11` hidden channels under the same strong MLX recipe
- confirmation that `9/3/1` and `11/3/1` are not reproducible and `10/3/1` is not exact
- exact post-selection rollout validation of the selected `12/3/1` model on `100x100` and `200x200` grids across four clean retraining seeds

Step 31 now establishes:

- a maintenance audit bringing README and tracking back in line with the current code base
- a small shared-logic refactor removing duplicated action-index movement code
- a fresh clean MLX retrain on the current code confirming the selected `12/3/1` model still reaches exact one-step behavior and exact larger-grid rollouts on a known-good seed

Step 32 now establishes:

- a short academic-style XeTeX report summarizing the implementation, experimental protocol, backend comparison, minimal-model search, and current reproducibility findings
- data-driven figures and tables exported from the recorded run artifacts under `runs/`
- a compiled PDF at `report/technical_report.pdf`

The report source is in `report/technical_report.tex`, and the corresponding build data for figures is in `report/data/`.

Step 34 now establishes:

- a beamer presentation summarizing the project motivation, implementation approach, backend comparison, minimal-model search, and verified findings
- a compiled slide deck at `report/technical_presentation.pdf`

The slide source is in `report/technical_presentation.tex`.

Step 38 now establishes:

- a dedicated MLX seed-stability sweep for the selected `12/3/1` model
- a stronger reproducibility result under `96` mazes and `500` epochs across seeds `0..7`
- exact `30x30` and `50x50` rollout behavior on all `8/8` tested seeds
- follow-up `100x100` and `200x200` rollout checks on additional representative checkpoints

Step 39 now establishes:

- direct confirmation that Python `3.14` is compatible with the current dependency set, test suite, and MLX training path
- promotion of Python `3.14` to the default interpreter target for the repository
- retention of Python `3.13` as a secondary re-test path after major changes

## Experimental CUDA Path

PyTorch now has an experimental CUDA-enabled code path for collaborators running the project on Linux with NVIDIA GPUs. In practical terms, the existing PyTorch training and evaluation entrypoints can be used with `--device cuda`, and `--device auto` will prefer CUDA when it is available.

This CUDA path is not tested in the current macOS/Apple Silicon environment and is not covered by the reported experimental results in this repository. The current verified results remain those obtained on CPU, MPS, and MLX in the recorded macOS experiments.

## Commands

Default interpreter target:

- Python `3.14`
- Python `3.13` remains supported as an occasional secondary re-test path after major changes

Train a one-step model:

```bash
.venv/bin/python scripts/train_one_step.py --output-dir runs/demo --height 6 --width 6 --epochs 100
```

On a Linux machine with CUDA, the same PyTorch path can be invoked explicitly as:

```bash
.venv/bin/python scripts/train_one_step.py --output-dir runs/demo-cuda --height 6 --width 6 --epochs 100 --device cuda
```

Each training run now writes live progress into:

- `runs/.../progress.jsonl`
- `runs/.../latest_status.json`
- `runs/.../metrics.json`

The CLI also prints one line per epoch with loss and throughput.

Train a maze-aware model:

```bash
.venv/bin/python scripts/train_one_step.py --task maze --output-dir runs/maze9 --height 9 --width 9 --num-mazes 32 --eval-num-mazes 8 --epochs 50
```

Train an exit-aware maze model:

```bash
.venv/bin/python scripts/train_one_step.py --task maze_exit --output-dir runs/maze-exit9 --height 9 --width 9 --num-mazes 32 --eval-num-mazes 8 --epochs 50
```

Train the currently selected minimal exact model:

```bash
.venv/bin/python scripts/train_mlx_one_step.py --task maze_exit --output-dir runs/mlx_h12_p3_u1 --height 9 --width 9 --num-mazes 64 --eval-num-mazes 8 --epochs 300 --batch-size 128 --hidden-channels 12 --perception-kernel-size 3 --update-kernel-size 1
```

Train the stronger current reproducibility recipe for the selected minimal model:

```bash
.venv/bin/python scripts/train_mlx_one_step.py --task maze_exit --output-dir runs/mlx_h12_p3_u1_strong --height 9 --width 9 --num-mazes 96 --eval-num-mazes 8 --epochs 500 --batch-size 128 --hidden-channels 12 --perception-kernel-size 3 --update-kernel-size 1
```

Run the multi-seed MLX stability sweep used for the updated reproducibility study:

```bash
.venv/bin/python scripts/sweep_mlx_seed_stability.py --output-dir runs/mlx-seed-stability-96m-500e --num-mazes 96 --epochs 500 --batch-size 128 --seeds 0,1,2,3,4,5,6,7 --rollout-num-sequences 32 --rollout-steps-per-sequence 100
```

Train the older 32-channel MLX reference recipe:

```bash
.venv/bin/python scripts/train_mlx_one_step.py --task maze_exit --output-dir runs/mlx_h32_p3_u1 --height 9 --width 9 --num-mazes 16 --eval-num-mazes 4 --epochs 150 --batch-size 64 --hidden-channels 32 --perception-kernel-size 3 --update-kernel-size 1
```

Evaluate a checkpoint:

```bash
.venv/bin/python scripts/evaluate_one_step.py --checkpoint runs/demo/checkpoint.pt
```

Evaluate long random rollouts:

```bash
.venv/bin/python scripts/evaluate_rollout.py --checkpoint runs/demo/checkpoint.pt --num-sequences 256 --steps-per-sequence 1000
```

Evaluate cross-grid rollout generalization on a different maze size:

```bash
.venv/bin/python scripts/evaluate_generalization.py --checkpoint runs/maze-exit9/checkpoint.pt --height 30 --width 30 --num-sequences 64 --steps-per-sequence 200
```

Evaluate an MLX checkpoint:

```bash
.venv/bin/python scripts/evaluate_mlx_one_step.py --checkpoint runs/mlx_h32_p3_u1/checkpoint_mlx
.venv/bin/python scripts/evaluate_mlx_generalization.py --checkpoint runs/mlx_h32_p3_u1/checkpoint_mlx --height 30 --width 30 --num-sequences 16 --steps-per-sequence 50
```

Infer one step from a chosen state/action:

```bash
.venv/bin/python scripts/infer_one_step.py --checkpoint runs/demo/checkpoint.pt --height 6 --width 6 --row 0 --col 5 --action right
```

Launch the interactive visual comparison app:

```bash
.venv/bin/python scripts/interactive_compare.py --checkpoint runs/demo/checkpoint.pt --height 6 --width 6
```

Then open `http://127.0.0.1:8000` in your browser. The visualizer advances on a fixed clock and applies `none` automatically when no keypress is queued. You can tune the clock with `--tick-ms`.

For the maze checkpoint, the visualizer automatically uses the maze seed stored in the checkpoint config:

```bash
.venv/bin/python scripts/interactive_compare.py --checkpoint runs/maze9/checkpoint.pt --height 9 --width 9 --port 8767
```

The same applies to exit-aware maze checkpoints. On reset, the browser app generates a fresh solvable maze:

```bash
.venv/bin/python scripts/interactive_compare.py --checkpoint runs/maze-exit9/checkpoint.pt --height 9 --width 9 --port 8768
```

## Planned Stack

- `uv` for environment and dependency management
- Python `3.14` as the main interpreter target
- Python `3.13` as a secondary re-test path
- MLX as the preferred Apple Silicon training backend
- PyTorch retained for CPU compatibility and future CUDA-class machines
- `pytest` for automated verification

## Repository Layout

- `src/nca_control/`: Python package
- `tests/`: automated tests
- `docs/`: planning and experiment documentation
- `tracking/`: step tracking files

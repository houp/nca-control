# Technical Report Plan

## Goal

Produce a short academic-style report that summarizes the current state of the controllable NCA maze-exit project, using only claims that are consistent with:

- the current code base
- tracked markdown documentation
- recorded run artifacts under `runs/`

The report should be typeset with XeTeX and compiled to PDF.

## Report Scope

The report will cover:

1. problem definition
2. deterministic environment semantics
3. model architecture and implementation
4. training and evaluation protocols
5. experimental results to date
6. current limitations and interpretation

The report will stay short and technical rather than trying to be a full conference paper.

## Verified Claims To Use

The report may rely on the following verified findings:

- the project implements a controllable NCA-like transition model for a `maze_exit` task with four semantic cell classes:
  - empty
  - blocked
  - active
  - exit-fill / terminal state
- the deterministic reference system supports:
  - wall-aware movement
  - periodic boundaries
  - explicit exit detection
  - post-terminal lockout
  - gradual terminal fill propagation
- MLX is currently the preferred Apple Silicon training backend for this project
- the stronger MLX minimal-model search protocol used:
  - `9x9` training grids
  - `64` mazes
  - `300` epochs
  - `batch_size=128`
  - reproducibility seeds `0,1,2,3`
- the current selected minimal exact model is:
  - hidden channels `12`
  - perception kernel `3`
  - update kernel `1`
- smaller boundary candidates fail under the same strong recipe:
  - `9/3/1` fails reproducibility
  - `10/3/1` fails screening
  - `11/3/1` fails reproducibility
- the selected `12/3/1` model is exact on:
  - one-step evaluation for seeds `0,1,2,3`
  - rollout checks on `30x30`, `50x50`, `100x100`, and `200x200` in the documented validation runs
- after the recent maintenance refactor:
  - tests still pass
  - a clean seed `0` retrain reproduced the exact regime
- an exploratory seed `4` run did not converge; this should be reported carefully as an observed failure outside the documented reproducibility seed set, not as a contradiction

## Figure Plan

The report should include at least the following visuals:

1. environment/model schematic
   - a TikZ diagram showing the channels and one-step transition structure
2. backend throughput comparison
   - bar chart using `runs/backend-bench-9x9-fixed/summary.json`
3. minimal-model search result
   - small bar/scatter/summary figure or table using `runs/mlx-minimal-sweep-tight/report.md`
4. training dynamics
   - loss curves for:
     - clean seed `0` rerun
     - exploratory seed `4` run
   using `runs/final_h12_p3_u1_seed0_rerun/progress.jsonl` and `runs/final_h12_p3_u1_seed4/progress.jsonl`

## Table Plan

The report should include:

1. task semantics table
2. minimal-model comparison table
3. verified rollout generalization table

## Bibliography Plan

The report should cite a small set of core references:

- neural cellular automata / differentiable self-organizing systems
- cellular automata background
- PyTorch
- MLX

Because the report is short, a concise bibliography is sufficient.

## Implementation Plan

1. create this plan and update tracking
2. add a small script to export report data from the recorded run artifacts
3. generate CSV files for plots
4. draft the LaTeX report section by section
5. compile with XeTeX
6. verify the output PDF and commit the step


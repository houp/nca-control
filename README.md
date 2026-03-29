# Controllable NCA Research Experiment

This repository contains a stepwise research implementation for a controllable Neural Cellular Automata (NCA) system.

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

Step 2 is in progress and adds:

- scripted text-mode rollouts
- CLI-based deterministic simulation

Step 2 now establishes:

- deterministic trajectory generation
- a text-mode simulation CLI for scripted control sequences

Step 3 now establishes:

- torch-based supervised transition dataset generation
- explicit control-channel encoding for future NCA training

Learned NCA training and interactive visualization follow after that.

## Planned Stack

- `uv` for environment and dependency management
- Python `3.13`
- PyTorch on Apple Silicon MPS for training/inference
- `pytest` for automated verification

## Repository Layout

- `src/nca_control/`: Python package
- `tests/`: automated tests
- `docs/`: planning and experiment documentation
- `tracking/`: step tracking files

# Project Plan

## Objective

Build a research-quality controllable NCA experiment where a single non-zero cell moves on a 2D grid according to discrete user actions while preserving:

- identity
- value
- uniqueness
- periodic boundary conditions

## Design Principles

1. Build a deterministic reference system first.
2. Separate environment semantics from learned model behavior.
3. Make every step observable, testable, and commit-sized.
4. Prefer tensor-friendly implementations that map cleanly to GPU execution.
5. Keep documentation synchronized with implementation progress.

## Research Scope

### Problem Statement

We want a learned NCA to emulate a transition rule:

- input: current grid state and control action
- output: next grid state
- constraint: exactly one active cell remains active
- invariant: active value is constant over time
- topology: toroidal 2D grid

### Initial Simplification

The first implementation step does **not** train a model. It defines the deterministic target behavior that later training must reproduce exactly.

## Proposed Milestones

## Step 1: Planning and deterministic baseline

- Create repository structure
- Document roadmap and testing approach
- Implement a deterministic transition engine
- Add exhaustive unit tests for action semantics

## Step 2: Environment and textual simulation

- Add CLI simulation loop
- Render grid state in plain text
- Support scripted action sequences
- Add golden tests for trajectories

## Step 3: Data generation

- Generate supervised transition pairs from the deterministic baseline
- Support batched tensor generation
- Add dataset-level invariance checks

## Step 4: Minimal learned NCA model

- Implement the smallest viable NCA in PyTorch
- Encode action as control channels
- Train on one-step prediction
- Verify exactness on small grids

## Step 5: Multi-step training and stability

- Add rollout training
- Penalize disappearance, duplication, and value drift
- Evaluate long-horizon control fidelity

## Step 6: Interactive visualizer

- Add keyboard-driven live inference app
- Show action, position, and rollout status
- Compare learned model vs deterministic reference

## Step 7: Research instrumentation

- Metrics for control accuracy, persistence, value invariance, and boundary wrapping
- Experiment configs
- Result logging and markdown reports

## Maze Extension

The next experiment variant adds three cell types:

- empty cells: traversable
- blocked cells: static walls
- active cell: the controllable player

### Maze Variant Plan

1. Add deterministic wall-aware stepping and a maze generator.
2. Extend state encoding to include wall information.
3. Train the NCA on generated mazes so movement respects walls.
4. Add rollout evaluation on mazes.
5. Update the interactive visualizer to render empty, blocked, and active cells.
6. Add a goal-state exit cell, terminal lockout, and gradual end-state spread.

## Acceptance Criteria

### Functional

- `none` keeps the active cell stationary
- directional inputs move the cell by one step
- wrap-around works on all edges
- active cell count remains exactly one
- active value remains unchanged

### Engineering

- reproducible environment setup through `uv`
- automated tests for each layer
- markdown-based tracking for every step
- local git commit after each completed step

## Known Risks

- MPS-specific PyTorch behavior may differ from CUDA assumptions
- sandboxed execution may not expose Apple GPU availability even when MPS works outside the sandbox
- exact single-cell preservation may require strong inductive bias or constrained decoding
- interactive rendering choice must remain lightweight on macOS

## Immediate Next Steps

1. Lock deterministic semantics in code and tests.
2. Add text-mode simulation for scripted verification.
3. Install and validate the PyTorch-based training stack.

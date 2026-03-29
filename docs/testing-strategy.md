# Testing Strategy

## Testing Goals

We need to verify both correctness and research utility.

## Layered Test Plan

## 1. Deterministic semantics

These tests validate the reference behavior:

- no-op action preserves position
- each direction moves by exactly one cell
- periodic boundaries wrap correctly
- active value is preserved
- exactly one active cell exists before and after the step

## 2. Text simulation

These tests will validate:

- deterministic action sequences
- reproducible rendered trajectories
- CLI argument parsing

## 3. Dataset generation

These tests will validate:

- shapes and dtypes
- label correctness
- action conditioning correctness
- invariants over batched samples

## 4. Learned model

These tests will validate:

- forward-pass tensor shapes
- device placement
- training step execution
- checkpoint save/load

## 5. End-to-end control

These tests will validate:

- multi-step rollout correctness
- long-horizon persistence
- exact or near-exact control accuracy metrics

## 6. Maze exit dynamics

These tests validate:

- explicit exit-cell encoding in datasets and inference inputs
- terminal lockout after the player reaches the exit
- gradual exit-color spread over future steps
- browser-session reset generating a fresh solvable maze

## Verification Standard

Each implementation step should include:

1. unit tests for the new behavior
2. a local verification command
3. a documented result in tracking files before commit

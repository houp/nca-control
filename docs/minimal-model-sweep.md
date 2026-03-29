# Minimal Model Sweep

## Goal

Find the smallest fresh `maze_exit` model that still behaves correctly on multiple grid sizes after the `runs/` directory reset.

## Search Strategy

The sweep started from the simplest architectures and increased complexity only when the previous candidate failed:

1. keep the task fixed at `maze_exit`
2. train on `9x9` mazes from scratch
3. require:
   - one-step `full_state_accuracy = 1.0`
   - cross-grid rollout `exact_rollout_rate = 1.0` on `30x30`
   - cross-grid rollout `exact_rollout_rate = 1.0` on `50x50`
4. stop at the first candidate that meets all criteria

## Candidate Results

| Hidden | Perception Kernel | Update Kernel | Epochs | One-Step Full-State Accuracy | `30x30` Rollout | `50x50` Rollout | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 20 | `0.3023` | `0.0` | not needed | Failed |
| 1 | 3 | 1 | 60 | `0.3023` | `0.0` | not needed | Failed |
| 2 | 3 | 1 | 80 | `0.3023` | `0.0` | not needed | Failed |
| 4 | 3 | 1 | 120 | `0.7256` | `0.0` | `0.0` | Failed |
| 8 | 3 | 1 | 150 | `0.7488` | `0.0` | `0.0` | Failed |
| 16 | 3 | 1 | 180 | `0.7279` | `0.0` | `0.0` | Failed |
| 4 | 3 | 3 | 150 | `0.7256` | `0.0` | not needed | Failed |
| 32 | 3 | 1 | 150 | `1.0` | `1.0` | `1.0` | Selected |

## Selected Minimal Model

The first architecture that satisfied the exactness criteria was:

- hidden channels: `32`
- perception kernel size: `3`
- update kernel size: `1`

Training output:

- checkpoint: `runs/sweep_h32_p3_u1/checkpoint.pt`
- metrics: `runs/sweep_h32_p3_u1/metrics.json`
- final loss: `0.027645`
- training time on CPU in this environment: `101.678 s`

Verification output:

- one-step evaluation:
  - `full_state_accuracy = 1.0`
  - `termination_accuracy = 1.0`
- generalization evaluation:
  - `30x30`, `16` sequences, `50` steps: `exact_rollout_rate = 1.0`
  - `50x50`, `16` sequences, `50` steps: `exact_rollout_rate = 1.0`

## Interpretation

- `1x1` perception is too weak because it cannot observe neighboring cells and therefore cannot implement movement.
- very small widths (`1`, `2`) collapse into a bad regime even with a valid `3x3` perception kernel.
- moderate widths (`4`, `8`, `16`) learn much of the task but still make systematic one-cell directional mistakes.
- increasing the hidden update kernel from `1` to `3` did not recover exactness at low width.
- under this sweep, `32` hidden channels is the first architecture that works exactly while remaining substantially smaller than the previous `64`-channel reference model.

## MPS Replication Check

The selected `32/3/1` architecture was retrained from scratch on MPS with three seeds using the same training recipe:

- task: `maze_exit`
- grid: `9x9`
- mazes: `16`
- epochs: `150`
- batch size: `64`

Results:

| Seed | Device | Final Loss | One-Step Full-State Accuracy | `30x30` Rollout | `50x50` Rollout | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `mps` | `0.027663` | `1.0` | `1.0` | `1.0` | Exact |
| 1 | `mps` | `0.029040` | `0.998837` | `1.0` | `1.0` | Near-exact |
| 2 | `mps` | `0.031843` | `1.0` | `1.0` | `1.0` | Exact |

Interpretation:

- the architecture is robust enough to preserve exact rollout behavior across larger grids in all three tested runs
- the current training recipe is **not yet perfectly reproducible** at the stricter one-step exactness level, because one of the three fresh MPS runs missed one-step perfection by a small margin
- the next sensible step is to harden the recipe, likely by increasing training budget or adding checkpoint selection criteria, while keeping the same `32/3/1` architecture

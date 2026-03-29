# MLX Minimal Model Sweep

## Goal

Revisit the minimal `maze_exit` architecture using the faster MLX backend, a larger training set, longer runs, and an explicit reproducibility gate.

## Motivation

The earlier result in `docs/minimal-model-sweep.md` selected:

- hidden channels: `32`
- perception kernel: `3`
- update kernel: `1`

That sweep relied on much smaller budgets, mostly around `150` epochs and `16` mazes. After adding MLX, those budgets were no longer compelling because Apple-native training became much faster.

## Search Protocol

The revisit followed `docs/mlx-minimal-model-plan.md`:

- backend: `MLX`
- task: `maze_exit`
- training grid: `9x9`
- mazes: `64`
- epochs: `300`
- batch size: `128`
- perception kernel: `3`
- update kernel: `1`
- hidden candidates: `8, 12, 16, 20, 24, 28, 32`
- reproducibility seeds: `0, 1, 2, 3`

Acceptance criteria:

1. one-step:
   - `full_state_accuracy = 1.0`
   - `termination_accuracy = 1.0`
2. rollout:
   - `30x30 exact_rollout_rate = 1.0`
   - `50x50 exact_rollout_rate = 1.0`
3. reproducibility:
   - all clean retrains on seeds `0,1,2,3` must pass the same criteria

## Candidate Results

The automated sweep output is stored in:

- `runs/mlx-minimal-sweep-strong/summary.json`
- `runs/mlx-minimal-sweep-strong/report.md`

Observed candidate results:

| Hidden | Screen One-Step | `30x30` Rollout | `50x50` Rollout | Repro Seeds Passed | Result |
| --- | --- | --- | --- | --- | --- |
| `8` | `0.929240` | `0.0` | `0.0625` | `0/0` | Failed |
| `12` | `1.0` | `1.0` | `1.0` | `4/4` | Selected |

The sweep stopped at `12` because the protocol was to stop at the first exact and reproducible candidate.

## Selected Minimal Model

Under the stronger MLX regime, the first accepted model is:

- hidden channels: `12`
- perception kernel: `3`
- update kernel: `1`

This is materially smaller than the previous `32/3/1` result.

## Reproducibility Results

The selected `12/3/1` architecture was retrained from scratch four times:

| Seed | Final Loss | One-Step Full-State Accuracy | `30x30` Rollout | `50x50` Rollout | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `0` | `0.023089` | `1.0` | `1.0` | `1.0` | Exact |
| `1` | `0.022599` | `1.0` | `1.0` | `1.0` | Exact |
| `2` | `0.023673` | `1.0` | `1.0` | `1.0` | Exact |
| `3` | `0.025511` | `1.0` | `1.0` | `1.0` | Exact |

This satisfies the current reproducibility requirement.

## Interpretation

- The old `32`-channel result was not the true minimal architecture for this task; it was the minimal architecture under a weaker and slower training regime.
- Once training moved to MLX and the budget increased to `64` mazes and `300` epochs, the minimal exact model dropped sharply to `12` hidden channels.
- The candidate `8/3/1` still fails under the stronger regime, so `12` is not a lucky result sandwiched between many viable smaller widths. At least one immediately smaller candidate still breaks.
- This is a stronger result than the earlier one because it combines:
  - larger training data
  - longer optimization
  - multiple exact retrains

## Current Recommendation

For Apple Silicon:

- use MLX for future minimal-model and recipe searches
- treat `12/3/1` as the current best minimal exact and reproducible `maze_exit` model

For portability:

- keep the PyTorch implementation as the reference backend for CPU and future CUDA-class machines

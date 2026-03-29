# MLX Minimal Model Revisit Plan

## Goal

Revisit the minimal `maze_exit` architecture using the new MLX backend, larger training data, and longer training runs than the earlier CPU/MPS sweep.

## Why Revisit

The previous minimal-model result in `docs/minimal-model-sweep.md` selected:

- hidden channels: `32`
- perception kernel: `3`
- update kernel: `1`

That result was based on relatively short runs, mostly `150` epochs, and small maze sets. MLX is much faster on this machine, so the old sweep likely under-tested smaller widths.

## Fixed Architectural Assumptions

This revisit keeps the convolution kernels fixed at:

- perception kernel: `3`
- update kernel: `1`

Reasoning:

- `1x1` perception is not sufficient for movement because the model cannot see neighboring cells.
- `3x3` is therefore the smallest perception kernel worth testing.
- `1x1` update is already the smallest update kernel and has already been shown to work.

Given that, the cleanest minimality search now is hidden-width only.

## Success Criteria

A candidate width is accepted only if it passes all of the following:

1. one-step evaluation:
   - `full_state_accuracy = 1.0`
   - `termination_accuracy = 1.0`
2. rollout evaluation on larger unseen mazes:
   - `30x30 exact_rollout_rate = 1.0`
   - `50x50 exact_rollout_rate = 1.0`
3. reproducibility:
   - the same recipe retrained from scratch on multiple seeds must pass all of the above

## Training Budget

Screening budget:

- task: `maze_exit`
- training grid: `9x9`
- mazes: `64`
- epochs: `300`
- batch size: `128`

Reproducibility budget:

- same architecture and task
- same `64` mazes
- same `300` epochs
- multiple seeds

This is already substantially stronger than the earlier `16`-maze, `150`-epoch baseline.

## Search Order

Candidates will be tested from smaller to larger width:

- `8`
- `12`
- `16`
- `20`
- `24`
- `28`
- `32`

The sweep stops at the first width that is exact **and** reproducible.

## Reproducibility Rule

The first candidate that passes screening will be retrained on seeds:

- `0`
- `1`
- `2`
- `3`

If any seed fails exactness, the candidate is rejected and the next larger width is tested.

## Deliverables

- a machine-readable sweep summary in `runs/`
- a markdown research report with candidate-by-candidate results
- updated tracking files
- a local git commit

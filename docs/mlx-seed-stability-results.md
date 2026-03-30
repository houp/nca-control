# MLX Seed-Stability Investigation Results

## Goal

Revisit the previously suspicious seed-`4` MLX run and determine whether it exposed a broad stability problem in the selected minimal `12/3/1` model.

## Final Experiment

Fixed model and task:

- task: `maze_exit`
- hidden channels: `12`
- perception kernel: `3`
- update kernel: `1`

Stronger training budget:

- backend: `MLX`
- training grid: `9x9`
- mazes: `96`
- epochs: `500`
- batch size: `128`
- seeds: `0,1,2,3,4,5,6,7`

Acceptance criteria per seed:

- one-step `full_state_accuracy = 1.0`
- one-step `termination_accuracy = 1.0`
- `30x30 exact_rollout_rate = 1.0`
- `50x50 exact_rollout_rate = 1.0`

Recorded run directory:

- `runs/mlx-seed-stability-96m-500e`

## Aggregate Result

The stronger recipe passed on all eight tested seeds:

- exact pass fraction: `8/8 = 1.0`
- final loss mean ± std: `0.021730 ± 0.000664`
- final loss median: `0.021758`
- minimum loss mean ± std: `0.021634 ± 0.000674`
- best seed/final loss: `7 / 0.020560`
- worst seed/final loss: `1 / 0.022619`

## Per-Seed Result Table

| Seed | Final Loss | Min Loss | One-Step | `30x30` | `50x50` | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `0.021669` | `0.021489` | `1.0` | `1.0` | `1.0` | Exact |
| `1` | `0.022619` | `0.022619` | `1.0` | `1.0` | `1.0` | Exact |
| `2` | `0.022498` | `0.022441` | `1.0` | `1.0` | `1.0` | Exact |
| `3` | `0.021578` | `0.021423` | `1.0` | `1.0` | `1.0` | Exact |
| `4` | `0.021864` | `0.021760` | `1.0` | `1.0` | `1.0` | Exact |
| `5` | `0.021847` | `0.021660` | `1.0` | `1.0` | `1.0` | Exact |
| `6` | `0.021202` | `0.021162` | `1.0` | `1.0` | `1.0` | Exact |
| `7` | `0.020560` | `0.020516` | `1.0` | `1.0` | `1.0` | Exact |

## Interpretation

The original exploratory seed-`4` failure from the maintenance audit appears to have been a recipe-level weakness rather than evidence that `12/3/1` is broadly unstable. Once the budget was increased from `64` mazes and `300` epochs to `96` mazes and `500` epochs, the previously suspicious seed `4` converged into the same exact regime as the other retrains.

This is a materially stronger result than the earlier reproducibility claim:

- the seed set doubled from `4` to `8`
- the training budget increased in both data and optimization length
- the aggregate statistics show a narrow final-loss spread across successful runs

## Additional Large-Grid Checks

Representative larger-grid checks were repeated after the sweep:

- seed `4`, `100x100`, `32` sequences, `200` steps: exact rollout rate `1.0`
- seed `4`, `200x200`, `16` sequences, `200` steps: exact rollout rate `1.0`
- seed `1`, `100x100`, `32` sequences, `200` steps: exact rollout rate `1.0`
- seed `1`, `200x200`, `16` sequences, `200` steps: exact rollout rate `1.0`

These follow-up checks suggest that the stronger recipe does not merely repair one-step fitting; it preserves the previously claimed larger-grid behavior on both the former suspicious seed and the worst-loss seed in the new sweep.

## Conclusion

For the current code base and the current Apple Silicon MLX backend:

- the selected `12/3/1` model still appears to be the minimal exact model
- the previously concerning seed-`4` behavior is not reproduced under a stronger recipe
- the minimal model should now be described as strongly reproducible under the tested `96`-maze, `500`-epoch protocol over seeds `0..7`

The claim is still empirical rather than absolute:

- it is evidence over a larger tested seed set, not a proof for all future seeds

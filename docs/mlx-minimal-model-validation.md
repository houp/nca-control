# MLX Minimal Model Validation

## Goal

Close the remaining gaps after the first strong MLX sweep:

- test the smaller boundary candidates just below `12/3/1`
- verify the selected `12/3/1` model on much larger grids than the original `30x30` and `50x50` acceptance checks

## Validation Protocol

The baseline strong MLX recipe stayed fixed:

- backend: `MLX`
- task: `maze_exit`
- training grid: `9x9`
- training mazes: `64`
- epochs: `300`
- batch size: `128`
- perception kernel: `3`
- update kernel: `1`

Exactness criteria stayed strict:

1. one-step:
   - `full_state_accuracy = 1.0`
   - `termination_accuracy = 1.0`
2. rollout:
   - `exact_rollout_rate = 1.0`
3. reproducibility:
   - all clean retrains in the gate must satisfy the same criteria

## Boundary Search Results

After the original sweep had shown `8` failing and `12` passing, the missing smaller widths were tested explicitly.

Tighter sweep outputs:

- `runs/mlx-minimal-sweep-tight/summary.json`
- `runs/mlx-minimal-sweep-tight/report.md`

### Candidate `9/3/1`

Result:

- screening one-step: `1.0`
- screening `30x30` rollout: `1.0`
- screening `50x50` rollout: `1.0`
- reproducibility: `1/2` passed
- final verdict: failed

Interpretation:

- `9/3/1` is capable of solving the task in at least one run
- it does not satisfy the reproducibility requirement

### Candidate `10/3/1`

Result from the strong MLX rerun:

- screening one-step: `0.862573`
- screening `30x30` rollout: `0.0`
- screening `50x50` rollout: `0.0`
- it never entered reproducibility testing

Interpretation:

- `10/3/1` is below the exactness threshold under the current strong recipe

### Candidate `11/3/1`

Result from the strong MLX rerun:

- screening one-step: exact
- screening `30x30` rollout: exact
- screening `50x50` rollout: exact
- reproducibility: `3/4` passed
- failing seed: `3`

Interpretation:

- `11/3/1` is close, but still not robust enough to be the selected model

## Large-Grid Validation For `12/3/1`

The selected `12/3/1` MLX checkpoints from `runs/mlx-minimal-sweep-strong/h12_repro_seed{0,1,2,3}` were evaluated on much larger mazes:

- `100x100`
- `200x200`

Rollout settings:

- `64` random sequences
- `200` steps per sequence
- `12800` total rollout steps per evaluation

### Results

| Seed | Grid | Exact Rollout Rate | Failed Sequences | Result |
| --- | --- | ---: | ---: | --- |
| `0` | `100x100` | `1.0` | `0` | Exact |
| `0` | `200x200` | `1.0` | `0` | Exact |
| `1` | `100x100` | `1.0` | `0` | Exact |
| `1` | `200x200` | `1.0` | `0` | Exact |
| `2` | `100x100` | `1.0` | `0` | Exact |
| `2` | `200x200` | `1.0` | `0` | Exact |
| `3` | `100x100` | `1.0` | `0` | Exact |
| `3` | `200x200` | `1.0` | `0` | Exact |

## Conclusion

Under the current strongest tested MLX recipe:

- `9/3/1` is not reproducible
- `10/3/1` is not exact
- `11/3/1` is not reproducible
- `12/3/1` remains the smallest exact and reproducible model found

The strengthened conclusion is:

- the current selected minimal model is still `hidden_channels=12`, `perception_kernel_size=3`, `update_kernel_size=1`
- this model is not only exact on the search grids, but also exact on `100x100` and `200x200` rollouts across all four clean retraining seeds tested

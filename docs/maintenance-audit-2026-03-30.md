# Maintenance Audit 2026-03-30

## Goal

Bring the repository back into a tight, internally consistent state:

- align documents and tracking with the current code base
- remove small sources of duplication and dead code
- re-run the selected minimal MLX model from a clean training run after the refactor

## Documentation Audit

The main drift found at the start of the audit was:

- `tracking/current-step.md` still described the deferred patch-local review instead of the active maintenance work
- `README.md` still contained an old "Step 2 is in progress" line
- `README.md` still described PyTorch MPS as the Apple Silicon training default even though MLX is now the preferred backend

Those were corrected.

## Code Cleanup

Two low-risk cleanup steps were applied:

1. documentation/tracking alignment
2. shared movement-logic refactor

The movement refactor introduced a single authoritative torch helper for action-index stepping and reused the same action-index constants in:

- `src/nca_control/dataset.py`
- `src/nca_control/evaluate.py`
- `src/nca_control/mlx_backend.py`

This removed repeated nested `where` logic and one unused constant while preserving behavior.

## Test Results

After the cleanup:

- focused tests: `30 passed`
- full suite: `62 passed, 1 skipped`

## Final Clean MLX Retraining

Selected model under test:

- task: `maze_exit`
- hidden channels: `12`
- perception kernel: `3`
- update kernel: `1`
- training grid: `9x9`
- mazes: `64`
- epochs: `300`
- batch size: `128`

### Exploratory Seed `4`

An exploratory fresh run on seed `4` was trained first:

- output: `runs/final_h12_p3_u1_seed4`
- final loss: `0.150796`

This run did not enter the known exact regime. That does not contradict the current documented claim, which is only that the recipe was verified as reproducible on seeds `0,1,2,3`.

### Clean Regression Seed `0`

The regression check then re-trained the same recipe on a known-good seed:

- output: `runs/final_h12_p3_u1_seed0_rerun`
- final loss: `0.023089`
- training throughput: `95681.87 samples/s`
- total train time: `42.986 s`

## Final Verification Of The Fresh Seed `0` Checkpoint

One-step evaluation on `runs/final_h12_p3_u1_seed0_rerun/checkpoint_mlx`:

- `full_state_accuracy = 1.0`
- `termination_accuracy = 1.0`
- `exit_fill_exact_accuracy = 1.0`
- `active_presence_accuracy = 1.0`
- `active_position_accuracy = 1.0`

Rollout evaluation on the same fresh checkpoint:

| Grid | Sequences | Steps | Exact Rollout Rate |
| --- | ---: | ---: | ---: |
| `50x50` | `32` | `200` | `1.0` |
| `100x100` | `32` | `200` | `1.0` |
| `200x200` | `16` | `200` | `1.0` |

## Conclusion

The repository is back in sync after the maintenance pass:

- documents now reflect the current implementation and backend recommendation
- the movement logic is simpler and less duplicated
- the full automated suite still passes
- the selected MLX `12/3/1` model still reproduces correctly on the current code when re-trained on a known-good seed

The remaining caveat is unchanged:

- the current minimal-model claim is evidence-backed for the tested seed set and validation runs
- it is not a guarantee that every arbitrary new seed will converge without fail


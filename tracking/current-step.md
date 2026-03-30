# Current Step

## Step ID

32

## Title

Patch-local training prototype review

## Scope

- review the earlier idea of training the maze controller from local `3x3` patches instead of full mazes
- determine whether a patch-local objective can preserve the exit-lockout and fill semantics
- decide whether a prototype is worth implementing next

## Exit Criteria

- the patch-local idea is reviewed against the current `maze_exit` task semantics
- feasibility, likely benefits, and likely failure modes are recorded in markdown
- a recommendation is made on whether to prototype it
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 31 completed the maintenance pass:

- README and tracking were aligned with the current code base
- shared movement logic was simplified across dataset, evaluation, and MLX code paths
- full test suite passed: `62 passed, 1 skipped`
- a fresh MLX `12/3/1` seed `0` rerun reproduced exact one-step behavior and exact `50x50`, `100x100`, and `200x200` rollouts

The deferred next item is again the patch-local review. The current selected minimal model remains MLX `12/3/1`:

- `9/3/1` failed reproducibility
- `10/3/1` failed screening
- `11/3/1` failed reproducibility
- `12/3/1` remained exact on `100x100` and `200x200` rollouts across seeds `0,1,2,3`

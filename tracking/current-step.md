# Current Step

## Step ID

29

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

Step 27 completed the first strong MLX minimal-model search and selected `12/3/1`.

Step 28 then tightened the boundary search and extended the scale checks:

- `9/3/1` failed reproducibility
- `10/3/1` failed screening
- `11/3/1` failed reproducibility
- the selected `12/3/1` model remained exact on `100x100` and `200x200` rollouts across seeds `0,1,2,3`

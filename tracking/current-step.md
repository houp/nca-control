# Current Step

## Step ID

28

## Title

Extended MLX minimal-model validation

## Scope

- stress-test the selected MLX `12/3/1` model beyond the search budget
- evaluate longer rollouts and possibly larger grids than the current `30x30` and `50x50`, `50`-step checks
- confirm the selected model remains exact under stronger post-selection validation

## Exit Criteria

- longer-horizon rollout checks are run on the selected `12/3/1` model
- the model remains exact under stronger validation than the selection sweep
- findings are recorded in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 25 changed the hardware recommendation: MLX is now the preferred Apple Silicon backend because it is much faster than both PyTorch CPU and PyTorch MPS on the current `9x9` best-model workload while still preserving exact one-step and rollout behavior.

Step 26 added the MLX minimal-model search protocol and automated sweep driver.

Step 27 completed that search. Under the stronger MLX regime (`64` mazes, `300` epochs, `batch_size=128`), the first exact and reproducible candidate was `hidden=12`, `perception=3`, `update=1`. The smaller `8/3/1` model still failed, while `12/3/1` passed exact one-step checks and exact `30x30` / `50x50` rollouts on four clean retrains with seeds `0,1,2,3`.

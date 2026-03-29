# Current Step

## Step ID

27

## Title

MLX reproducible minimal-model search

## Scope

- revisit the minimal `maze_exit` model using the MLX backend
- use a stronger training budget than the earlier 150-epoch sweep
- require exact larger-grid behavior and multi-seed reproducibility before accepting a candidate

## Exit Criteria

- ascending hidden-width candidates are tested with the stronger MLX budget
- the first accepted candidate is exact on one-step and larger-grid rollout checks
- the selected candidate reproduces across multiple clean retrains
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 25 changed the hardware recommendation: MLX is now the preferred Apple Silicon backend because it is much faster than both PyTorch CPU and PyTorch MPS on the current `9x9` best-model workload while still preserving exact one-step and rollout behavior.

Step 26 created the new MLX minimal-model search protocol and an automated sweep driver. The search fixes the kernels at the smallest plausible setting (`perception=3`, `update=1`) and revisits hidden width with a much stronger training budget (`64` mazes, `300` epochs, `batch_size=128`) plus a four-seed reproducibility gate.

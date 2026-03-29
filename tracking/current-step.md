# Current Step

## Step ID

16

## Title

Broader maze sweep and reporting

## Scope

- run maze experiments on larger grids and larger maze sets
- compare one-step and rollout behavior across scales
- capture concise findings in markdown

## Exit Criteria

- at least one broader maze sweep is recorded
- findings are captured in markdown tracking
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 15 optimized the training path by removing Python per-sample maze tensor construction from the hot loop. Maze batches are now materialized vectorially, the model uses channels-last on GPU devices, and training reports throughput metrics. A short `30x30` benchmark on MPS (`16` mazes, `2` epochs, `batch_size=256`, `hidden_channels=64`) completed at about `2207.78 samples/s` over `28.336 s`.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

# Current Step

## Step ID

20

## Title

Patch-local training prototype

## Scope

- prototype a local `3x3` patch-based training task for the maze-control rule
- compare its training cost against the current full-maze path
- check whether it preserves exact cross-grid rollout behavior

## Exit Criteria

- a minimal patch-local training path exists
- automated tests cover patch dataset generation and evaluation
- a trained checkpoint is compared against the current maze baseline
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 19 established that the current fully convolutional `maze_exit` model already generalizes from `9x9` training mazes to much larger mazes in rollout evaluation, including `50x50`. The patch-local idea remains interesting mainly as a training-speed optimization and a cleaner locality experiment.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

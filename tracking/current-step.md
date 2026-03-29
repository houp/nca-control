# Current Step

## Step ID

23

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

Step 22 reran the architecture search from scratch after the `runs/` cleanup and found the first exact model at `hidden_channels=32`, `perception_kernel_size=3`, `update_kernel_size=1`. That model remained exact on both `30x30` and `50x50` rollout checks.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

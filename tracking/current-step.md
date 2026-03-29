# Current Step

## Step ID

21

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

Step 20 changed the browser visualizer from keypress-driven stepping to a fixed simulation clock. When no input is queued, the visualizer now applies `none` automatically, so terminal exit-fill spread continues without extra keypresses. The training stack already included `Action.NONE`, so no dataset or model change was needed for this slice.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

# Current Step

## Step ID

15

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

Step 14 updated the browser visualizer so it now preserves and renders walls correctly. The maze-aware visualizer starts successfully against `runs/maze9_smoke/checkpoint.pt` and the full automated suite remains green.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

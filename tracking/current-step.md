# Current Step

## Step ID

19

## Title

Maze exit scaling and MPS validation

## Scope

- repeat the `maze_exit` experiment on larger grids
- validate MPS-backed training throughput for the exit-aware task
- compare rollout exactness across grid sizes and model widths

## Exit Criteria

- at least one larger-grid `maze_exit` checkpoint is trained
- one-step and rollout metrics are recorded for the larger-grid run
- MPS throughput is measured outside the sandbox if needed
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 18 completed the learned exit-aware training slice. The model now learns wall-respecting control, exit detection, and terminal lockout, while decoded post-terminal fill expansion is kept deterministic to preserve exact gameplay semantics during rollout and visualization.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

# Current Step

## Step ID

18

## Title

Maze exit-aware training and evaluation

## Scope

- extend dataset encoding with exit-state information
- train the NCA to stop control at the exit and model gradual end-state spread
- verify one-step and rollout behavior on exit-aware mazes

## Exit Criteria

- maze-aware training data includes exit semantics
- trained checkpoints respect wall and exit behavior
- automated tests cover exit-aware data and evaluation
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 17 completed the browser/session slice for exit-aware visualization: reset now regenerates a new maze, the session snapshot includes terminal and exit-fill state, and the browser renderer draws the goal-state fill correctly. The next slice teaches the NCA to model those exit dynamics directly.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

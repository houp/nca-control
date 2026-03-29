# Current Step

## Step ID

8

## Title

Interactive visual verification app

## Scope

- add a keyboard-driven visualizer for the learned model
- show the deterministic reference and learned model side by side
- keep the app simple and local so it can be run directly on macOS
- add tests for non-UI control logic where practical

## Exit Criteria

- a local interactive app can be launched
- arrow keys and no-op controls update the grids
- the app helps verify learned vs reference behavior visually
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 7 delivered a materially improved model: on the same `6x6`, `100`-epoch evaluation used in Step 6, the refined setup reached `argmax_accuracy=1.0`, `mse~=7.9e-7`, and `mean_predicted_max~=0.995`. Step 8 will expose that behavior through an interactive visual tool.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

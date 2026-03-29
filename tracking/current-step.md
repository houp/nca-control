# Current Step

## Step ID

13

## Title

Maze-aware dataset and training pipeline

## Scope

- add a wall channel to encoded model inputs
- generate supervised training data from mazes
- train and evaluate a maze-aware checkpoint
- verify that learned one-step transitions respect walls

## Exit Criteria

- maze training data is generated correctly
- training and evaluation work on maze layouts
- automated tests cover the new data and inference path
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 12 locked down the deterministic maze target. Step 13 teaches the model about walls by adding wall-aware inputs and maze-generated training data.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

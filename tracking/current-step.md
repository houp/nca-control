# Current Step

## Step ID

5

## Title

One-step training and checkpointed inference

## Scope

- add a minimal supervised training loop
- train on deterministic one-step transitions
- save checkpoints and basic metrics
- add an inference script that loads a checkpoint and predicts the next state

## Exit Criteria

- training script runs end-to-end on the deterministic dataset
- checkpoint save and load work
- inference script can evaluate a scripted state/action pair
- automated tests cover training utilities where practical
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 1 locked down movement semantics. Step 2 added text-based simulation and CLI verification. Step 3 produced deterministic supervised training data. Step 4 introduced the first trainable NCA model. Step 5 will add the first optimization path and checkpointed inference flow.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

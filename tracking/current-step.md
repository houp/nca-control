# Current Step

## Step ID

4

## Title

Minimal learned NCA model

## Scope

- implement the smallest viable controllable NCA in PyTorch
- define a forward pass over state and control channels
- add tests for tensor shapes and bounded output behavior
- keep the model simple enough to debug before training code exists

## Exit Criteria

- model forward pass works on batched tensors
- output shape matches target grid shape
- automated tests cover the model interface
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 1 locked down movement semantics. Step 2 added text-based simulation and CLI verification. Step 3 produced deterministic supervised training data. Step 4 introduces the first trainable NCA model without adding optimization code yet.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

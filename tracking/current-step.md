# Current Step

## Step ID

6

## Title

Training quality evaluation and rollout metrics

## Scope

- evaluate whether the trained model actually learns correct control behavior
- add exactness metrics for next-state prediction
- add a repeatable evaluation command or utility
- use the results to guide the next model/training refinement step

## Exit Criteria

- evaluation reports next-state correctness metrics
- automated tests cover metric computation where practical
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 1 locked down movement semantics. Step 2 added text-based simulation and CLI verification. Step 3 produced deterministic supervised training data. Step 4 introduced the first trainable NCA model. Step 5 added the first training and inference scripts. Step 6 will measure whether that model is actually learning the desired behavior accurately enough.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

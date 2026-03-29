# Current Step

## Step ID

7

## Title

Exactness-oriented model and loss refinement

## Scope

- improve the model and/or training objective for exact single-cell control
- bias the output toward one active location with preserved value
- re-evaluate after training
- keep the changes narrowly targeted at correctness

## Exit Criteria

- refined training/evaluation improves next-state exactness materially
- automated tests cover the changed model or decoding behavior
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 6 showed that the initial one-step setup is not sufficient: on a `6x6` training/evaluation run, argmax accuracy reached only about `86.7%`, with a mean predicted peak around `0.42`. Step 7 will respond to that evidence directly.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

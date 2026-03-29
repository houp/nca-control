# Current Step

## Step ID

11

## Title

Broader rollout and size sweep

## Scope

- extend rollout testing across more grid sizes and training settings
- identify limits where one-step exactness stops implying rollout stability
- keep the process scripted and reproducible

## Exit Criteria

- at least one broader sweep is added beyond the current 20x20 success case
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 10 added scripted rollout evaluation and validated the current refined `20x20` checkpoint over `256` random sequences of `1000` steps each with `0` failures. The next useful step is a broader sweep rather than a single-size spot check.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

# Current Step

## Step ID

24

## Title

MPS reproducibility hardening

## Scope

- harden the selected `32/3/1` maze-exit recipe so that fresh MPS retrains are exact consistently, not just usually
- compare small training-budget adjustments without changing the selected architecture
- keep cross-grid rollout exactness on larger mazes

## Exit Criteria

- multiple fresh MPS retrains are run
- the tightened recipe reaches exact one-step and rollout results consistently
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 23 recorded the first MPS replication check. The selected `32/3/1` architecture preserved exact `30x30` and `50x50` rollouts across three fresh MPS runs, but one of the three runs missed strict one-step full-state exactness slightly (`0.998837`). The next slice should harden the recipe, not the architecture.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

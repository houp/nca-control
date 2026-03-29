# Current Step

## Step ID

26

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

Step 24 delivered training progress artifacts and a device-throughput investigation instead. That clarified that the current MPS path is not fundamentally broken: it is slower than CPU on tiny `9x9` workloads, but reaches rough parity on `30x30` once the convolution workload is large enough. The reproducibility-hardening work remains pending.

Step 25 delivered an MLX backend instead. That result materially changes the Apple Silicon recommendation: on the current `9x9` best-model workload, MLX was much faster than both PyTorch CPU and PyTorch MPS while still reaching exact one-step and rollout behavior. The pending MPS hardening work is now lower priority than deciding whether MLX should become the default training backend on Apple machines.

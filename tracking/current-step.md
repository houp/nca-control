# Current Step

## Step ID

10

## Title

Multi-step rollout evaluation

## Scope

- evaluate repeated control over longer horizons
- compare learned rollout behavior against the deterministic reference
- detect drift, duplication, or stability failures over sequences
- keep the tooling compatible with the existing checkpoint workflow

## Exit Criteria

- rollout metrics exist beyond one-step evaluation
- longer-horizon behavior can be checked automatically
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 9 replaced the original Tk-based visualizer with a browser-based local server because the active Python 3.13 environment on this machine does not provide `_tkinter`. The replacement visualizer now starts successfully when run unsandboxed and serves a local page for manual verification.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

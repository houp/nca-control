# Current Step

## Step ID

9

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

Step 8 added a side-by-side interactive visualizer. In this headless environment I verified the non-UI control logic and script compilation, but the actual Tk window still requires a manual launch on macOS for final visual confirmation.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

# Current Step

## Step ID

39

## Title

Python 3.14 Target Investigation

## Scope

- determine why the repository currently targets Python `3.13` rather than the locally installed Python `3.14`
- test whether the current dependency set and runtime paths actually support `3.14`
- if feasible, promote `3.14` to the primary interpreter target while preserving an occasional `3.13` re-test path

## Exit Criteria

- a written investigation plan and work-in-progress log are added to `docs/`
- the real blocker for Python `3.14` is identified by direct compatibility testing
- if `3.14` is viable, repo defaults and documentation are updated to make it the main target while keeping `3.13` as an optional re-test path
- if `3.14` is not viable, the blocker is documented clearly in code and docs
- tracking is updated
- local git commit is created

## Notes

Step 38 strengthened the reproducibility evidence for the selected `12/3/1` MLX model: the stronger `96`-maze, `500`-epoch recipe passed on seeds `0..7`, and the formerly suspicious seed `4` remained exact on `100x100` and `200x200` rollout checks.

The completed verification now shows that Python `3.14` itself is healthy locally, the current dependency set installs under `3.14`, the default repository `.venv` has been refreshed to Python `3.14.3`, `uv run pytest` passes on the new default environment, and an unsandboxed MLX smoke training run succeeds under `3.14`.

The next deferred research question after this portability step remains whether patch-local supervision can retain the current guarantees while reducing training cost further.

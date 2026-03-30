# Python 3.14 Target Investigation Plan

## Goal

Determine whether the repository can move from Python `3.13` to Python `3.14` as the main interpreter target without breaking the currently verified research workflows.

## Immediate Observation

At the start of this step:

- `.python-version` points to `3.13`
- `pyproject.toml` declares `requires-python = ">=3.13,<3.14"`
- `uv.lock` is pinned to `==3.13.*`
- `/opt/homebrew/bin/python3.14` is installed locally and starts normally

This means the repo is intentionally excluding `3.14`, but the reason is not yet proven.

## Main Hypothesis

The most likely reason is stale repository pinning rather than a broken local interpreter. MLX is the main suspected dependency risk, so that path needs explicit testing before the default target changes.

## Investigation Steps

1. verify that the installed `python3.14` interpreter itself is healthy
2. test dependency resolution and environment creation under `3.14`
3. identify which package, if any, blocks `3.14`
4. if a compatible package set exists, update the repo to support `3.14`
5. if no compatible package set exists, document the blocker and keep `3.13` as the justified default

## Success Criteria

The step is successful if one of the following is established clearly:

### Outcome A: `3.14` support is viable

- the repo resolves and installs under Python `3.14`
- the key automated tests pass
- `pyproject.toml`, `.python-version`, `uv.lock`, and documentation are updated so `3.14` is the primary target
- `3.13` is explicitly documented as an occasional re-test path

### Outcome B: `3.14` support is not yet viable

- the blocking package or packaging gap is identified directly
- the reason is documented in project docs
- current `3.13` targeting remains justified and explicit

## Constraints

- do not break the verified macOS MLX workflow
- keep the PyTorch paths working
- update AGENTS, README, and the report/presentation if the supported Python target changes materially

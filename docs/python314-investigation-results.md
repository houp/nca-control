# Python 3.14 Target Investigation Results

## Summary

Python `3.14` is viable for this repository and should be the default interpreter target.

## Verified Findings

- `/opt/homebrew/bin/python3.14` starts normally and reports `Python 3.14.3`
- `uv` can create a Python `3.14` environment for this project
- the current dependency set installs under Python `3.14`
- the full automated test suite passes under Python `3.14`
- an unsandboxed MLX smoke training run completes under Python `3.14`
- the default project `.venv` now runs on Python `3.14.3`
- `uv run pytest` on the refreshed default environment reports `67 passed, 1 skipped`

## Root Cause

The repository was still on Python `3.13` because the project configuration had not been updated:

- `.python-version` pointed to `3.13`
- `pyproject.toml` excluded `3.14`
- `uv.lock` was generated for `3.13`

This was a policy/configuration issue, not an interpreter or package-runtime failure.

## Decision

- Python `3.14` becomes the primary interpreter target
- Python `3.13` remains an occasional secondary re-test path after major changes

## Follow-Up

Python `3.14` should remain the main target in repository configuration and documentation. Python `3.13` can still be used for occasional portability re-checks after substantial changes.

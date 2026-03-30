# Python 3.14 Target Investigation WIP

## Status

Completed.

## Confirmed Facts So Far

- the local Python `3.14` interpreter exists at `/opt/homebrew/bin/python3.14`
- it reports `Python 3.14.3`
- the repository currently excludes `3.14` explicitly in both `pyproject.toml` and `uv.lock`
- the current virtual environment is on Python `3.13.12`
- a temporary Python `3.14` environment installs `mlx`, `numpy`, `torch`, `typer`, and `pytest` successfully
- the full test suite passes under Python `3.14`
- an unsandboxed MLX smoke training run succeeds under Python `3.14`

## Leading Conclusion

The repository is still on Python `3.13` because of stale project pinning, not because the current dependency set is incompatible with Python `3.14`.

## Next Checks

1. repo metadata and default interpreter selection were updated to prefer `3.14`
2. the lockfile and default `.venv` were regenerated under `3.14`
3. the repository test suite and an MLX smoke training run were rerun on the new default environment

# Current Step

## Step ID

1

## Title

Planning, tracking, and deterministic movement baseline

## Scope

- create project documentation and tracking files
- scaffold the Python package
- implement the deterministic reference transition rule
- add baseline tests for movement invariants

## Exit Criteria

- repository documentation exists
- deterministic movement code is implemented
- automated tests pass
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

This step intentionally avoids learned NCA training. The objective is to freeze the target dynamics before training infrastructure is introduced.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

## Verification

- `.venv/bin/pytest` -> `5 passed`
- `.venv/bin/python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"` -> `2.11.0 / True / True` when run unsandboxed

# Current Step

## Step ID

3

## Title

Supervised transition data generation

## Scope

- generate one-step supervised examples from the deterministic baseline
- create tensor-friendly encodings for state and action
- add tests for dataset shapes and semantic correctness
- keep the implementation simple enough to inspect manually

## Exit Criteria

- dataset generation works programmatically
- generated samples preserve deterministic transition semantics
- automated tests cover shapes and labels
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 1 locked down movement semantics. Step 2 added text-based simulation and CLI verification. Step 3 will turn the deterministic rule into supervised training data for the first learned NCA model.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

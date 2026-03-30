# Current Step

## Step ID

45

## Title

Activated venv Python Invocation Policy

## Scope

- verify that activating `.venv` and then using plain `python` works correctly on the local machine
- update documentation to prefer `source .venv/bin/activate` followed by `python ...`
- add a standing rule in `AGENTS.md` so future terminal commands follow the same convention

## Exit Criteria

- local checks confirm that the activated-venv `python` path works for the relevant entrypoints
- `README.md` prefers `source .venv/bin/activate` and `python ...` over hard-coded interpreter paths
- `AGENTS.md` records the convention for future work
- tracking is updated
- local git commit is created

## Notes

The local environment already supports the cleaner workflow: activate `.venv` once, then run `python ...`. The documentation should reflect that, and the repository rules should preserve the same convention in future edits.

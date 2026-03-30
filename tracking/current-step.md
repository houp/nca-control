# Current Step

## Step ID

44

## Title

README uv Sync Simplification

## Scope

- remove the optional `UV_CACHE_DIR` override from the README quick-start command
- keep the quick-start environment setup as simple as possible
- document the default `uv` workflow rather than an implementation detail of earlier runs

## Exit Criteria

- the README quick-start setup uses plain `uv sync --python 3.14`
- the quick-start command remains consistent with the current project target
- tracking is updated
- local git commit is created

## Notes

The remaining issue is not correctness but simplicity: the quick-start command still shows an unnecessary cache override. The README should prefer the default `uv` behavior unless there is a real project-level need for something more specific.

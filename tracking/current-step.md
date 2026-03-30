# Current Step

## Step ID

43

## Title

Markdown Path Cleanup

## Scope

- remove hard-coded user-specific absolute paths from Markdown documentation
- keep the quick-start instructions portable across clones and machines
- verify that the Markdown documentation set no longer contains those user-specific paths

## Exit Criteria

- no Markdown documentation file contains user-specific absolute paths
- the README quick-start command still communicates the intended `uv` setup clearly
- tracking is updated
- local git commit is created

## Notes

The current issue is documentation portability rather than runtime behavior. Markdown docs should not embed a specific local home-directory path when a repo-relative or generic command is sufficient.

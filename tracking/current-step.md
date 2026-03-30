# Current Step

## Step ID

32

## Title

Academic Technical Report

## Scope

- write a short academic-style technical report on the current project state
- keep every claim consistent with the code base and recorded experimental artifacts
- typeset the report with XeTeX and compile a PDF

## Exit Criteria

- a report plan is written in markdown
- the LaTeX source and bibliography are added to the repository
- figures/tables are generated from verified data sources
- the XeTeX PDF compiles successfully
- tracking is updated
- local git commit is created

## Notes

Step 31 completed the maintenance pass and re-verified the selected MLX model on the current code.

The report should center the current selected minimal model:

- `9/3/1` failed reproducibility
- `10/3/1` failed screening
- `11/3/1` failed reproducibility
- `12/3/1` remained exact on `100x100` and `200x200` rollouts across seeds `0,1,2,3`

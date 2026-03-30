# Technical Presentation Plan

## Goal

Produce a short academic-style presentation, consistent with the current technical report and verified experimental artifacts, using LaTeX and a standard presentation package.

## Source of Truth

The presentation must stay aligned with:

- `report/technical_report.tex`
- `report/data/*.csv`
- tracked documentation in `README.md` and `tracking/`
- verified MLX evaluation outputs for the selected `12/3/1` checkpoint

## Scope

The deck should summarize:

1. project motivation and problem definition
2. deterministic environment and learned update rule
3. implementation stack and backend choice
4. experimental protocol
5. minimal-model search result
6. current generalization and reproducibility findings
7. limitations and next steps

## Slide Plan

1. title and context
2. task definition and environment semantics
3. implementation pipeline
4. software stack and evaluation workflow
5. backend performance comparison
6. minimal-model search boundary
7. fresh verification and large-grid generalization
8. conclusions and open questions

## Visual Plan

Use simple data-backed visuals:

- reuse the learned-transition pipeline diagram in a presentation-friendly form
- include the backend benchmark bar chart
- include the minimal-model boundary chart
- include a compact summary table for final verified results

## Style Constraints

- professional academic beamer style
- minimal color palette
- readable slides without dense paragraphs
- direct consistency with the technical report wording and claims

## Verification Plan

1. compile with XeTeX
2. inspect for LaTeX warnings and layout issues
3. keep only durable presentation artifacts in git
4. update tracking and commit the step

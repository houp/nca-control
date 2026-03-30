# Current Step

## Step ID

33

## Title

Patch-Local Training Prototype

## Scope

- evaluate the deferred patch-local training idea against the current full-state training pipeline
- test whether local `3x3` supervision can preserve exactness, reproducibility, and cross-grid generalization
- compare training cost and model size implications under the MLX backend

## Exit Criteria

- a concrete patch-local training formulation is specified in code and documentation
- at least one MLX-trained patch-local model is evaluated on one-step and rollout metrics
- the result is compared directly against the selected `12/3/1` full-state baseline
- tracking is updated
- local git commit is created

## Notes

Step 32 completed a short academic report at `report/technical_report.tex` and `report/technical_report.pdf`.

The next deferred research question is whether patch-local supervision can retain the current guarantees while reducing training cost further.

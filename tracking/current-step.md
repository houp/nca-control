# Current Step

## Step ID

39

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

Step 38 strengthened the reproducibility evidence for the selected `12/3/1` MLX model: the stronger `96`-maze, `500`-epoch recipe passed on seeds `0..7`, and the formerly suspicious seed `4` remained exact on `100x100` and `200x200` rollout checks.

The next deferred research question is whether patch-local supervision can retain the current guarantees while reducing training cost further.

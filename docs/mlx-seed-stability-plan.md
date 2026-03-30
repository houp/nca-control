# MLX Seed-Stability Investigation Plan

## Goal

Determine whether the previously observed exploratory seed-`4` failure is:

- a minor optimization outlier that disappears under a stronger training recipe, or
- evidence that the current minimal `maze_exit` model `12/3/1` is only conditionally reproducible.

## Why This Step Is Needed

The current report contains two facts that are both true but not yet fully reconciled:

1. the selected `12/3/1` MLX model was exact on the original reproducibility seeds `0,1,2,3`
2. a later exploratory retrain on seed `4` plateaued at a much higher loss and did not enter the exact regime

That is enough to justify a larger experiment. The present step is intended to replace the current anecdotal seed-`4` evidence with a broader and better-instrumented stability study.

## Fixed Model And Task

The investigation keeps the current selected architecture fixed:

- task: `maze_exit`
- hidden channels: `12`
- perception kernel: `3`
- update kernel: `1`

The question here is recipe stability, not architecture search.

## Stronger Training Budget

The new seed-stability sweep should exceed the maintenance rerun budget in two dimensions:

- more epochs than `300`
- more mazes than `64`

Initial target budget for the first large sweep:

- backend: `MLX`
- training grid: `9x9`
- mazes: `96`
- epochs: `500`
- batch size: `128`

If that still shows repeated failures, the follow-up budget may increase again to:

- mazes: `128`
- epochs: `600`

The experiment should stop as soon as the evidence is strong enough to support one of the two conclusions:

- the failures largely disappear and the recipe is stable enough, or
- the failure rate remains material and the current minimal recipe should be described as only partially stable.

## Seed Protocol

The study should cover more than the original `0..3` set. The first pass should use:

- screening seeds: `0..7`

If needed, extend to:

- confirmation seeds: `8..15`

## Metrics To Record

Per run:

- final loss
- minimum loss
- final one-step metrics
- rollout exactness on `30x30` and `50x50`
- whether the run reached the exact regime

Aggregate:

- exact pass fraction
- mean and standard deviation of final loss
- mean and standard deviation of minimum loss
- median final loss
- best/worst run identifiers

## Reporting Requirement

The updated report and slides should replace the current two-curve seed comparison with:

- multiple training traces
- a compact statistical summary over seeds
- a clear statement about whether the selected minimal model is robust or fragile under fresh retrains

## Deliverables

- work-in-progress markdown log for the experiment
- automation/script support for the multi-seed MLX sweep
- machine-readable summary files under `runs/`
- updated report data export
- refreshed technical report and presentation

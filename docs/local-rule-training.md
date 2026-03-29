# Local-Rule Training Review

## Question

Can we avoid training on large full mazes by training only on local `3x3` neighborhoods and then apply the learned convolutional rule to arbitrarily large grids?

## Short Answer

Yes. The idea is technically sound for this project.

For the current maze-control and maze-exit rules, the one-step transition is local:

- movement depends on the active cell, local walls, and the requested action
- reaching the exit depends on local contact with the exit cell
- post-terminal fill expansion is also local once the terminal state is known

That means a convolutional model can, in principle, learn the rule from local neighborhoods and then reuse the same weights on larger grids.

## What We Verified

The current `maze_exit` model is already much closer to that ideal than it may look from the training scripts:

- the learned path uses a `3x3` circular convolution followed by `1x1` updates
- the learned one-step transition therefore has a local receptive field
- the weights are shared across space, so inference is naturally grid-size agnostic

Using the verified checkpoint `runs/maze_exit_smoke_long/checkpoint.pt`, trained on `9x9`, we checked rollout behavior on larger unseen mazes:

- `15x15`: exact rollout match
- `30x30`: exact rollout match
- `50x50`, `64` random sequences, `500` steps: exact rollout rate `1.0`

## Interpretation

For this rule family, training on very large mazes is **not required** to obtain a model that works on large mazes.

The current full-maze training path already learns a local rule that transfers cleanly to larger grids, provided that:

- the model remains fully convolutional
- decoding preserves the intended semantics
- the task itself stays local

## Would Patch-Local Training Still Help?

Possibly, but mainly for engineering reasons:

- faster training
- smaller synthetic datasets
- a cleaner proof that the learned rule is local by construction
- less dependence on maze generation during training

## Risks of a Pure Patch Dataset

- local datasets must encode only legal state combinations, or the model may learn irrelevant cases
- boundary and terminal semantics still need careful encoding
- some tasks that look local can hide global bookkeeping requirements
- the current project already uses deterministic decoding for exact gameplay semantics, so part of the benefit is already realized

## Recommended Direction

Treat patch-local training as an optimization experiment, not a prerequisite for size generalization.

The evidence so far says:

1. the current full-maze exit-aware model already generalizes across grid sizes
2. the patch-local idea is still worth prototyping if training speed becomes the main bottleneck
3. any patch-local prototype should be compared against the current `maze_exit` baseline on:
   - training wall-clock time
   - one-step exactness
   - long-rollout exactness
   - cross-grid generalization

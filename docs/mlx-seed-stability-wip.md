# MLX Seed-Stability Investigation WIP

## Status

Completed.

## Fixed Hypothesis

The selected `12/3/1` MLX model may still be stable under fresh retrains if the training budget is strengthened beyond the earlier `64`-maze, `300`-epoch configuration.

## Completed Tasks

1. add repeatable experiment tooling for multi-seed MLX retraining and evaluation
2. run the first stronger sweep on seeds `0..7`
3. summarize exactness rate and loss statistics
4. decide whether a second, larger sweep is needed

## Outcome

The first stronger sweep was sufficient:

- recipe: `96` mazes, `500` epochs, `batch_size=128`
- seeds: `0..7`
- exact pass fraction: `8/8 = 1.0`
- former exploratory failure seed `4` became exact again under the stronger budget

Representative larger-grid checks also passed:

- seed `4`: exact on `100x100` and `200x200`
- worst-loss seed `1`: exact on `100x100` and `200x200`

## Notes

- keep the architecture fixed during this step
- focus on seed stability, not on finding a smaller model
- update the report and slides only after the new aggregate results are verified

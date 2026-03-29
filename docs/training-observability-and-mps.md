# Training Observability and MPS Throughput

## Goal

Make training runs inspectable while they are still executing, then measure whether the current Apple Silicon MPS path is underperforming because of implementation issues or because the current workloads are too small.

## Training Progress Artifacts

Each training run now writes the following files into its run directory while training is still in progress:

- `progress.jsonl`
- `latest_status.json`
- `metrics.json`

`progress.jsonl` receives one JSON record per epoch. Each record includes:

- `epoch`
- `epochs_total`
- `loss`
- `epoch_time_sec`
- `elapsed_sec`
- `epoch_samples_per_second`
- `running_samples_per_second`
- `device`
- `num_samples`

`latest_status.json` is overwritten every epoch and is suitable for quick polling from another process. It starts with `status = "running"` and ends with `status = "completed"`.

The CLI training script now also prints one progress line per epoch so a foreground training run is no longer silent.

## Implementation Change Relevant To MPS

For maze-based tasks, the dataset previously copied several static tensor banks to the target device on every batch:

- blocked-cell grids
- action-channel grids
- exit-fill tensors
- exit-cell metadata
- fill-stage lengths

That repeated transfer was unnecessary. The dataset now caches those tensors per device and reuses them across batches.

This does not remove all CPU-side work. Batch metadata, shuffling, and some control logic still originate on CPU, so very small workloads still pay non-trivial launch and transfer overhead on MPS.

## Benchmarks

Benchmarks were run after the device-tensor caching change using the current selected architecture:

- hidden channels: `32`
- perception kernel: `3`
- update kernel: `1`

### Small Workload: `9x9`

Command shape:

- task: `maze_exit`
- mazes: `16`
- epochs: `40`
- batch size: `64`

| Device | Samples / Second | Total Train Time (s) | Final Loss |
| --- | ---: | ---: | ---: |
| `cpu` | `4752.23` | `28.955` | `0.971998` |
| `mps` | `2601.95` | `52.883` | `0.972063` |

Interpretation:

- MPS is substantially slower on the small workload.
- The gap remains even after removing repeated static tensor transfers.
- This indicates the workload is too small to amortize MPS launch and synchronization overhead.

### Larger Workload: `30x30`

Command shape:

- task: `maze_exit`
- mazes: `16`
- epochs: `10`
- batch size: `256`

| Device | Samples / Second | Total Train Time (s) | Final Loss |
| --- | ---: | ---: | ---: |
| `cpu` | `2167.36` | `159.595` | `5.987938` |
| `mps` | `2179.77` | `158.687` | `5.987938` |

Interpretation:

- At `30x30`, MPS reaches rough parity with CPU and is slightly faster in this probe.
- The implementation is therefore not fundamentally broken on MPS.
- The main issue is workload scale: the current minimal-model search configuration is too small to benefit from GPU execution.

## Practical Guidance

- Use `cpu` for tiny architecture sweeps and small-grid probes such as `9x9` with modest batch sizes.
- Use `mps` for larger grids or larger batches where the convolution workload is heavier.
- When comparing devices, inspect `progress.jsonl` to see whether the first epoch is a warmup outlier before judging the steady-state speed.

## Next Optimization Directions

The current changes improve observability and remove one obvious repeated transfer, but further MPS-focused work is still possible:

1. Keep more batch construction on-device instead of mixing CPU metadata with device-side tensor selection.
2. Add a dedicated benchmark script so device comparisons are standardized.
3. Search larger `batch_size` values on MPS to see where throughput improves without harming convergence.
4. Consider pre-materializing full training tensors for fixed small datasets when memory allows.

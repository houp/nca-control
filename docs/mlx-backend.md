# MLX Backend Comparison

## Goal

Add a parallel Apple-native MLX implementation without removing the existing PyTorch path, then compare correctness and runtime on the current selected maze-exit model recipe.

## Scope

This step adds:

- an MLX NCA model with periodic-boundary convolutions
- MLX training and evaluation scripts
- torch-to-MLX checkpoint conversion for backend parity checks
- a three-backend benchmark script

The PyTorch training and inference stack remains unchanged.

## Files Added

- `src/nca_control/mlx_backend.py`
- `scripts/train_mlx_one_step.py`
- `scripts/evaluate_mlx_one_step.py`
- `scripts/evaluate_mlx_generalization.py`
- `scripts/convert_torch_checkpoint_to_mlx.py`
- `scripts/benchmark_backends.py`
- `tests/test_mlx_backend.py`

## Correctness Checks

### Backend Parity

The MLX implementation includes a direct torch-to-MLX weight copy path.

Verification:

- the test suite copies a torch model state dict into an MLX model
- both backends evaluate `forward_logits`
- outputs match numerically within `1e-5`

This is stronger than comparing only final training metrics because it checks that the MLX periodic-convolution implementation matches the PyTorch model semantics directly.

### Native MLX Training

A fresh MLX training run was executed with the current selected architecture:

- task: `maze_exit`
- grid: `9x9`
- mazes: `16`
- epochs: `150`
- batch size: `64`
- hidden channels: `32`
- perception kernel: `3`
- update kernel: `1`
- seed: `0`

Output:

- checkpoint: `runs/mlx_h32_p3_u1/checkpoint_mlx`
- weights: `runs/mlx_h32_p3_u1/weights.npz`
- final loss: `0.029818`
- training time: `17.052 s`
- samples / second: `30260.19`

Evaluation:

- one-step:
  - `full_state_accuracy = 1.0`
  - `termination_accuracy = 1.0`
- rollout:
  - `30x30`, `16` sequences, `50` steps: `exact_rollout_rate = 1.0`
  - `50x50`, `16` sequences, `50` steps: `exact_rollout_rate = 1.0`

## Runtime Benchmark

The corrected three-backend benchmark used the same recipe:

- task: `maze_exit`
- grid: `9x9`
- mazes: `16`
- epochs: `40`
- batch size: `64`
- hidden channels: `32`
- perception kernel: `3`
- update kernel: `1`

Benchmark output:

- `runs/backend-bench-9x9-fixed/summary.json`

| Backend | Device | Samples / Second | Total Train Time (s) | Final Loss |
| --- | --- | ---: | ---: | ---: |
| PyTorch CPU | `cpu` | `4977.51` | `27.644` | `0.971998` |
| PyTorch MPS | `mps` | `3082.96` | `44.632` | `0.972063` |
| MLX | `mlx` | `40871.85` | `3.367` | `0.979590` |

## Interpretation

- MLX is dramatically faster than both PyTorch CPU and PyTorch MPS on the current small `9x9` best-model workload.
- The earlier PyTorch MPS slowdown is therefore not just “GPU overhead vs CPU work”; on this machine, MLX is a much better fit for Apple Silicon than the current PyTorch MPS path.
- The MLX backend still reaches the same functional target:
  - exact one-step maze-exit behavior
  - exact larger-grid rollout behavior
- PyTorch should still be kept because it remains the portable backend for CPU and CUDA-class machines.

## Runtime Notes

- MLX import and execution work correctly outside the Codex sandbox on this machine.
- In-sandbox MLX imports abort the interpreter, so MLX-specific tests are gated behind `RUN_MLX_TESTS=1` and are intended to be run unsandboxed.

## Recommended Next Step

If Apple Silicon is the primary target, use the MLX backend for future training sweeps and keep the PyTorch path as the cross-platform reference implementation.

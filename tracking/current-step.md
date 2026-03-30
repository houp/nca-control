# Current Step

## Step ID

42

## Title

MLX Visualizer Checkpoint Support

## Scope

- make the browser visualizer work with MLX checkpoints
- keep the existing PyTorch visualization path working
- update docs and tests so the supported checkpoint types are clear

## Exit Criteria

- the browser visualizer accepts MLX `checkpoint_mlx` files without crashing
- the existing PyTorch checkpoint path still works
- tests cover backend-aware visualizer checkpoint handling
- README documents the MLX visualizer path clearly
- tracking is updated
- local git commit is created

## Notes

The current user-facing failure is specific and reproducible: `scripts/interactive_compare.py` still assumes a PyTorch checkpoint loader, so MLX `checkpoint_mlx` files fail immediately. The fix should stay small and should not regress the existing PyTorch visualization path.

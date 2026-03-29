# Current Step

## Step ID

22

## Title

Minimal maze-exit architecture sweep

## Scope

- retrain the `maze_exit` task from scratch after the `runs/` cleanup
- start from the simplest architectures and increase complexity only as needed
- stop at the first architecture that is exact on the training grid and on larger unseen grids

## Exit Criteria

- at least one fresh checkpoint is trained from scratch
- hidden-channel and kernel-size sweep results are recorded
- the selected minimal model is verified on multiple grid sizes
- findings are captured in markdown tracking
- step is logged in `tracking/step-history.md`
- local git commit is created

## Notes

Step 21 added architecture knobs for `perception_kernel_size` and `update_kernel_size`, so the sweep can now compare genuinely simpler and more complex convolutional models instead of only changing hidden width.

Environment note: PyTorch MPS is available on this machine when checked outside the Codex sandbox. In-sandbox checks report `False`, so GPU-sensitive verification may require unsandboxed execution.

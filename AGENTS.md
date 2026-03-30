# Repository Working Agreement

This repository hosts a stepwise research experiment for controllable Neural Cellular Automata (NCA).

## Core Workflow

1. Work in very small steps.
2. Start each step by updating the Markdown planning and tracking files.
3. Implement the smallest viable change for the current step.
4. Add or extend tests in the same step.
5. Run verification before committing.
6. Commit each completed step to local git.

## Technical Direction

- Environment management: `uv`
- Primary interpreter target: Python `3.13`
- ML framework targets: MLX on Apple Silicon and PyTorch for portable CPU, Apple Silicon MPS, and experimental Linux CUDA execution
- First principle: lock down deterministic movement semantics before training any model

## Documentation Requirements

- `README.md`: high-level project entry point
- `docs/project-plan.md`: detailed roadmap and experiment design
- `docs/testing-strategy.md`: verification and test matrix
- `tracking/current-step.md`: current step objective and exit criteria
- `tracking/step-history.md`: immutable step log
- whenever code changes, update the corresponding documentation in the same step so the repository stays aligned with the actual implementation
- whenever a substantial project change is made, update the technical report and presentation so they reflect the current code base and the current best verified simulation results

## Implementation Priorities

1. Deterministic reference dynamics
2. Text-mode simulation and automated tests
3. GPU-ready training pipeline
4. Interactive visualization and manual verification
5. Research instrumentation, metrics, and ablations

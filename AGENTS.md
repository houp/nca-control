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
- ML framework target: PyTorch with Apple Silicon MPS support
- First principle: lock down deterministic movement semantics before training any model

## Documentation Requirements

- `README.md`: high-level project entry point
- `docs/project-plan.md`: detailed roadmap and experiment design
- `docs/testing-strategy.md`: verification and test matrix
- `tracking/current-step.md`: current step objective and exit criteria
- `tracking/step-history.md`: immutable step log

## Implementation Priorities

1. Deterministic reference dynamics
2. Text-mode simulation and automated tests
3. GPU-ready training pipeline
4. Interactive visualization and manual verification
5. Research instrumentation, metrics, and ablations


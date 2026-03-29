# Step History

| Step | Title | Status | Verification | Commit |
| --- | --- | --- | --- | --- |
| 1 | Planning, tracking, and deterministic movement baseline | Completed | `pytest: 5 passed`; `torch 2.11.0`, `MPS available=True` (unsandboxed check) | `step-1: bootstrap deterministic baseline` |
| 2 | Text-mode simulation and CLI verification | Completed | `pytest: 9 passed`; manual CLI rollout verified with `nca-control simulate --height 2 --width 3 --row 0 --col 0 --actions right,down` | `step-2: add text simulation cli` |
| 3 | Supervised transition data generation | Completed | `pytest: 14 passed`; dataset smoke check produced `inputs [30, 6, 2, 3]`, `targets [30, 1, 2, 3]` for a `2x3` grid | `step-3: add supervised transition dataset` |
| 4 | Minimal learned NCA model | In progress | Pending | Pending |

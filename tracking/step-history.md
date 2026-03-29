# Step History

| Step | Title | Status | Verification | Commit |
| --- | --- | --- | --- | --- |
| 1 | Planning, tracking, and deterministic movement baseline | Completed | `pytest: 5 passed`; `torch 2.11.0`, `MPS available=True` (unsandboxed check) | `step-1: bootstrap deterministic baseline` |
| 2 | Text-mode simulation and CLI verification | Completed | `pytest: 9 passed`; manual CLI rollout verified with `nca-control simulate --height 2 --width 3 --row 0 --col 0 --actions right,down` | `step-2: add text simulation cli` |
| 3 | Supervised transition data generation | Completed | `pytest: 14 passed`; dataset smoke check produced `inputs [30, 6, 2, 3]`, `targets [30, 1, 2, 3]` for a `2x3` grid | `step-3: add supervised transition dataset` |
| 4 | Minimal learned NCA model | Completed | `pytest: 17 passed`; forward smoke check produced output shape `[2, 1, 3, 3]` | `step-4: add minimal controllable nca model` |
| 5 | One-step training and checkpointed inference | Completed | `pytest: 20 passed`; `scripts/train_one_step.py` wrote a checkpoint; `scripts/infer_one_step.py` loaded it and produced a prediction grid | `step-5: add one-step train and infer scripts` |
| 6 | Training quality evaluation and rollout metrics | Completed | `pytest: 22 passed`; `scripts/evaluate_one_step.py` on a `6x6`, `100`-epoch run reported `argmax_accuracy=0.8667`, `mse=0.0112`, `mean_predicted_max=0.4220` | `step-6: add checkpoint evaluation metrics` |
| 7 | Exactness-oriented model and loss refinement | In progress | Pending | Pending |

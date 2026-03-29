# Step History

| Step | Title | Status | Verification | Commit |
| --- | --- | --- | --- | --- |
| 1 | Planning, tracking, and deterministic movement baseline | Completed | `pytest: 5 passed`; `torch 2.11.0`, `MPS available=True` (unsandboxed check) | `step-1: bootstrap deterministic baseline` |
| 2 | Text-mode simulation and CLI verification | Completed | `pytest: 9 passed`; manual CLI rollout verified with `nca-control simulate --height 2 --width 3 --row 0 --col 0 --actions right,down` | `step-2: add text simulation cli` |
| 3 | Supervised transition data generation | Completed | `pytest: 14 passed`; dataset smoke check produced `inputs [30, 6, 2, 3]`, `targets [30, 1, 2, 3]` for a `2x3` grid | `step-3: add supervised transition dataset` |
| 4 | Minimal learned NCA model | Completed | `pytest: 17 passed`; forward smoke check produced output shape `[2, 1, 3, 3]` | `step-4: add minimal controllable nca model` |
| 5 | One-step training and checkpointed inference | Completed | `pytest: 20 passed`; `scripts/train_one_step.py` wrote a checkpoint; `scripts/infer_one_step.py` loaded it and produced a prediction grid | `step-5: add one-step train and infer scripts` |
| 6 | Training quality evaluation and rollout metrics | Completed | `pytest: 22 passed`; `scripts/evaluate_one_step.py` on a `6x6`, `100`-epoch run reported `argmax_accuracy=0.8667`, `mse=0.0112`, `mean_predicted_max=0.4220` | `step-6: add checkpoint evaluation metrics` |
| 7 | Exactness-oriented model and loss refinement | Completed | `pytest: 23 passed`; refined `6x6`, `100`-epoch run reported `argmax_accuracy=1.0`, `mse~=7.9e-7`, `mean_predicted_max~=0.995`; boundary inference check correctly mapped `(0,5)+right -> (0,0)` | `step-7: refine model for exact periodic control` |
| 8 | Interactive visual verification app | Completed | `pytest: 25 passed`; `python -m py_compile scripts/interactive_compare.py` passed; GUI launch still requires manual macOS verification outside the headless environment | `step-8: add interactive comparison app` |
| 9 | Browser-based visualizer compatibility fix | Completed | `pytest: 28 passed`; `python -m py_compile scripts/interactive_compare.py` passed; unsandboxed startup served `http://127.0.0.1:8765` successfully | `step-9: replace tk visualizer with browser app` |
| 10 | Multi-step rollout evaluation | Completed | `pytest: 29 passed`; `scripts/evaluate_rollout.py` on the `20x20` checkpoint reported `256000` rollout steps checked with `0` failed sequences | `step-10: add rollout evaluation tooling` |
| 11 | Browser visualizer ordering hardening | Completed | `pytest: 30 passed`; interactive session versioning tests added; unsandboxed visualizer startup re-verified at `http://127.0.0.1:8766` | `step-11: fix browser visualizer request ordering` |
| 12 | Deterministic maze semantics and generator | Completed | `pytest: 35 passed`; focused grid/maze tests passed with `10` cases verifying walls, rendering, and maze connectivity | `step-12: add deterministic maze semantics` |
| 13 | Maze-aware dataset and training pipeline | Completed | `pytest: 40 passed`; maze smoke training on MPS reached `final_loss=0.000071`; held-out one-step maze evaluation reported `argmax_accuracy=1.0`; maze rollout evaluation over `25600` steps found `0` failed sequences | `step-13: add maze-aware training pipeline` |
| 14 | Maze-aware visualizer update | In progress | Pending | Pending |

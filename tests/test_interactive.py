from __future__ import annotations

import torch

from nca_control.actions import Action
from nca_control.interactive import action_from_keysym, prediction_to_grid_state


def test_action_from_keysym_maps_controls() -> None:
    assert action_from_keysym("Up") == Action.UP
    assert action_from_keysym("Down") == Action.DOWN
    assert action_from_keysym("Left") == Action.LEFT
    assert action_from_keysym("Right") == Action.RIGHT
    assert action_from_keysym("space") == Action.NONE
    assert action_from_keysym("Return") is None


def test_prediction_to_grid_state_selects_argmax_location() -> None:
    prediction = torch.tensor([[[0.1, 0.2, 0.3], [0.0, 0.9, 0.1]]], dtype=torch.float32)

    state = prediction_to_grid_state(prediction, value=1.0)

    assert (state.row, state.col) == (1, 1)
    assert state.value == 1.0

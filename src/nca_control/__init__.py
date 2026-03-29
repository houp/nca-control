"""Controllable NCA research package."""

from .actions import Action
from .dataset import (
    ACTION_ORDER,
    action_to_one_hot,
    build_transition_dataset,
    encode_control_input,
    state_to_tensor,
)
from .device import resolve_device
from .grid import GridState, step_grid
from .inference import load_checkpoint, predict_next_state
from .model import ControllableNCAModel
from .simulation import parse_actions, rollout_states
from .train import TrainConfig, train_one_step

__all__ = [
    "ACTION_ORDER",
    "Action",
    "ControllableNCAModel",
    "GridState",
    "TrainConfig",
    "action_to_one_hot",
    "build_transition_dataset",
    "encode_control_input",
    "load_checkpoint",
    "parse_actions",
    "predict_next_state",
    "resolve_device",
    "rollout_states",
    "state_to_tensor",
    "step_grid",
    "train_one_step",
]

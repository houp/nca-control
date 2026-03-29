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
from .evaluate import decode_argmax_positions, evaluate_checkpoint
from .grid import GridState, step_grid
from .inference import load_checkpoint, predict_next_state
from .interactive import action_from_keysym, prediction_to_grid_state
from .model import ControllableNCAModel
from .simulation import parse_actions, rollout_states
from .train import TrainConfig, train_one_step

__all__ = [
    "ACTION_ORDER",
    "Action",
    "ControllableNCAModel",
    "GridState",
    "TrainConfig",
    "action_from_keysym",
    "action_to_one_hot",
    "build_transition_dataset",
    "decode_argmax_positions",
    "encode_control_input",
    "evaluate_checkpoint",
    "load_checkpoint",
    "parse_actions",
    "prediction_to_grid_state",
    "predict_next_state",
    "resolve_device",
    "rollout_states",
    "state_to_tensor",
    "step_grid",
    "train_one_step",
]

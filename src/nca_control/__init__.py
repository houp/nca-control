"""Controllable NCA research package."""

from .actions import Action
from .dataset import (
    ACTION_ORDER,
    MazeExitTransitionDataset,
    MazeTransitionDataset,
    action_to_one_hot,
    blocked_to_tensor,
    build_maze_exit_transition_dataset,
    build_maze_transition_dataset,
    build_transition_dataset,
    encode_control_input,
    exit_fill_to_tensor,
    state_to_tensor,
)
from .device import resolve_device
from .evaluate import (
    decode_argmax_positions,
    evaluate_checkpoint,
    evaluate_rollout_checkpoint,
)
from .grid import GridState, step_grid
from .inference import decode_prediction_state, load_checkpoint, predict_next_state
from .interactive import action_from_keysym, prediction_to_grid_state
from .maze import MazeLayout, generate_maze
from .model import ControllableNCAModel
from .simulation import parse_actions, rollout_states
from .train import TrainConfig, train_one_step

__all__ = [
    "ACTION_ORDER",
    "Action",
    "ControllableNCAModel",
    "MazeExitTransitionDataset",
    "GridState",
    "MazeTransitionDataset",
    "MazeLayout",
    "TrainConfig",
    "action_from_keysym",
    "action_to_one_hot",
    "blocked_to_tensor",
    "build_maze_exit_transition_dataset",
    "build_maze_transition_dataset",
    "build_transition_dataset",
    "decode_prediction_state",
    "decode_argmax_positions",
    "encode_control_input",
    "evaluate_checkpoint",
    "evaluate_rollout_checkpoint",
    "exit_fill_to_tensor",
    "generate_maze",
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

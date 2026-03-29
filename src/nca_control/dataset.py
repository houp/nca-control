from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .actions import Action
from .grid import GridState, step_grid
from .maze import MazeLayout, generate_maze

ACTION_ORDER = [
    Action.NONE,
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
]


@dataclass(frozen=True, slots=True)
class TransitionDataset:
    inputs: torch.Tensor
    targets: torch.Tensor
    action_indices: torch.Tensor
    positions: torch.Tensor


class MazeTransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        layouts: list[MazeLayout],
        examples: list[tuple[int, int, int, int]],
        value: float = 1.0,
    ) -> None:
        self.layouts = layouts
        self.examples = examples
        self.value = value

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        layout_index, row, col, action_index = self.examples[index]
        layout = self.layouts[layout_index]
        state = layout.to_grid_state(row=row, col=col, value=self.value)
        action = ACTION_ORDER[action_index]
        return encode_control_input(state, action), state_to_tensor(step_grid(state, action))


def state_to_tensor(state: GridState, device: str | torch.device = "cpu") -> torch.Tensor:
    grid = torch.zeros((1, state.height, state.width), dtype=torch.float32, device=device)
    grid[0, state.row, state.col] = state.value
    return grid


def blocked_to_tensor(state: GridState, device: str | torch.device = "cpu") -> torch.Tensor:
    grid = torch.zeros((1, state.height, state.width), dtype=torch.float32, device=device)
    for row, col in state.blocked:
        grid[0, row, col] = 1.0
    return grid


def action_to_one_hot(action: Action, device: str | torch.device = "cpu") -> torch.Tensor:
    one_hot = torch.zeros((len(ACTION_ORDER),), dtype=torch.float32, device=device)
    one_hot[ACTION_ORDER.index(action)] = 1.0
    return one_hot


def encode_control_input(
    state: GridState,
    action: Action,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    state_channel = state_to_tensor(state, device=device)
    blocked_channel = blocked_to_tensor(state, device=device)
    action_channels = action_to_one_hot(action, device=device).view(len(ACTION_ORDER), 1, 1)
    action_channels = action_channels.expand(len(ACTION_ORDER), state.height, state.width)
    return torch.cat([state_channel, blocked_channel, action_channels], dim=0)


def build_transition_dataset(
    height: int,
    width: int,
    value: float = 1.0,
    device: str | torch.device = "cpu",
) -> TransitionDataset:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    action_indices: list[int] = []
    positions: list[tuple[int, int]] = []

    for row in range(height):
        for col in range(width):
            state = GridState(height=height, width=width, row=row, col=col, value=value)
            for action_index, action in enumerate(ACTION_ORDER):
                inputs.append(encode_control_input(state, action, device=device))
                targets.append(state_to_tensor(step_grid(state, action), device=device))
                action_indices.append(action_index)
                positions.append((row, col))

    return TransitionDataset(
        inputs=torch.stack(inputs, dim=0),
        targets=torch.stack(targets, dim=0),
        action_indices=torch.tensor(action_indices, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.long, device=device),
    )


def build_maze_transition_dataset(
    height: int,
    width: int,
    num_mazes: int,
    seed: int = 0,
    value: float = 1.0,
) -> MazeTransitionDataset:
    layouts = [generate_maze(height=height, width=width, seed=seed + index) for index in range(num_mazes)]
    examples: list[tuple[int, int, int, int]] = []
    for layout_index, layout in enumerate(layouts):
        for row, col in layout.open_cells():
            for action_index, _action in enumerate(ACTION_ORDER):
                examples.append((layout_index, row, col, action_index))
    return MazeTransitionDataset(layouts=layouts, examples=examples, value=value)

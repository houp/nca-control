from __future__ import annotations

"""Dataset builders and encoders for the deterministic control tasks."""

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
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
UP_ACTION_INDEX = ACTION_TO_INDEX[Action.UP]
DOWN_ACTION_INDEX = ACTION_TO_INDEX[Action.DOWN]
LEFT_ACTION_INDEX = ACTION_TO_INDEX[Action.LEFT]
RIGHT_ACTION_INDEX = ACTION_TO_INDEX[Action.RIGHT]


@dataclass(frozen=True, slots=True)
class TransitionDataset:
    """Fully materialized plain-control dataset used by the smallest experiments."""

    inputs: torch.Tensor
    targets: torch.Tensor
    action_indices: torch.Tensor
    positions: torch.Tensor


def propose_action_positions_torch(
    rows: torch.Tensor,
    cols: torch.Tensor,
    action_indices: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized proposal step shared by dataset creation and rollout evaluation."""

    proposed_rows = torch.where(
        action_indices == UP_ACTION_INDEX,
        (rows - 1) % height,
        torch.where(action_indices == DOWN_ACTION_INDEX, (rows + 1) % height, rows),
    )
    proposed_cols = torch.where(
        action_indices == LEFT_ACTION_INDEX,
        (cols - 1) % width,
        torch.where(action_indices == RIGHT_ACTION_INDEX, (cols + 1) % width, cols),
    )
    return proposed_rows, proposed_cols


class MazeTransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy maze dataset that reconstructs batches from cached static tensor banks."""

    def __init__(
        self,
        layouts: list[MazeLayout],
        examples: list[tuple[int, int, int, int]],
        value: float = 1.0,
    ) -> None:
        self.layouts = layouts
        self.examples = torch.tensor(examples, dtype=torch.long)
        self.value = value
        self.height = layouts[0].height
        self.width = layouts[0].width
        self.blocked_grids = torch.zeros((len(layouts), self.height, self.width), dtype=torch.float32)
        self.action_grids = torch.zeros(
            (len(ACTION_ORDER), len(ACTION_ORDER), self.height, self.width),
            dtype=torch.float32,
        )
        self._tensor_bank_cache: dict[str, dict[str, torch.Tensor]] = {}
        for layout_index, layout in enumerate(layouts):
            for row, col in layout.blocked:
                self.blocked_grids[layout_index, row, col] = 1.0
        for action_index in range(len(ACTION_ORDER)):
            self.action_grids[action_index, action_index, :, :] = 1.0

    def __len__(self) -> int:
        return int(self.examples.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.materialize_batch(torch.tensor([index], dtype=torch.long))
        return inputs[0], targets[0]

    def materialize_batch(
        self,
        indices: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = torch.device(device)
        tensor_bank = self._tensor_bank_for_device(target_device)
        metadata = self.examples.index_select(0, indices.to("cpu")).to(target_device)
        layout_indices = metadata[:, 0]
        rows = metadata[:, 1]
        cols = metadata[:, 2]
        action_indices = metadata[:, 3]

        batch_size = int(metadata.shape[0])
        batch_index = torch.arange(batch_size, device=target_device)
        blocked = tensor_bank["blocked_grids"].index_select(0, layout_indices)
        action_channels = tensor_bank["action_grids"].index_select(0, action_indices)

        state_channel = torch.zeros((batch_size, 1, self.height, self.width), dtype=torch.float32, device=target_device)
        state_channel[batch_index, 0, rows, cols] = self.value
        inputs = torch.cat([state_channel, blocked.unsqueeze(1), action_channels], dim=1)

        # Compute the exact reference target in a vectorized form instead of
        # calling the Python reference engine example-by-example.
        proposed_rows, proposed_cols = propose_action_positions_torch(
            rows,
            cols,
            action_indices,
            height=self.height,
            width=self.width,
        )
        blocked_targets = blocked[batch_index, proposed_rows, proposed_cols] > 0.5
        target_rows = torch.where(blocked_targets, rows, proposed_rows)
        target_cols = torch.where(blocked_targets, cols, proposed_cols)

        targets = torch.zeros((batch_size, 1, self.height, self.width), dtype=torch.float32, device=target_device)
        targets[batch_index, 0, target_rows, target_cols] = self.value
        return inputs, targets

    def _tensor_bank_for_device(self, device: torch.device) -> dict[str, torch.Tensor]:
        device_key = str(device)
        if device_key not in self._tensor_bank_cache:
            if device.type == "cpu":
                self._tensor_bank_cache[device_key] = {
                    "blocked_grids": self.blocked_grids,
                    "action_grids": self.action_grids,
                }
            else:
                self._tensor_bank_cache[device_key] = {
                    "blocked_grids": self.blocked_grids.to(device),
                    "action_grids": self.action_grids.to(device),
                }
        return self._tensor_bank_cache[device_key]


class MazeExitTransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Maze dataset with explicit terminal fill stages for the exit-aware task."""

    def __init__(
        self,
        layouts: list[MazeLayout],
        examples: list[tuple[int, int, int, int, int, int]],
        value: float = 1.0,
    ) -> None:
        self.layouts = layouts
        self.examples = torch.tensor(examples, dtype=torch.long)
        self.value = value
        self.height = layouts[0].height
        self.width = layouts[0].width
        self.blocked_grids = torch.zeros((len(layouts), self.height, self.width), dtype=torch.float32)
        self.exit_fill_grids, self.fill_stage_lengths = _build_exit_fill_tensor_bank(layouts)
        self.exit_cells = torch.zeros((len(layouts), 2), dtype=torch.long)
        self.action_grids = torch.zeros(
            (len(ACTION_ORDER), len(ACTION_ORDER), self.height, self.width),
            dtype=torch.float32,
        )
        self._tensor_bank_cache: dict[str, dict[str, torch.Tensor]] = {}
        for layout_index, layout in enumerate(layouts):
            for row, col in layout.blocked:
                self.blocked_grids[layout_index, row, col] = 1.0
            exit_row, exit_col = layout.exit_cell
            self.exit_cells[layout_index, 0] = exit_row
            self.exit_cells[layout_index, 1] = exit_col
        for action_index in range(len(ACTION_ORDER)):
            self.action_grids[action_index, action_index, :, :] = 1.0

    def __len__(self) -> int:
        return int(self.examples.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.materialize_batch(torch.tensor([index], dtype=torch.long))
        return inputs[0], targets[0]

    def materialize_batch(
        self,
        indices: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = torch.device(device)
        tensor_bank = self._tensor_bank_for_device(target_device)
        metadata = self.examples.index_select(0, indices.to("cpu")).to(target_device)
        layout_indices = metadata[:, 0]
        rows = metadata[:, 1]
        cols = metadata[:, 2]
        action_indices = metadata[:, 3]
        terminated = metadata[:, 4].to(torch.bool)
        fill_stage_indices = metadata[:, 5]

        batch_size = int(metadata.shape[0])
        batch_index = torch.arange(batch_size, device=target_device)
        blocked = tensor_bank["blocked_grids"].index_select(0, layout_indices)
        exit_fill = _select_exit_fill_grids(
            tensor_bank["exit_fill_grids"],
            layout_indices,
            fill_stage_indices,
        )
        action_channels = tensor_bank["action_grids"].index_select(0, action_indices)
        exit_cells = tensor_bank["exit_cells"].index_select(0, layout_indices)
        exit_rows = exit_cells[:, 0]
        exit_cols = exit_cells[:, 1]

        active_channel = torch.zeros((batch_size, 1, self.height, self.width), dtype=torch.float32, device=target_device)
        active_batch_index = batch_index[~terminated]
        if active_batch_index.numel() > 0:
            active_channel[active_batch_index, 0, rows[~terminated], cols[~terminated]] = self.value
        inputs = torch.cat([active_channel, exit_fill.unsqueeze(1), blocked.unsqueeze(1), action_channels], dim=1)

        targets = torch.zeros((batch_size, 2, self.height, self.width), dtype=torch.float32, device=target_device)
        targets[:, 1, :, :] = exit_fill

        active_examples = ~terminated
        if active_examples.any():
            active_rows = rows[active_examples]
            active_cols = cols[active_examples]
            active_actions = action_indices[active_examples]
            active_layout_rows = exit_rows[active_examples]
            active_layout_cols = exit_cols[active_examples]
            active_indices = batch_index[active_examples]

            proposed_rows, proposed_cols = propose_action_positions_torch(
                active_rows,
                active_cols,
                active_actions,
                height=self.height,
                width=self.width,
            )
            blocked_targets = blocked[active_indices, proposed_rows, proposed_cols] > 0.5
            target_rows = torch.where(blocked_targets, active_rows, proposed_rows)
            target_cols = torch.where(blocked_targets, active_cols, proposed_cols)
            # Active cells disappear as soon as they hit the exit; subsequent
            # fill expansion is handled by the terminated branch below.
            reached_exit = (target_rows == active_layout_rows) & (target_cols == active_layout_cols)
            surviving = ~reached_exit
            if surviving.any():
                targets[active_indices[surviving], 0, target_rows[surviving], target_cols[surviving]] = self.value

        terminated_examples = terminated
        if terminated_examples.any():
            terminated_indices = batch_index[terminated_examples]
            max_stage_indices = (
                tensor_bank["fill_stage_lengths"].index_select(0, layout_indices[terminated_examples]) - 1
            )
            # Once terminated, the target is "advance the fill by one clock tick".
            next_stage_indices = torch.minimum(fill_stage_indices[terminated_examples] + 1, max_stage_indices)
            next_exit_fill = _select_exit_fill_grids(
                tensor_bank["exit_fill_grids"],
                layout_indices[terminated_examples],
                next_stage_indices,
            )
            targets[terminated_indices, 1, :, :] = next_exit_fill
        return inputs, targets

    def state_for_index(self, index: int) -> GridState:
        layout_index, row, col, _action_index, terminated_flag, fill_stage_index = self.examples[index].tolist()
        layout = self.layouts[layout_index]
        exit_fill = frozenset(
            tuple(cell)
            for cell in torch.nonzero(self.exit_fill_grids[layout_index, fill_stage_index], as_tuple=False).tolist()
        )
        return GridState(
            height=self.height,
            width=self.width,
            row=row,
            col=col,
            value=self.value,
            blocked=layout.blocked,
            exit_cell=layout.exit_cell,
            exit_fill=exit_fill,
            terminated=bool(terminated_flag),
        )

    def _tensor_bank_for_device(self, device: torch.device) -> dict[str, torch.Tensor]:
        device_key = str(device)
        if device_key not in self._tensor_bank_cache:
            if device.type == "cpu":
                self._tensor_bank_cache[device_key] = {
                    "blocked_grids": self.blocked_grids,
                    "exit_fill_grids": self.exit_fill_grids,
                    "fill_stage_lengths": self.fill_stage_lengths,
                    "exit_cells": self.exit_cells,
                    "action_grids": self.action_grids,
                }
            else:
                self._tensor_bank_cache[device_key] = {
                    "blocked_grids": self.blocked_grids.to(device),
                    "exit_fill_grids": self.exit_fill_grids.to(device),
                    "fill_stage_lengths": self.fill_stage_lengths.to(device),
                    "exit_cells": self.exit_cells.to(device),
                    "action_grids": self.action_grids.to(device),
                }
        return self._tensor_bank_cache[device_key]


def state_to_tensor(state: GridState, device: str | torch.device = "cpu") -> torch.Tensor:
    grid = torch.zeros((1, state.height, state.width), dtype=torch.float32, device=device)
    if not state.terminated:
        grid[0, state.row, state.col] = state.value
    return grid


def blocked_to_tensor(state: GridState, device: str | torch.device = "cpu") -> torch.Tensor:
    grid = torch.zeros((1, state.height, state.width), dtype=torch.float32, device=device)
    for row, col in state.blocked:
        grid[0, row, col] = 1.0
    return grid


def exit_fill_to_tensor(state: GridState, device: str | torch.device = "cpu") -> torch.Tensor:
    grid = torch.zeros((1, state.height, state.width), dtype=torch.float32, device=device)
    for row, col in state.exit_fill or frozenset():
        grid[0, row, col] = 1.0
    return grid


def action_to_one_hot(action: Action, device: str | torch.device = "cpu") -> torch.Tensor:
    one_hot = torch.zeros((len(ACTION_ORDER),), dtype=torch.float32, device=device)
    one_hot[ACTION_TO_INDEX[action]] = 1.0
    return one_hot


def encode_control_input(
    state: GridState,
    action: Action,
    device: str | torch.device = "cpu",
    include_exit_dynamics: bool = False,
) -> torch.Tensor:
    """Encode a deterministic state/action pair into model input channels."""

    state_channel = state_to_tensor(state, device=device)
    blocked_channel = blocked_to_tensor(state, device=device)
    action_channels = action_to_one_hot(action, device=device).view(len(ACTION_ORDER), 1, 1)
    action_channels = action_channels.expand(len(ACTION_ORDER), state.height, state.width)
    if include_exit_dynamics:
        exit_fill_channel = exit_fill_to_tensor(state, device=device)
        return torch.cat([state_channel, exit_fill_channel, blocked_channel, action_channels], dim=0)
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


def build_maze_exit_transition_dataset(
    height: int,
    width: int,
    num_mazes: int,
    seed: int = 0,
    value: float = 1.0,
) -> MazeExitTransitionDataset:
    layouts = [generate_maze(height=height, width=width, seed=seed + index) for index in range(num_mazes)]
    examples: list[tuple[int, int, int, int, int, int]] = []
    for layout_index, layout in enumerate(layouts):
        exit_row, exit_col = layout.exit_cell
        fill_stage_count = len(_build_exit_fill_stages(layout))
        for row, col in layout.open_cells():
            if (row, col) == layout.exit_cell:
                continue
            for action_index, _action in enumerate(ACTION_ORDER):
                examples.append((layout_index, row, col, action_index, 0, 0))
        for fill_stage_index in range(fill_stage_count):
            for action_index, _action in enumerate(ACTION_ORDER):
                examples.append((layout_index, exit_row, exit_col, action_index, 1, fill_stage_index))
    return MazeExitTransitionDataset(layouts=layouts, examples=examples, value=value)


def _build_exit_fill_stages(layout: MazeLayout) -> list[frozenset[tuple[int, int]]]:
    """Precompute the deterministic fill schedule used after the exit is reached."""

    state = GridState(
        height=layout.height,
        width=layout.width,
        row=layout.exit_cell[0],
        col=layout.exit_cell[1],
        blocked=layout.blocked,
        exit_cell=layout.exit_cell,
        terminated=True,
    )
    stages = [state.exit_fill or frozenset()]
    while True:
        next_state = step_grid(state, Action.NONE)
        if next_state.exit_fill == state.exit_fill:
            break
        stages.append(next_state.exit_fill or frozenset())
        state = next_state
    return stages


def _build_exit_fill_tensor_bank(
    layouts: list[MazeLayout],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack all fill schedules into a dense tensor bank for fast batch indexing."""

    stage_sets = [_build_exit_fill_stages(layout) for layout in layouts]
    max_stages = max(len(stages) for stages in stage_sets)
    height = layouts[0].height
    width = layouts[0].width
    tensors = torch.zeros((len(layouts), max_stages, height, width), dtype=torch.float32)
    lengths = torch.zeros((len(layouts),), dtype=torch.long)
    for layout_index, stages in enumerate(stage_sets):
        lengths[layout_index] = len(stages)
        for stage_index, stage in enumerate(stages):
            for row, col in stage:
                tensors[layout_index, stage_index, row, col] = 1.0
        if len(stages) < max_stages:
            tensors[layout_index, len(stages) :, :, :] = tensors[layout_index, len(stages) - 1, :, :]
    return tensors, lengths


def _select_exit_fill_grids(
    exit_fill_grids: torch.Tensor,
    layout_indices: torch.Tensor,
    fill_stage_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather the correct fill stage for each example in a batch."""

    return exit_fill_grids[layout_indices, fill_stage_indices]

from __future__ import annotations

import torch

from nca_control.actions import Action
from nca_control.dataset import (
    ACTION_ORDER,
    action_to_one_hot,
    blocked_to_tensor,
    build_maze_exit_transition_dataset,
    build_maze_transition_dataset,
    build_transition_dataset,
    encode_control_input,
    exit_fill_to_tensor,
    state_to_tensor,
)
from nca_control.grid import GridState


def _active_position(grid: torch.Tensor) -> tuple[int, int]:
    locations = torch.nonzero(grid[0], as_tuple=False)
    assert locations.shape == (1, 2)
    return int(locations[0, 0]), int(locations[0, 1])


def test_state_to_tensor_encodes_exactly_one_active_cell() -> None:
    tensor = state_to_tensor(GridState(height=3, width=4, row=1, col=2, value=2.5))

    assert tensor.shape == (1, 3, 4)
    assert torch.count_nonzero(tensor) == 1
    assert tensor[0, 1, 2].item() == 2.5


def test_action_to_one_hot_respects_declared_action_order() -> None:
    one_hot = action_to_one_hot(Action.LEFT)

    assert one_hot.shape == (5,)
    assert one_hot.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_encode_control_input_broadcasts_action_channels() -> None:
    encoded = encode_control_input(GridState(height=2, width=3, row=1, col=2), Action.UP)

    assert encoded.shape == (7, 2, 3)
    assert encoded[0, 1, 2].item() == 1.0
    assert torch.count_nonzero(encoded[1]) == 0
    assert torch.all(encoded[3] == 1.0)
    assert torch.count_nonzero(encoded[2]) == 0


def test_blocked_to_tensor_encodes_wall_cells() -> None:
    blocked = blocked_to_tensor(GridState(height=3, width=4, row=1, col=2, blocked=frozenset({(0, 0), (2, 3)})))

    assert blocked.shape == (1, 3, 4)
    assert blocked[0, 0, 0].item() == 1.0
    assert blocked[0, 2, 3].item() == 1.0
    assert torch.count_nonzero(blocked) == 2


def test_build_transition_dataset_has_expected_shape() -> None:
    dataset = build_transition_dataset(height=2, width=3)

    assert dataset.inputs.shape == (2 * 3 * 5, 7, 2, 3)
    assert dataset.targets.shape == (2 * 3 * 5, 1, 2, 3)
    assert dataset.action_indices.shape == (2 * 3 * 5,)
    assert dataset.positions.shape == (2 * 3 * 5, 2)


def test_build_transition_dataset_targets_match_reference_rule() -> None:
    dataset = build_transition_dataset(height=2, width=2)

    samples_by_key: dict[tuple[int, int, Action], int] = {}
    for index, (row, col) in enumerate(dataset.positions.tolist()):
        action = ACTION_ORDER[dataset.action_indices[index].item()]
        samples_by_key[(row, col, action)] = index

    right_index = samples_by_key[(0, 0, Action.RIGHT)]
    up_index = samples_by_key[(0, 0, Action.UP)]
    none_index = samples_by_key[(1, 1, Action.NONE)]

    assert _active_position(dataset.targets[right_index]) == (0, 1)
    assert _active_position(dataset.targets[up_index]) == (1, 0)
    assert _active_position(dataset.targets[none_index]) == (1, 1)


def test_build_maze_transition_dataset_respects_walls() -> None:
    dataset = build_maze_transition_dataset(height=9, width=9, num_mazes=1, seed=0)

    sample_inputs, sample_target = dataset[0]

    assert sample_inputs.shape[0] == 7
    assert sample_target.shape == (1, 9, 9)
    assert torch.isfinite(sample_inputs).all()


def test_exit_fill_to_tensor_encodes_exit_state_cells() -> None:
    grid = exit_fill_to_tensor(
        GridState(height=3, width=3, row=1, col=1, exit_cell=(2, 2), exit_fill=frozenset({(2, 2), (1, 2)}))
    )

    assert grid.shape == (1, 3, 3)
    assert grid[0, 1, 2].item() == 1.0
    assert grid[0, 2, 2].item() == 1.0
    assert torch.count_nonzero(grid) == 2


def test_encode_control_input_supports_exit_dynamics() -> None:
    encoded = encode_control_input(
        GridState(height=3, width=3, row=1, col=1, exit_cell=(2, 2)),
        Action.RIGHT,
        include_exit_dynamics=True,
    )

    assert encoded.shape == (8, 3, 3)
    assert encoded[0, 1, 1].item() == 1.0
    assert encoded[1, 2, 2].item() == 1.0
    assert torch.count_nonzero(encoded[2]) == 0


def test_build_maze_exit_transition_dataset_includes_terminal_examples() -> None:
    dataset = build_maze_exit_transition_dataset(height=9, width=9, num_mazes=1, seed=0)

    sample_inputs, sample_targets = dataset[0]
    terminal_indices = torch.nonzero(dataset.examples[:, 4] == 1, as_tuple=False)

    assert sample_inputs.shape[0] == 8
    assert sample_targets.shape == (2, 9, 9)
    assert terminal_indices.numel() > 0


def test_maze_exit_terminal_targets_expand_exit_fill() -> None:
    dataset = build_maze_exit_transition_dataset(height=5, width=5, num_mazes=1, seed=0)
    terminal_index = int(torch.nonzero(dataset.examples[:, 4] == 1, as_tuple=False)[0, 0].item())

    inputs, targets = dataset[terminal_index]

    assert torch.count_nonzero(inputs[0]).item() == 0
    assert torch.count_nonzero(targets[0]).item() == 0
    assert torch.count_nonzero(targets[1]).item() >= torch.count_nonzero(inputs[1]).item()

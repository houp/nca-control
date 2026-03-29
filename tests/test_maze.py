from __future__ import annotations

from collections import deque

from nca_control.actions import Action
from nca_control.grid import GridState, step_grid
from nca_control.maze import generate_maze


def test_blocked_cell_prevents_movement() -> None:
    state = GridState(height=5, width=5, row=2, col=2, blocked=frozenset({(2, 3)}))

    next_state = step_grid(state, Action.RIGHT)

    assert (next_state.row, next_state.col) == (2, 2)
    assert next_state.blocked == state.blocked


def test_text_render_shows_walls_and_player() -> None:
    state = GridState(height=3, width=4, row=1, col=2, blocked=frozenset({(0, 0), (2, 3)}))

    rendered = state.as_text()

    assert rendered.splitlines() == [
        "#...",
        "..X.",
        "...#",
    ]


def test_maze_generator_keeps_outer_border_blocked() -> None:
    layout = generate_maze(height=11, width=11, seed=3)

    for row in range(layout.height):
        assert (row, 0) in layout.blocked
        assert (row, layout.width - 1) in layout.blocked
    for col in range(layout.width):
        assert (0, col) in layout.blocked
        assert (layout.height - 1, col) in layout.blocked


def test_maze_generator_creates_connected_open_region() -> None:
    layout = generate_maze(height=11, width=11, seed=7)
    open_cells = layout.open_cells()
    assert open_cells

    visited: set[tuple[int, int]] = set()
    queue = deque([open_cells[0]])
    visited.add(open_cells[0])

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if not (0 <= next_row < layout.height and 0 <= next_col < layout.width):
                continue
            if (next_row, next_col) in layout.blocked:
                continue
            if (next_row, next_col) in visited:
                continue
            visited.add((next_row, next_col))
            queue.append((next_row, next_col))

    assert set(open_cells) == visited


def test_maze_layout_converts_to_grid_state() -> None:
    layout = generate_maze(height=9, width=9, seed=1)
    row, col = layout.open_cells()[0]

    state = layout.to_grid_state(row=row, col=col, value=2.0)

    assert state.blocked == layout.blocked
    assert state.value == 2.0

from __future__ import annotations

from nca_control.actions import Action
from nca_control.grid import GridState, step_grid


def test_none_action_preserves_position_and_value() -> None:
    state = GridState(height=5, width=7, row=2, col=3, value=4.0)

    next_state = step_grid(state, Action.NONE)

    assert next_state.row == 2
    assert next_state.col == 3
    assert next_state.value == 4.0


def test_directional_actions_move_exactly_one_cell() -> None:
    state = GridState(height=5, width=5, row=2, col=2, value=1.0)

    up_state = step_grid(state, Action.UP)
    down_state = step_grid(state, Action.DOWN)
    left_state = step_grid(state, Action.LEFT)
    right_state = step_grid(state, Action.RIGHT)

    assert (up_state.row, up_state.col) == (1, 2)
    assert (down_state.row, down_state.col) == (3, 2)
    assert (left_state.row, left_state.col) == (2, 1)
    assert (right_state.row, right_state.col) == (2, 3)


def test_periodic_boundary_conditions_wrap_on_all_edges() -> None:
    assert (step_grid(GridState(4, 6, 0, 3), Action.UP).row, step_grid(GridState(4, 6, 0, 3), Action.UP).col) == (3, 3)
    assert (step_grid(GridState(4, 6, 3, 3), Action.DOWN).row, step_grid(GridState(4, 6, 3, 3), Action.DOWN).col) == (0, 3)
    assert (step_grid(GridState(4, 6, 2, 0), Action.LEFT).row, step_grid(GridState(4, 6, 2, 0), Action.LEFT).col) == (2, 5)
    assert (step_grid(GridState(4, 6, 2, 5), Action.RIGHT).row, step_grid(GridState(4, 6, 2, 5), Action.RIGHT).col) == (2, 0)


def test_text_render_contains_exactly_one_active_marker() -> None:
    state = GridState(height=3, width=4, row=1, col=2)

    rendered = state.as_text()

    assert rendered.count("X") == 1
    assert rendered.count(".") == 11


def test_zero_value_is_rejected() -> None:
    try:
        GridState(height=3, width=3, row=0, col=0, value=0.0)
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero-valued active cell")

from __future__ import annotations

from typer.testing import CliRunner

from nca_control.actions import Action
from nca_control.cli import app
from nca_control.grid import GridState
from nca_control.simulation import parse_actions, rollout_states

runner = CliRunner()


def test_parse_actions_accepts_csv_input() -> None:
    parsed = parse_actions("right, down,none,left")

    assert parsed == [Action.RIGHT, Action.DOWN, Action.NONE, Action.LEFT]


def test_rollout_states_returns_initial_frame_and_successors() -> None:
    initial = GridState(height=3, width=3, row=0, col=0)

    states = rollout_states(initial, [Action.RIGHT, Action.DOWN, Action.LEFT])

    assert [(state.row, state.col) for state in states] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
    ]


def test_rollout_states_wraps_periodically_over_multiple_steps() -> None:
    initial = GridState(height=2, width=2, row=0, col=0)

    states = rollout_states(initial, [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN])

    assert [(state.row, state.col) for state in states] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),
    ]


def test_cli_simulate_renders_all_frames() -> None:
    result = runner.invoke(
        app,
        [
            "simulate",
            "--height",
            "2",
            "--width",
            "3",
            "--row",
            "0",
            "--col",
            "0",
            "--actions",
            "right,down",
        ],
    )

    assert result.exit_code == 0
    assert "step=0 row=0 col=0 value=1.0" in result.stdout
    assert "step=1 row=0 col=1 value=1.0" in result.stdout
    assert "step=2 row=1 col=1 value=1.0" in result.stdout
    assert "X.." in result.stdout
    assert ".X." in result.stdout

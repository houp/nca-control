from __future__ import annotations

import torch

from nca_control.actions import Action
from nca_control.grid import GridState
from nca_control.interactive import (
    InteractiveCompareSession,
    action_from_keysym,
    prediction_to_grid_state,
    serialize_grid_state,
)


def test_action_from_keysym_maps_controls() -> None:
    assert action_from_keysym("Up") == Action.UP
    assert action_from_keysym("ArrowUp") == Action.UP
    assert action_from_keysym("Down") == Action.DOWN
    assert action_from_keysym("Left") == Action.LEFT
    assert action_from_keysym("Right") == Action.RIGHT
    assert action_from_keysym("space") == Action.NONE
    assert action_from_keysym("Return") is None


def test_prediction_to_grid_state_selects_argmax_location() -> None:
    prediction = torch.tensor([[[0.1, 0.2, 0.3], [0.0, 0.9, 0.1]]], dtype=torch.float32)

    state = prediction_to_grid_state(
        prediction,
        value=1.0,
        blocked=frozenset({(0, 0)}),
        exit_cell=(0, 2),
    )

    assert (state.row, state.col) == (1, 1)
    assert state.value == 1.0
    assert state.blocked == frozenset({(0, 0)})
    assert state.exit_cell == (0, 2)


def test_prediction_to_grid_state_decodes_exit_aware_prediction() -> None:
    prediction = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    state = prediction_to_grid_state(
        prediction,
        previous_state=GridState(height=2, width=3, row=0, col=0, value=1.0, exit_cell=(0, 2)),
    )

    assert state.terminated is True
    assert state.exit_fill == frozenset({(0, 2), (1, 1)})


def test_serialize_grid_state_returns_json_ready_dict() -> None:
    payload = serialize_grid_state(
        GridState(height=4, width=5, row=1, col=2, value=3.0, blocked=frozenset({(0, 0), (3, 4)}))
    )

    assert payload == {
        "height": 4,
        "width": 5,
        "row": 1,
        "col": 2,
        "value": 3.0,
        "blocked": [[0, 0], [3, 4]],
        "exit_fill": [],
        "exit_cell": None,
        "terminated": False,
    }


def test_interactive_session_reset_restores_initial_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint_path),
        initial_state=GridState(
            height=3,
            width=3,
            row=1,
            col=1,
            value=1.0,
            blocked=frozenset({(0, 0)}),
            exit_cell=(2, 2),
        ),
        device="cpu",
    )
    session.reference_state = GridState(
        height=3,
        width=3,
        row=0,
        col=1,
        value=1.0,
        blocked=frozenset({(0, 0)}),
        exit_cell=(2, 2),
    )
    session.model_state = GridState(
        height=3,
        width=3,
        row=2,
        col=2,
        value=1.0,
        blocked=frozenset({(0, 0)}),
        exit_cell=(2, 2),
        terminated=True,
    )
    session.last_action = Action.LEFT

    payload = session.reset()

    assert payload["version"] == 1
    assert payload["last_action"] == "none"
    assert payload["reference"]["row"] == 1
    assert payload["model"]["col"] == 1
    assert payload["reference"]["blocked"] == [[0, 0]]
    assert payload["reference"]["exit_cell"] == [2, 2]


def test_interactive_session_apply_action_updates_reference_and_model(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint_path),
        initial_state=GridState(
            height=3,
            width=3,
            row=0,
            col=0,
            value=1.0,
            blocked=frozenset({(2, 2)}),
            exit_cell=(0, 1),
        ),
        device="cpu",
    )

    def fake_predict_next_state(*args, **kwargs):  # type: ignore[no-untyped-def]
        return torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr("nca_control.interactive.predict_next_state_any", fake_predict_next_state)

    payload = session.apply_action(Action.RIGHT)

    assert payload["version"] == 1
    assert payload["last_action"] == "right"
    assert payload["reference"]["row"] == 0
    assert payload["reference"]["col"] == 1
    assert payload["model"]["row"] == 0
    assert payload["model"]["col"] == 1
    assert payload["reference"]["terminated"] is True
    assert payload["model"]["terminated"] is True
    assert payload["match"] is True
    assert payload["model"]["blocked"] == [[2, 2]]


def test_interactive_session_version_increments_monotonically(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint_path),
        initial_state=GridState(height=3, width=3, row=0, col=0, value=1.0, blocked=frozenset({(1, 1)}), exit_cell=(2, 2)),
        device="cpu",
    )

    def fake_predict_next_state(*args, **kwargs):  # type: ignore[no-untyped-def]
        return torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)

    monkeypatch.setattr("nca_control.interactive.predict_next_state_any", fake_predict_next_state)

    initial = session.snapshot()
    after_step = session.apply_action(Action.RIGHT)
    after_reset = session.reset()

    assert initial["version"] == 0
    assert after_step["version"] == 1
    assert after_reset["version"] == 2


def test_interactive_session_reset_factory_generates_new_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    generated = [
        GridState(height=3, width=3, row=0, col=0, value=1.0, exit_cell=(2, 2)),
        GridState(height=3, width=3, row=1, col=1, value=1.0, exit_cell=(0, 2)),
    ]

    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint_path),
        initial_state=generated[0],
        device="cpu",
        reset_factory=lambda: generated.pop(0),
    )

    payload = session.reset()

    assert payload["reference"]["row"] == 0
    payload = session.reset()
    assert payload["reference"]["row"] == 1


def test_interactive_session_none_ticks_expand_terminal_exit_fill(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_text("unused", encoding="utf-8")
    session = InteractiveCompareSession(
        checkpoint_path=str(checkpoint_path),
        initial_state=GridState(
            height=3,
            width=3,
            row=0,
            col=0,
            value=1.0,
            exit_cell=(0, 1),
        ),
        device="cpu",
    )

    def fake_predict_next_state(*args, **kwargs):  # type: ignore[no-untyped-def]
        return torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

    monkeypatch.setattr("nca_control.interactive.predict_next_state_any", fake_predict_next_state)

    first = session.apply_action(Action.RIGHT)
    second = session.apply_action(Action.NONE)

    assert first["reference"]["terminated"] is True
    assert first["model"]["terminated"] is True
    assert sorted(map(tuple, second["reference"]["exit_fill"])) == [(0, 0), (0, 1), (0, 2), (1, 1)]
    assert second["reference"]["exit_fill"] == second["model"]["exit_fill"]
    assert second["match"] is True

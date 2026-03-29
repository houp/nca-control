from __future__ import annotations

import torch

from nca_control.dataset import build_transition_dataset
from nca_control.model import ControllableNCAModel


def test_model_forward_matches_target_grid_shape() -> None:
    dataset = build_transition_dataset(height=3, width=3)
    model = ControllableNCAModel()

    outputs = model(dataset.inputs[:4])

    assert outputs.shape == (4, 1, 3, 3)


def test_model_outputs_are_bounded_and_finite() -> None:
    dataset = build_transition_dataset(height=2, width=2)
    model = ControllableNCAModel(cell_value_max=1.0)

    outputs = model(dataset.inputs[:5])

    assert torch.isfinite(outputs).all()
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)


def test_model_supports_backpropagation() -> None:
    dataset = build_transition_dataset(height=2, width=2)
    model = ControllableNCAModel()

    outputs = model(dataset.inputs[:5])
    loss = torch.nn.functional.mse_loss(outputs, dataset.targets[:5])
    loss.backward()

    assert model.perception.weight.grad is not None
    assert model.update[-1].weight.grad is not None

from __future__ import annotations

import torch

from nca_control.dataset import build_transition_dataset
from nca_control.model import ControllableNCAModel


def test_model_forward_matches_target_grid_shape() -> None:
    dataset = build_transition_dataset(height=3, width=3)
    model = ControllableNCAModel()

    outputs = model(dataset.inputs[:4])

    assert outputs.shape == (4, 1, 3, 3)
    assert model.perception.padding_mode == "circular"


def test_model_accepts_custom_kernel_sizes() -> None:
    dataset = build_transition_dataset(height=3, width=3)
    model = ControllableNCAModel(perception_kernel_size=5, update_kernel_size=3)

    outputs = model(dataset.inputs[:4])

    assert outputs.shape == (4, 1, 3, 3)
    assert model.perception.kernel_size == (5, 5)
    assert model.update[1].kernel_size == (3, 3)


def test_model_rejects_even_kernel_sizes() -> None:
    try:
        ControllableNCAModel(perception_kernel_size=2)
    except ValueError as error:
        assert "odd integer" in str(error)
    else:
        raise AssertionError("expected ValueError for even perception kernel")

    try:
        ControllableNCAModel(update_kernel_size=4)
    except ValueError as error:
        assert "odd integer" in str(error)
    else:
        raise AssertionError("expected ValueError for even update kernel")


def test_model_outputs_are_bounded_and_finite() -> None:
    dataset = build_transition_dataset(height=2, width=2)
    model = ControllableNCAModel(cell_value_max=1.0)

    outputs = model(dataset.inputs[:5])

    assert torch.isfinite(outputs).all()
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
    assert torch.allclose(outputs.sum(dim=(1, 2, 3)), torch.ones(5))


def test_multi_channel_model_outputs_are_bounded_and_finite() -> None:
    inputs = torch.zeros((3, 8, 4, 4), dtype=torch.float32)
    model = ControllableNCAModel(input_channels=8, state_channels=2, cell_value_max=1.0)

    outputs = model(inputs)

    assert outputs.shape == (3, 2, 4, 4)
    assert torch.isfinite(outputs).all()
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)


def test_model_supports_backpropagation() -> None:
    dataset = build_transition_dataset(height=2, width=2)
    model = ControllableNCAModel()

    logits = model.forward_logits(dataset.inputs[:5])
    target_indices = torch.argmax(dataset.targets[:5].view(5, -1), dim=1)
    loss = torch.nn.functional.cross_entropy(logits.view(5, -1), target_indices)
    loss.backward()

    assert model.perception.weight.grad is not None
    assert model.update[-1].weight.grad is not None

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

if os.environ.get("RUN_MLX_TESTS") != "1":
    pytest.skip("MLX tests require unsandboxed Apple runtime", allow_module_level=True)

import mlx.core as mx

from nca_control.actions import Action
from nca_control.grid import GridState
from nca_control.inference import predict_next_state
from nca_control.mlx_backend import (
    MLXControllableNCAModel,
    apply_torch_state_dict_to_mlx_model,
    convert_torch_checkpoint_to_mlx,
    evaluate_mlx_checkpoint,
    predict_next_state_mlx,
    train_one_step_mlx,
)
from nca_control.model import ControllableNCAModel
from nca_control.train import TrainConfig, train_one_step


def test_mlx_forward_logits_match_torch_after_weight_copy() -> None:
    torch.manual_seed(0)
    torch_model = ControllableNCAModel(
        input_channels=8,
        state_channels=2,
        hidden_channels=4,
        perception_kernel_size=3,
        update_kernel_size=1,
    )
    mlx_model = MLXControllableNCAModel(
        input_channels=8,
        state_channels=2,
        hidden_channels=4,
        perception_kernel_size=3,
        update_kernel_size=1,
    )
    apply_torch_state_dict_to_mlx_model(mlx_model, torch_model.state_dict())

    torch_inputs = torch.randn((2, 8, 5, 5), dtype=torch.float32)
    torch_logits = torch_model.forward_logits(torch_inputs).detach().cpu().numpy()
    mlx_inputs = mx.array(np.transpose(torch_inputs.detach().cpu().numpy(), (0, 2, 3, 1)))
    mlx_logits = mlx_model.forward_logits(mlx_inputs)
    mx.eval(mlx_logits)

    assert np.allclose(torch_logits, np.transpose(np.array(mlx_logits), (0, 3, 1, 2)), atol=1e-5)


def test_train_one_step_mlx_creates_checkpoint_and_metrics(tmp_path) -> None:
    result = train_one_step_mlx(
        TrainConfig(
            task="maze_exit",
            height=5,
            width=5,
            num_mazes=2,
            eval_num_mazes=1,
            epochs=1,
            batch_size=8,
            hidden_channels=8,
            seed=0,
        ),
        output_dir=tmp_path,
    )

    metrics = result["metrics"]

    assert result["checkpoint_path"].exists()
    assert result["weights_path"].exists()
    assert result["config_path"].exists()
    assert result["metrics_path"].exists()
    assert result["progress_path"].exists()
    assert result["latest_status_path"].exists()
    assert metrics["device"] == "mlx"
    latest_status = json.loads(result["latest_status_path"].read_text(encoding="utf-8"))
    assert latest_status["status"] == "completed"
    assert latest_status["epoch"] == 1


def test_converted_torch_checkpoint_matches_mlx_prediction(tmp_path) -> None:
    torch_result = train_one_step(
        TrainConfig(
            task="maze_exit",
            height=5,
            width=5,
            num_mazes=2,
            eval_num_mazes=1,
            epochs=1,
            batch_size=8,
            hidden_channels=8,
            device="cpu",
            seed=0,
        ),
        output_dir=tmp_path / "torch",
    )
    mlx_checkpoint = convert_torch_checkpoint_to_mlx(
        torch_result["checkpoint_path"],
        tmp_path / "mlx_converted",
    )["checkpoint_path"]
    state = GridState(height=5, width=5, row=1, col=1, exit_cell=(3, 3))

    torch_prediction = predict_next_state(
        torch_result["checkpoint_path"],
        state,
        Action.RIGHT,
        device="cpu",
        hard_decode=False,
    )
    mlx_prediction = predict_next_state_mlx(
        mlx_checkpoint,
        state,
        Action.RIGHT,
        hard_decode=False,
    )

    assert torch.allclose(torch_prediction, mlx_prediction, atol=1e-5)
    metrics = evaluate_mlx_checkpoint(mlx_checkpoint, batch_size=16)
    assert "full_state_accuracy" in metrics

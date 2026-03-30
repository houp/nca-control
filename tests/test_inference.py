from __future__ import annotations

import json

from nca_control.inference import detect_checkpoint_backend, load_checkpoint_config


def test_detect_checkpoint_backend_recognizes_mlx_json_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"task": "maze_exit", "maze_seed": 7}), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint_mlx"
    checkpoint_path.write_text(
        json.dumps({"weights": str(tmp_path / "weights.npz"), "config": str(config_path)}),
        encoding="utf-8",
    )

    assert detect_checkpoint_backend(checkpoint_path) == "mlx"


def test_load_checkpoint_config_reads_mlx_config_without_torch_load(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"task": "maze_exit", "maze_seed": 11}), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint_mlx"
    checkpoint_path.write_text(
        json.dumps({"weights": str(tmp_path / "weights.npz"), "config": str(config_path)}),
        encoding="utf-8",
    )

    config = load_checkpoint_config(checkpoint_path, device="cpu")

    assert config["task"] == "maze_exit"
    assert config["maze_seed"] == 11

from pathlib import Path

import pytest

from vns.config.config_manager import ConfigManager, InvalidConfigError


def test_load_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_root.yaml"
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(
        InvalidConfigError,
        match="top-level YAML document must be a mapping",
    ):
        ConfigManager.load(config_path)


def test_load_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("camera: [640, 480\n", encoding="utf-8")

    with pytest.raises(InvalidConfigError, match="Failed to parse configuration YAML"):
        ConfigManager.load(config_path)


def test_get_returns_copy_for_nested_values() -> None:
    config = ConfigManager({"camera": {"intrinsics": {"fx": 550.0}}})

    intrinsics = config.get("camera.intrinsics")

    assert intrinsics == {"fx": 550.0}
    assert isinstance(intrinsics, dict)

    intrinsics["fx"] = 1.0

    assert config.get("camera.intrinsics.fx") == 550.0


def test_get_rejects_empty_key_path() -> None:
    config = ConfigManager({"camera": {"width": 640}})

    with pytest.raises(ValueError, match="key_path must not be empty"):
        config.get("")


def test_validation_rejects_invalid_camera_distortion_length() -> None:
    with pytest.raises(InvalidConfigError, match="camera.distortion"):
        ConfigManager({"camera": {"distortion": [0.1, 0.2, 0.3]}})


def test_model_returns_validated_defaults() -> None:
    config = ConfigManager({"database": {"path": "custom.vnsdb"}})

    model = config.model

    assert model.database.path == "custom.vnsdb"
    assert model.preprocessing.undistort is True
    assert model.logging.format == "json"

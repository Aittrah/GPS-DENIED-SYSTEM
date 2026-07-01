from pathlib import Path

import yaml

from vns.config import ConfigManager, GroundTextureConfig
from vns.config.models import MavlinkConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "simulation" / "config" / "simulation.yaml"


def test_active_simulation_config_uses_px4_sdk_udp_port() -> None:
    config = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))

    assert config["mavlink"]["connection_string"] == "udp://:14540"


def test_mavlink_model_default_matches_px4_sdk_udp_port() -> None:
    assert MavlinkConfig().connection_string == "udp://:14540"


def test_ground_texture_config_parses_and_resolves_to_model_asset() -> None:
    config = ConfigManager.load(SIM_CONFIG)
    ground_texture = config.model.ground_texture

    assert isinstance(ground_texture, GroundTextureConfig)
    assert ground_texture.texture_path == (
        "../models/qau_ground_plane/materials/textures/qau_satellite.png"
    )
    assert ground_texture.world_size_east_m == 600.0
    assert ground_texture.world_size_north_m == 450.6
    assert config.resolve_path(ground_texture.texture_path).exists()

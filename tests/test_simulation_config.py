from pathlib import Path

import yaml

from vns.config.models import MavlinkConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO_ROOT / "simulation" / "config" / "simulation.yaml"


def test_active_simulation_config_uses_px4_sdk_udp_port() -> None:
    config = yaml.safe_load(SIM_CONFIG.read_text(encoding="utf-8"))

    assert config["mavlink"]["connection_string"] == "udp://:14540"


def test_mavlink_model_default_matches_px4_sdk_udp_port() -> None:
    assert MavlinkConfig().connection_string == "udp://:14540"

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "simulation" / "models" / "qau_ground_plane"
MODEL_CONFIG = MODEL_ROOT / "model.config"
MODEL_SDF = MODEL_ROOT / "model.sdf"
MATERIAL = MODEL_ROOT / "materials" / "scripts" / "qau_satellite.material"
TEXTURE = MODEL_ROOT / "materials" / "textures" / "qau_satellite.png"
WORLD_FILES = [
    REPO_ROOT / "simulation" / "worlds" / "uav_test_world.sdf",
    REPO_ROOT / "simulation" / "worlds" / "uav_localization_test.sdf",
]


def test_model_package_exists() -> None:
    assert MODEL_CONFIG.exists()
    assert MODEL_SDF.exists()
    assert MATERIAL.exists()
    assert TEXTURE.exists()


def test_model_config_and_sdf_parse() -> None:
    config_root = ET.parse(MODEL_CONFIG).getroot()
    sdf_root = ET.parse(MODEL_SDF).getroot()

    assert config_root.findtext("name") == "QAU Ground Plane (Satellite Textured)"
    assert sdf_root.tag == "sdf"
    model = sdf_root.find("model")
    assert model is not None
    assert model.attrib["name"] == "qau_ground_plane"


def test_model_sdf_uses_portable_model_uris_only() -> None:
    content = MODEL_SDF.read_text(encoding="utf-8")

    assert "model://qau_ground_plane/materials/scripts" in content
    assert "model://qau_ground_plane/materials/textures" in content
    assert "file://" not in content
    assert "/home/" not in content


def test_active_worlds_include_qau_ground_plane_model() -> None:
    for world_file in WORLD_FILES:
        content = world_file.read_text(encoding="utf-8")
        root = ET.parse(world_file).getroot()

        assert "model://qau_ground_plane" in content
        assert 'model name="qau_ground"' not in content
        assert "file://materials/scripts" not in content
        assert "file://materials/textures" not in content

        world = root.find("world")
        assert world is not None
        includes = [
            include.findtext("uri")
            for include in world.findall("include")
            if include.findtext("uri")
        ]
        assert "model://qau_ground_plane" in includes

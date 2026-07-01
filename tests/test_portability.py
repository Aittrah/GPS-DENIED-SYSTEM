from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from vns.config.config_manager import ConfigManager
from vns.database.reference_db import DatabaseEntry, ReferenceDatabase
from vns.utils.paths import (
    prepend_search_path,
    resolve_simulation_root,
    resolve_vns_python,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_ROOT = REPO_ROOT / "simulation"
DB_INDEX = SIMULATION_ROOT / "database" / "images" / "database_index.yaml"
COMMITTED_DB = SIMULATION_ROOT / "database" / "qau_campus.vnsdb"


def _make_simulation_layout(root: Path) -> Path:
    simulation_root = root / "simulation"
    for child in ("config", "database", "launch", "models", "scripts", "worlds"):
        (simulation_root / child).mkdir(parents=True, exist_ok=True)
    return simulation_root


def test_resolve_simulation_root_from_source_tree_anchor(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simulation_root = _make_simulation_layout(repo_root)
    anchor = repo_root / "src" / "vns" / "cli.py"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("", encoding="utf-8")

    assert resolve_simulation_root(anchor) == simulation_root


def test_resolve_simulation_root_from_installed_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    prefix = tmp_path / "prefix"
    simulation_root = prefix / "share" / "vns" / "simulation"
    for child in ("config", "database", "launch", "models", "scripts", "worlds"):
        (simulation_root / child).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sys, "prefix", str(prefix))

    assert resolve_simulation_root() == simulation_root


def test_resolve_vns_python_prefers_env_then_repo_venv(tmp_path: Path) -> None:
    simulation_root = _make_simulation_layout(tmp_path / "repo")
    venv_python = simulation_root.parent / ".venv-run" / "bin" / "python3"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")

    assert resolve_vns_python(simulation_root, env={"VNS_PYTHON": "/tmp/custom-python"}) == (
        "/tmp/custom-python"
    )
    assert resolve_vns_python(simulation_root, env={}) == str(venv_python)


def test_prepend_search_path_avoids_empty_path_segments() -> None:
    assert prepend_search_path("/new", "GAZEBO_MODEL_PATH", env={}) == "/new"
    assert prepend_search_path(
        "/new",
        "GAZEBO_MODEL_PATH",
        env={"GAZEBO_MODEL_PATH": "/old"},
    ) == "/new:/old"


def test_config_manager_resolves_database_path_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "simulation.yaml"
    config_path.write_text("database:\n  path: ../database/test.vnsdb\n", encoding="utf-8")

    config = ConfigManager.load(config_path)

    assert config.resolve_path(config.model.database.path) == tmp_path / "database" / "test.vnsdb"


def test_reference_database_resolves_source_paths_relative_to_db_file(tmp_path: Path) -> None:
    image_path = tmp_path / "artifacts" / "images" / "ref.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"test")

    db = ReferenceDatabase(name="portable")
    db.add_entry(
        DatabaseEntry(
            id="ref",
            source_path=str(image_path),
            latitude=33.7470,
            longitude=73.1370,
            altitude=550.0,
            heading=0.0,
            capture_time="2024-01-01T00:00:00",
            feature_count=0,
            feature_algorithm="ORB",
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 32), dtype=np.uint8),
            metadata={},
        )
    )

    db_path = tmp_path / "artifacts" / "portable.vnsdb"
    db.save(str(db_path))
    loaded = ReferenceDatabase.load(str(db_path))

    assert loaded.entries["ref"].source_path == "images/ref.jpg"
    assert loaded.resolve_source_path(loaded.entries["ref"].source_path) == image_path


def test_committed_database_index_uses_relative_filepaths() -> None:
    index = yaml.safe_load(DB_INDEX.read_text(encoding="utf-8"))

    assert index["images"]
    assert all(not Path(image["filepath"]).is_absolute() for image in index["images"])


def test_committed_database_source_paths_resolve_from_db_location() -> None:
    db = ReferenceDatabase.load(str(COMMITTED_DB))

    assert db.entries
    for entry in db.entries.values():
        assert not Path(entry.source_path).is_absolute()
        assert db.resolve_source_path(entry.source_path).exists()


def test_launch_and_world_assets_are_portable() -> None:
    launch_files = [
        SIMULATION_ROOT / "launch" / "full_simulation.launch.py",
        SIMULATION_ROOT / "launch" / "UAV_simulation.launch.py",
        SIMULATION_ROOT / "launch" / "localization_test.launch.py",
    ]
    world_files = [
        SIMULATION_ROOT / "worlds" / "uav_test_world.sdf",
        SIMULATION_ROOT / "worlds" / "uav_localization_test.sdf",
    ]

    for launch_file in launch_files:
        content = launch_file.read_text(encoding="utf-8")
        assert "/home/" not in content
        assert "GAZEBO_RESOURCE_PATH" in content

    for world_file in world_files:
        content = world_file.read_text(encoding="utf-8")
        assert "model://qau_ground_plane" in content
        assert "file://materials/scripts" not in content
        assert "file://materials/textures" not in content
        assert "/home/" not in content

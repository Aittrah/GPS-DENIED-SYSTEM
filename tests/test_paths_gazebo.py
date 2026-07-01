"""Unit tests for the shared Gazebo environment-path helpers."""

import os

import pytest

from vns.utils import paths
from vns.utils.paths import (
    compose_gazebo_model_path,
    compose_gazebo_resource_path,
    detect_gazebo_share_dir,
)


def _make_share(root, name, *, with_programs=True, with_media=True, with_models=False):
    base = root / name
    if with_programs:
        (base / "media" / "materials" / "programs").mkdir(parents=True, exist_ok=True)
    elif with_media:
        (base / "media").mkdir(parents=True, exist_ok=True)
    else:
        base.mkdir(parents=True, exist_ok=True)
    if with_models:
        (base / "models").mkdir(parents=True, exist_ok=True)
    return base


def test_detect_prefers_dir_with_shader_programs(tmp_path):
    # gazebo-9 only has media/, gazebo-11 has the shader programs.
    _make_share(tmp_path, "gazebo-9", with_programs=False, with_media=True)
    expected = _make_share(tmp_path, "gazebo-11", with_programs=True)
    assert detect_gazebo_share_dir(tmp_path) == str(expected)


def test_detect_picks_highest_version_when_several_have_programs(tmp_path):
    _make_share(tmp_path, "gazebo-9", with_programs=True)
    _make_share(tmp_path, "gazebo-10", with_programs=True)
    expected = _make_share(tmp_path, "gazebo-11", with_programs=True)
    # Lexicographic sort would wrongly pick gazebo-9; numeric key must win.
    assert detect_gazebo_share_dir(tmp_path) == str(expected)


def test_detect_falls_back_to_media_only_dir(tmp_path):
    expected = _make_share(tmp_path, "gazebo-11", with_programs=False, with_media=True)
    assert detect_gazebo_share_dir(tmp_path) == str(expected)


def test_detect_returns_none_when_absent(tmp_path):
    (tmp_path / "not-gazebo").mkdir()
    assert detect_gazebo_share_dir(tmp_path) is None


def test_compose_resource_path_orders_worlds_then_share_then_existing(monkeypatch):
    monkeypatch.setattr(paths, "detect_gazebo_share_dir", lambda *a, **k: "/usr/share/gazebo-11")
    value = compose_gazebo_resource_path(
        "/proj/worlds", env={"GAZEBO_RESOURCE_PATH": "/pre/existing"}
    )
    assert value.split(os.pathsep) == [
        "/proj/worlds",
        "/usr/share/gazebo-11",
        "/pre/existing",
    ]


def test_compose_resource_path_without_existing_or_share(monkeypatch):
    monkeypatch.setattr(paths, "detect_gazebo_share_dir", lambda *a, **k: None)
    value = compose_gazebo_resource_path("/proj/worlds", env={})
    assert value.split(os.pathsep) == ["/proj/worlds"]


def test_compose_model_path_appends_base_models(monkeypatch, tmp_path):
    share = _make_share(tmp_path, "gazebo-11", with_programs=True, with_models=True)
    monkeypatch.setattr(paths, "detect_gazebo_share_dir", lambda *a, **k: str(share))
    value = compose_gazebo_model_path("/proj/models", env={})
    assert value.split(os.pathsep) == ["/proj/models", str(share / "models")]


def test_compose_model_path_skips_missing_base_models(monkeypatch, tmp_path):
    # share dir exists but has no models/ subdir -> not appended.
    share = _make_share(tmp_path, "gazebo-11", with_programs=True, with_models=False)
    monkeypatch.setattr(paths, "detect_gazebo_share_dir", lambda *a, **k: str(share))
    value = compose_gazebo_model_path("/proj/models", env={"GAZEBO_MODEL_PATH": "/pre"})
    assert value.split(os.pathsep) == ["/proj/models", "/pre"]


def test_compose_model_path_appends_extra_model_dirs(monkeypatch, tmp_path):
    share = _make_share(tmp_path, "gazebo-11", with_programs=True, with_models=True)
    monkeypatch.setattr(paths, "detect_gazebo_share_dir", lambda *a, **k: str(share))
    extra = tmp_path / "px4-models"
    extra.mkdir()
    value = compose_gazebo_model_path(
        "/proj/models",
        extra_model_dirs=(extra,),
        env={},
    )
    assert value.split(os.pathsep) == [
        "/proj/models",
        str(extra),
        str(share / "models"),
    ]


def test_resolve_px4_gazebo_models_dir(tmp_path):
    models = tmp_path / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models"
    models.mkdir(parents=True)
    assert paths.resolve_px4_gazebo_models_dir(tmp_path) == models
    assert paths.resolve_px4_gazebo_models_dir(tmp_path / "missing") is None


def test_resolve_px4_gazebo_plugin_dir(tmp_path):
    plugins = tmp_path / "build/px4_sitl_default/build_gazebo-classic"
    plugins.mkdir(parents=True)
    assert paths.resolve_px4_gazebo_plugin_dir(tmp_path) == plugins
    assert paths.resolve_px4_gazebo_plugin_dir(tmp_path / "missing") is None


def test_compose_gazebo_plugin_path(tmp_path):
    plugins = tmp_path / "build/px4_sitl_default/build_gazebo-classic"
    plugins.mkdir(parents=True)
    value = paths.compose_gazebo_plugin_path(px4_root=tmp_path, env={})
    assert value.split(os.pathsep) == [str(plugins)]


def test_compose_gazebo_plugin_path_preserves_existing(tmp_path):
    plugins = tmp_path / "build/px4_sitl_default/build_gazebo-classic"
    plugins.mkdir(parents=True)
    value = paths.compose_gazebo_plugin_path(
        px4_root=tmp_path,
        env={"GAZEBO_PLUGIN_PATH": "/pre/existing"},
    )
    assert value.split(os.pathsep) == [str(plugins), "/pre/existing"]


def test_compose_px4_ld_library_path(tmp_path):
    plugins = tmp_path / "build/px4_sitl_default/build_gazebo-classic"
    plugins.mkdir(parents=True)
    value = paths.compose_px4_ld_library_path(
        px4_root=tmp_path,
        env={"LD_LIBRARY_PATH": "/pre/lib"},
    )
    assert value.split(os.pathsep) == [str(plugins), "/pre/lib"]

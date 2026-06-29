from pathlib import Path

import pytest

from vns.utils.camera_topic import (
    DEFAULT_CAMERA_TOPIC,
    render_sdf_with_camera_topic,
    split_camera_topic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SDF = REPO_ROOT / 'simulation' / 'models' / 'iris_downward_cam' / 'model.sdf'


def test_split_camera_topic_default() -> None:
    assert split_camera_topic(DEFAULT_CAMERA_TOPIC) == (
        '/vns_drone',
        'downward_camera',
    )


def test_split_camera_topic_override() -> None:
    assert split_camera_topic('/test_cam/downward_camera/image_raw') == (
        '/test_cam',
        'downward_camera',
    )


def test_split_camera_topic_rejects_missing_image_raw_suffix() -> None:
    with pytest.raises(ValueError, match="must end with '/image_raw'"):
        split_camera_topic('/test_cam/downward_camera')


def test_render_sdf_with_camera_topic_default_returns_unchanged_text() -> None:
    sdf_text = MODEL_SDF.read_text(encoding='utf-8')

    assert render_sdf_with_camera_topic(sdf_text, DEFAULT_CAMERA_TOPIC) == sdf_text


def test_render_sdf_with_camera_topic_override_updates_real_model() -> None:
    sdf_text = MODEL_SDF.read_text(encoding='utf-8')

    rendered = render_sdf_with_camera_topic(
        sdf_text,
        '/test_cam/downward_camera/image_raw',
    )

    assert '<namespace>/test_cam</namespace>' in rendered
    assert '<camera_name>downward_camera</camera_name>' in rendered
    assert rendered.count('<namespace>/vns_drone</namespace>') == 3
    assert "<plugin name='camera_controller' filename='libgazebo_ros_camera.so'>" in rendered

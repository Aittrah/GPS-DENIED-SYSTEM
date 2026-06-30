"""Unit tests for the downward-camera ImageFrameLogger."""

import json

import numpy as np

from vns.validation import ImageFrameLogger


def _frame(seed=0):
    rng = np.random.RandomState(seed)
    return (rng.rand(8, 8, 3) * 255).astype(np.uint8)


def test_disabled_logger_is_noop(tmp_path):
    logger = ImageFrameLogger(enabled=False, log_dir=tmp_path)
    assert logger.enabled is False
    assert logger.save(_frame()) is False
    assert not (tmp_path / "images").exists()


def test_enabled_logger_saves_png_and_manifest(tmp_path):
    logger = ImageFrameLogger(enabled=True, log_dir=tmp_path)
    assert logger.enabled is True
    assert logger.save(_frame(1), timestamp=12.5) is True
    logger.close()

    images_dir = tmp_path / "images"
    pngs = sorted(images_dir.glob("*.png"))
    manifests = list(images_dir.glob("manifest_*.jsonl"))
    assert [p.name for p in pngs] == ["frame_000000.png"]
    assert len(manifests) == 1

    records = [json.loads(line) for line in manifests[0].read_text().splitlines()]
    assert len(records) == 1
    rec = records[0]
    assert rec["seq"] == 0
    assert rec["timestamp"] == 12.5
    assert rec["file"] == "frame_000000.png"
    assert rec["width"] == 8 and rec["height"] == 8
    assert rec["std"] > 0  # non-flat content recorded


def test_stride_decimates_frames(tmp_path):
    logger = ImageFrameLogger(enabled=True, log_dir=tmp_path, stride=2)
    results = [logger.save(_frame(i)) for i in range(5)]  # seq 0..4
    logger.close()

    assert results == [True, False, True, False, True]
    assert logger.saved_count == 3
    pngs = sorted((tmp_path / "images").glob("*.png"))
    assert [p.name for p in pngs] == [
        "frame_000000.png",
        "frame_000002.png",
        "frame_000004.png",
    ]


def test_close_is_idempotent(tmp_path):
    logger = ImageFrameLogger(enabled=True, log_dir=tmp_path)
    logger.save(_frame())
    logger.close()
    logger.close()  # must not raise

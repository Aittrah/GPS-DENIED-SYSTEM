"""Shared fixtures for VNS vision pipeline tests."""

import cv2
import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry, ReferenceDatabase


def _make_textured(width: int = 640, height: int = 480) -> np.ndarray:
    """Synthetic grayscale image with many ORB-detectable corners."""
    img = np.zeros((height, width), dtype=np.uint8)
    block = 32
    for y in range(0, height, block):
        for x in range(0, width, block):
            if (x // block + y // block) % 2 == 0:
                img[y : y + block, x : x + block] = 200
    cv2.circle(img, (160, 120), 30, 128, -1)
    cv2.circle(img, (480, 360), 25, 180, -1)
    cv2.rectangle(img, (50, 300), (150, 400), 220, -1)
    rng = np.random.RandomState(42)
    noise = rng.randint(0, 20, (height, width), dtype=np.uint8)
    return cv2.add(img, noise)


@pytest.fixture
def textured_image() -> np.ndarray:
    return _make_textured()


@pytest.fixture
def sample_config() -> dict:
    return {
        "preprocessing": {
            "target_width": 640,
            "target_height": 480,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": 8,
        },
        "feature_extraction": {
            "algorithm": "ORB",
            "max_features": 500,
            "scale_factor": 1.2,
            "n_levels": 8,
        },
        "matching": {
            "confidence_threshold": 0.3,
            "ratio_test_threshold": 0.75,
            "min_matches": 5,
        },
        "retrieval": {"top_k": 5},
        "camera": {
            "width": 640,
            "height": 480,
            "fx": 554.25,
            "fy": 554.25,
            "cx": 320.0,
            "cy": 240.0,
        },
        "geo_reference": {
            "origin_latitude": 33.7470,
            "origin_longitude": 73.1370,
            "origin_altitude": 550.0,
        },
    }


def _extract_entry(
    image: np.ndarray,
    entry_id: str,
    lat: float,
    lon: float,
    alt: float = 550.0,
    heading: float = 0.0,
) -> DatabaseEntry:
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    kp, des = orb.detectAndCompute(image, None)
    if kp is None or len(kp) == 0:
        kp_arr = np.empty((0, 2), dtype=np.float32)
        des = np.empty((0, 32), dtype=np.uint8)
    else:
        kp_arr = np.array([[k.pt[0], k.pt[1]] for k in kp], dtype=np.float32)
    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)
    return DatabaseEntry(
        id=entry_id,
        source_path="synthetic",
        latitude=lat,
        longitude=lon,
        altitude=alt,
        heading=heading,
        capture_time="2024-01-01",
        feature_count=len(kp_arr),
        feature_algorithm="ORB",
        keypoints=kp_arr,
        descriptors=des,
        metadata={"width": image.shape[1], "height": image.shape[0]},
    )


@pytest.fixture
def sample_entry(textured_image) -> DatabaseEntry:
    return _extract_entry(textured_image, "synth_001", 33.7470, 73.1370)


@pytest.fixture
def sample_database(textured_image) -> ReferenceDatabase:
    db = ReferenceDatabase(name="Test DB")
    db.add_entry(
        _extract_entry(textured_image, "synth_001", 33.7470, 73.1370)
    )
    shifted = np.roll(textured_image, 60, axis=1)
    db.add_entry(
        _extract_entry(shifted, "synth_002", 33.7475, 73.1375)
    )
    inverted = cv2.bitwise_not(textured_image)
    db.add_entry(
        _extract_entry(inverted, "synth_003", 33.7480, 73.1380)
    )
    return db

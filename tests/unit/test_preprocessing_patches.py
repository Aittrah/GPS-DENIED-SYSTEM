"""
Unit tests for the satellite preprocessing → patch pipeline.

Checks that patches:
  - are uint8 (not float)
  - have pixel values in [0, 255]
  - are exactly 256×256×3
  - contain real image variation (std > 10), not uniform noise
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Make sure project root is on sys.path when running via pytest from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIXTURE = Path(__file__).parent.parent / "fixtures" / "test_sat.jpg"


@pytest.fixture(scope="module")
def patches():
    import cv2
    from src.vns.preprocessing.satellite_preprocessor import SatellitePreprocessor
    from src.vns.preprocessing.patch_generator import PatchGenerator
    from src.vns.core.geo_utils import NEDPoint

    img = cv2.imread(str(FIXTURE))
    assert img is not None, f"Test fixture not found: {FIXTURE}"

    preprocessor = SatellitePreprocessor()
    processed = preprocessor.load_and_preprocess_array(img)

    pg = PatchGenerator()
    center = NEDPoint(0.0, 0.0, 0.0)
    patch_objs = pg.generate_patches(
        image=processed,
        image_center=center,
        meters_per_pixel=0.5,
        source="satellite",
        base_id="test",
    )
    return [p.data for p in patch_objs]


def test_patches_generated(patches):
    assert len(patches) > 0, "No patches were generated"


def test_patch_dtype_is_uint8(patches):
    for i, p in enumerate(patches):
        assert p.dtype == np.uint8, (
            f"Patch {i} dtype={p.dtype} — float normalization must NOT "
            "be applied before saving patches"
        )


def test_patch_pixel_range(patches):
    for i, p in enumerate(patches):
        assert p.min() >= 0,   f"Patch {i} has negative pixel values"
        assert p.max() <= 255, f"Patch {i} has pixel values > 255"


def test_patch_size(patches):
    for i, p in enumerate(patches):
        assert p.shape == (256, 256, 3), (
            f"Patch {i} shape={p.shape}, expected (256, 256, 3)"
        )


def test_no_noise(patches):
    """
    A real image patch has moderate std (10–80).
    Pure noise has std ≈ 80+.
    A completely uniform/blank patch has std ≈ 0.
    Both extremes indicate a pipeline bug.
    """
    for i, p in enumerate(patches):
        std = float(p.std())
        assert std > 10, (
            f"Patch {i} std={std:.1f} is too low — patch looks blank/uniform"
        )
        assert std < 120, (
            f"Patch {i} std={std:.1f} is too high — patch looks like pure noise"
        )


def test_preprocessor_returns_uint8():
    import cv2
    from src.vns.preprocessing.satellite_preprocessor import SatellitePreprocessor
    img = cv2.imread(str(FIXTURE))
    processed = SatellitePreprocessor().load_and_preprocess_array(img)
    assert processed.dtype == np.uint8
    assert processed.min() >= 0
    assert processed.max() <= 255
    assert processed.shape[2] == 3       # 3-channel BGR
    assert processed.shape[0] >= 256     # tall enough for at least one patch
    assert processed.shape[1] >= 256     # wide enough for at least one patch

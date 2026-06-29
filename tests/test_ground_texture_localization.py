"""Regression test for the ``no_geometric_match`` ground-texture bug.

The downward camera over the QAU campus must localize against the georeferenced
ground-texture mosaic. The original mosaic painted the 18 building tiles on top
of the five ``grid_*`` tiles (e.g. all four ``qau_admin_*`` tiles share the exact
world origin), which corrupted every grid tile and left no region matching any
single reference entry -- so geometric verification returned ``no_geometric_match``
and no pose fix was ever published.

``build_ground_texture.py`` now paints only the clean grid tiles. These tests
render the downward-camera view from the committed texture (no Gazebo) and assert
that every grid centre -- ``grid_center`` in particular -- localizes with a
geo-accurate pose. If the building overlays ever return, ``grid_center`` fails.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "simulation" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import simulate_camera_view as scv  # noqa: E402
from vns.database.reference_db import ReferenceDatabase  # noqa: E402
from vns.vision.localizer import VisualLocalizer  # noqa: E402

pytestmark = pytest.mark.skipif(
    not scv.DB_PATH.exists() or not scv.TEXTURE.exists(),
    reason="simulation database / ground-texture artifacts not present",
)

# Generous tolerance: a centred camera over a clean tile localizes to ~1-2 m;
# 10 m comfortably proves it matched the correct tile (pose is anchored to the
# matched entry's geotag, so a wrong match would be hundreds of metres off).
_MAX_POS_ERR_M = 10.0


@pytest.fixture(scope="module")
def localizer():
    cfg = yaml.safe_load(scv.CONFIG.read_text())
    db = ReferenceDatabase.load(str(scv.DB_PATH))
    return cfg, VisualLocalizer(cfg, db)


@pytest.fixture(scope="module")
def texture():
    tex = cv2.imread(str(scv.TEXTURE))
    assert tex is not None, f"could not read texture {scv.TEXTURE}"
    return tex


def _localize_grid_centre(cfg, loc, texture, lat, lon):
    xc, yc = scv.latlon_to_world(lat, lon)
    view = scv.render_ground_view(
        texture, xc, yc, scv.DEMO_H_AGL, 0.0, cfg["camera"]
    )
    return loc.localize(
        view, altitude=scv.GROUND_ALT_MSL + scv.DEMO_H_AGL, heading_deg=0.0
    )


def _pos_error_m(pose_geodetic, lat, lon):
    return float(
        np.hypot(
            (pose_geodetic[0] - lat) * scv.M_PER_DEG_LAT,
            (pose_geodetic[1] - lon) * scv.M_PER_DEG_LON,
        )
    )


def test_grid_center_localizes(localizer, texture):
    """The exact regression: the camera over the world origin must localize.

    grid_center was the tile the test drone hovered over, and the building
    overlays made it the worst-corrupted region -> no_geometric_match.
    """
    cfg, loc = localizer
    lat, lon = scv.GRID_CENTRES["grid_center"]
    res = _localize_grid_centre(cfg, loc, texture, lat, lon)

    assert res.success, f"grid_center did not localize: reason={res.reason!r}"
    assert res.matched_ref_id == "grid_center"
    assert res.pose_geodetic is not None
    assert _pos_error_m(res.pose_geodetic, lat, lon) < _MAX_POS_ERR_M


@pytest.mark.parametrize("name", list(scv.GRID_CENTRES))
def test_all_grid_centres_localize(localizer, texture, name):
    """Every grid centre yields a successful, geo-accurate fix."""
    cfg, loc = localizer
    lat, lon = scv.GRID_CENTRES[name]
    res = _localize_grid_centre(cfg, loc, texture, lat, lon)

    assert res.success, f"{name} did not localize: reason={res.reason!r}"
    assert res.confidence > 0.0
    assert res.pose_geodetic is not None
    err = _pos_error_m(res.pose_geodetic, lat, lon)
    assert err < _MAX_POS_ERR_M, f"{name} pose error {err:.1f} m exceeds tolerance"

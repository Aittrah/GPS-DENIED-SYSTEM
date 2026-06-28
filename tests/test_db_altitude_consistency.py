#!/usr/bin/env python3
"""
Regression guard for FR-11 (Visual-Based Localization): reference-database entry
altitudes must equal the configured geo_reference ground datum.

Background
----------
Pose recovery (`src/vns/vision/pose_recovery.py::displacement_to_pose`) scales pixel
displacement to metres with::

    height = drone_altitude_MSL - entry.altitude

i.e. it treats `entry.altitude` as the GROUND / TERRAIN MSL datum under the tile, not
the camera capture altitude. The database was once tagged with capture altitudes
(buildings 580/590 m, grid tiles 600 m, the overhead grid_center 620 m). That
under-estimates `height`, compresses displacement-to-metres, and biases off-center
pose recovery toward the matched tile (worst case ~29 % at the 240 m AGL demo).

These tests fail if any entry — in the build-input index OR in the built `.vnsdb`
that the runtime actually loads — drifts away from the geo_reference origin altitude
(~550 m MSL), so the capture-altitude regression cannot silently return.
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vns.database.reference_db import ReferenceDatabase  # noqa: E402

SIM_CONFIG = REPO_ROOT / "simulation" / "config" / "simulation.yaml"
DB_INDEX = REPO_ROOT / "simulation" / "database" / "images" / "database_index.yaml"
BUILT_DB = REPO_ROOT / "simulation" / "database" / "qau_campus.vnsdb"

# Tolerance around the ground datum. The synthetic campus ground is flat, so every
# tile should sit exactly on the datum; a tight band still excludes the old
# 580/590/600/620 m capture altitudes by a wide margin.
DATUM_TOLERANCE_M = 25.0
# The original-bug capture-altitude band that must never reappear.
BAD_CAPTURE_BAND = (575.0, 625.0)


def _origin_altitude() -> float:
    cfg = yaml.safe_load(SIM_CONFIG.read_text())
    return float(cfg["geo_reference"]["origin_altitude"])


def test_index_entries_match_ground_datum():
    """Every entry in the build-input index sits on the geo_reference datum."""
    origin_alt = _origin_altitude()
    index = yaml.safe_load(DB_INDEX.read_text())

    # The database center must itself be the datum.
    assert float(index["database"]["center"]["altitude"]) == origin_alt

    offenders = {
        img["id"]: img["altitude"]
        for img in index["images"]
        if abs(float(img["altitude"]) - origin_alt) > DATUM_TOLERANCE_M
    }
    assert not offenders, (
        f"database_index.yaml entries deviate from the {origin_alt} m ground datum "
        f"(capture-altitude regression?): {offenders}"
    )


def test_built_db_entries_match_ground_datum():
    """The built .vnsdb (what the runtime loads) sits on the geo_reference datum."""
    origin_alt = _origin_altitude()
    db = ReferenceDatabase.load(str(BUILT_DB))
    assert db.entries, "built database has no entries"

    offenders = {
        eid: e.altitude
        for eid, e in db.entries.items()
        if abs(float(e.altitude) - origin_alt) > DATUM_TOLERANCE_M
    }
    assert not offenders, (
        f"qau_campus.vnsdb entries deviate from the {origin_alt} m ground datum; "
        f"rebuild from the corrected index (capture-altitude regression?): {offenders}"
    )


def test_no_entry_in_capture_altitude_band():
    """No built-DB entry may carry an old 580-620 m capture altitude."""
    lo, hi = BAD_CAPTURE_BAND
    db = ReferenceDatabase.load(str(BUILT_DB))
    offenders = {
        eid: e.altitude for eid, e in db.entries.items() if lo < float(e.altitude) < hi
    }
    assert not offenders, (
        f"entries carry capture-altitude values in the forbidden {lo}-{hi} m band "
        f"instead of the ground datum: {offenders}"
    )

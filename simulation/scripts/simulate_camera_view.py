#!/usr/bin/env python3
"""Offline model of the downward camera over the textured campus ground.

Renders what the Gazebo downward camera would see by sampling the georeferenced
ground-texture mosaic (``build_ground_texture.py`` output), then runs the real
:class:`~vns.vision.localizer.VisualLocalizer` on it. No Gazebo / PX4 / ROS
needed, so it is deterministic and CI-friendly.

This is the harness that pinned down the ``no_geometric_match`` failures: the
localizer self-matches clean reference tiles perfectly, so the fault was in what
the camera saw on the ground, not in the matcher. With the building overlays
removed from the mosaic, a camera at ~240 m over any grid centre now sees a
pristine reference tile and localizes with a strong, geo-accurate fix.

Camera model: pinhole looking straight down. At altitude ``h_agl`` above the
ground the footprint is ``width_px * h_agl / fx`` metres wide, so ~240 m AGL maps
one 278 m grid tile onto the full 640x480 frame.

Run directly to print a localization report over the five grid centres:
    python simulate_camera_view.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from vns.database.reference_db import ReferenceDatabase  # noqa: E402
from vns.vision.localizer import VisualLocalizer  # noqa: E402

CONFIG = _REPO / "simulation/config/simulation.yaml"
DB_PATH = _REPO / "simulation/database/qau_campus.vnsdb"
# The real satellite texture the Gazebo ground plane (qau_ground_plane) renders and
# that the reference tiles are cut from -- so this offline render matches the live
# world and the committed DB (see build_real_reference_tiles.py).
TEXTURE = _REPO / "simulation/models/qau_ground_plane/materials/textures/qau_satellite.png"

# Must match qau_ground_plane/model.sdf + build_satellite_ground_texture.py: the
# real satellite texture covers a CANVAS_EW x CANVAS_NS box (metres, east x north)
# centred on the world/DB origin (lat 33.7470, lon 73.1370). It is NOT square.
CANVAS_M = 600.0            # east-west extent (kept name for back-compat)
CANVAS_EW = 600.0
CANVAS_NS = 450.6
ORIGIN_LAT = 33.7470
ORIGIN_LON = 73.1370
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(ORIGIN_LAT))

# Altitude (above ground) at which the 60-deg-FOV camera footprint (~277 m)
# equals one grid tile; ground sits at the world elevation (550 m MSL).
DEMO_H_AGL = 240.0
GROUND_ALT_MSL = 550.0


def latlon_to_world(lat: float, lon: float) -> Tuple[float, float]:
    """Geodetic -> local ENU metres about the texture/world origin."""
    return (
        (lon - ORIGIN_LON) * M_PER_DEG_LON,
        (lat - ORIGIN_LAT) * M_PER_DEG_LAT,
    )


def render_ground_view(
    texture: np.ndarray,
    x_cam: float,
    y_cam: float,
    h_agl: float,
    heading_deg: float,
    camera_cfg: Dict,
    canvas_m: float = CANVAS_M,
) -> np.ndarray:
    """Render the downward-camera image at world ``(x_cam, y_cam, h_agl)``.

    Samples the ground texture with the same ENU convention the texture is
    painted in (+x East -> +u, +y North -> -v). ``heading_deg`` rotates the
    camera about the vertical (clockwise from north).
    """
    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 480))
    fx = float(camera_cfg.get("fx", 554.25))
    fy = float(camera_cfg.get("fy", 554.25))
    cx = float(camera_cfg.get("cx", width / 2.0))
    cy = float(camera_cfg.get("cy", height / 2.0))

    # Per-axis metres->pixels: the real satellite texture is non-square
    # (CANVAS_EW x CANVAS_NS metres over shape[1] x shape[0] pixels).
    px_per_m_x = texture.shape[1] / CANVAS_EW
    px_per_m_y = texture.shape[0] / CANVAS_NS

    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    # camera pixel -> ground metres (heading 0): +u east, +v south
    gx = x_cam + (uu - cx) * h_agl / fx
    gy = y_cam - (vv - cy) * h_agl / fy

    if heading_deg:
        th = np.radians(heading_deg)
        c, s = np.cos(th), np.sin(th)
        rx = c * (gx - x_cam) - s * (gy - y_cam) + x_cam
        ry = s * (gx - x_cam) + c * (gy - y_cam) + y_cam
        gx, gy = rx, ry

    map_x = ((gx + CANVAS_EW / 2.0) * px_per_m_x).astype(np.float32)
    map_y = ((CANVAS_NS / 2.0 - gy) * px_per_m_y).astype(np.float32)
    return cv2.remap(
        texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


GRID_CENTRES = {
    "grid_center": (33.74700, 73.13700),
    "grid_nw": (33.74800, 73.13550),
    "grid_ne": (33.74800, 73.13850),
    "grid_sw": (33.74600, 73.13550),
    "grid_se": (33.74600, 73.13850),
}


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    db = ReferenceDatabase.load(str(DB_PATH))
    loc = VisualLocalizer(cfg, db)
    texture = cv2.imread(str(TEXTURE))
    if texture is None:
        print(f"ERROR: texture not found at {TEXTURE} "
              "(run build_ground_texture.py first)")
        return 1

    cam = cfg.get("camera", {})
    drone_alt = GROUND_ALT_MSL + DEMO_H_AGL
    print(f"Texture {texture.shape}, camera @ {DEMO_H_AGL} m AGL "
          f"({drone_alt} m MSL), footprint "
          f"{cam.get('width', 640) * DEMO_H_AGL / cam.get('fx', 554.25):.0f} m\n")

    failures = 0
    for name, (lat, lon) in GRID_CENTRES.items():
        xc, yc = latlon_to_world(lat, lon)
        view = render_ground_view(texture, xc, yc, DEMO_H_AGL, 0.0, cam)
        res = loc.localize(view, altitude=drone_alt, heading_deg=0.0)
        if res.success and res.pose_geodetic is not None:
            err_m = np.hypot(
                (res.pose_geodetic[0] - lat) * M_PER_DEG_LAT,
                (res.pose_geodetic[1] - lon) * M_PER_DEG_LON,
            )
            print(f"  {name:12s} OK   ref={res.matched_ref_id:12s} "
                  f"inliers={res.inlier_count:3d} conf={res.confidence:.2f} "
                  f"pos_err={err_m:5.1f} m")
        else:
            failures += 1
            print(f"  {name:12s} FAIL reason='{res.reason}'")

    print(f"\n{len(GRID_CENTRES) - failures}/{len(GRID_CENTRES)} grid centres localized.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Cut geo-referenced reference tiles from the REAL QAU satellite texture.

The Gazebo ``qau_ground_plane`` model textures a 600 m (E-W) x 450.6 m (N-S)
box, centered on the world origin (geo_reference: 33.7470 N, 73.1370 E), with
``materials/textures/qau_satellite.png`` (2048x1538, ~0.293 m/px) -- the cleaned
real satellite image produced by build_satellite_ground_texture.py.

This script tiles that SAME image into reference patches whose footprint matches
the downward camera (60 deg hfov @ 240 m -> ~277 m x 208 m -> 640x480), with one
tile centered exactly on the origin (== the localization_test static-camera
footprint). Because the reference tiles and the world texture are the same
imagery, ORB geometric verification actually matches -- unlike the synthetic
building crops in the legacy DB.

Output: <out_dir>/tile_*.jpg + <out_dir>/database_index.yaml, ready for
build_reference_database.py.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

# --- World / texture geo-registration (see uav_localization_test.sdf +
#     qau_ground_plane/model.sdf + build_satellite_ground_texture.py) ---
ORIGIN_LAT = 33.7470
ORIGIN_LON = 73.1370
ORIGIN_ALT_MSL = 550.0        # geo_reference.origin_altitude
BOX_EW_M = 600.0              # textured box size, east-west (image width)
BOX_NS_M = 450.6              # textured box size, north-south (image height)

# Camera footprint at the localization_test capture altitude (240 m, 60 deg hfov,
# 4:3). width = 2*h*tan(hfov/2); height = width * 480/640.
CAM_ALT_AGL = 240.0
CAM_HFOV_RAD = 1.047
OUT_W, OUT_H = 640, 480

M_PER_DEG_LAT = 111_320.0

# Canonical reference points (must match simulate_camera_view.GRID_CENTRES). These
# replace the old synthetic grid_*.jpg crops with REAL satellite crops at the same
# geo-tags and names, so both the offline localization test and the live Gazebo
# world verify against the same tiles (project task P4).
GRID_CENTRES = {
    "grid_center": (33.74700, 73.13700),
    "grid_nw": (33.74800, 73.13550),
    "grid_ne": (33.74800, 73.13850),
    "grid_sw": (33.74600, 73.13550),
    "grid_se": (33.74600, 73.13850),
}


def _m_per_deg_lon(lat_deg: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument(
        "--texture",
        default=str(repo / "simulation/models/qau_ground_plane/materials/textures/qau_satellite.png"),
        help="Real satellite texture that the ground plane renders.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(repo / "simulation/database/images_real"),
        help="Output directory for tile images + database_index.yaml.",
    )
    ap.add_argument("--cols", type=int, default=5, help="Tile grid columns (E-W).")
    ap.add_argument("--rows", type=int, default=5, help="Tile grid rows (N-S).")
    args = ap.parse_args()

    tex_path = Path(args.texture)
    img = cv2.imread(str(tex_path))
    if img is None:
        print(f"ERROR: cannot read texture {tex_path}")
        return 1
    h_px, w_px = img.shape[:2]
    mpp_x = BOX_EW_M / w_px          # metres per pixel, east-west
    mpp_y = BOX_NS_M / h_px          # metres per pixel, north-south
    cx_px, cy_px = w_px / 2.0, h_px / 2.0

    # Camera footprint (metres) -> crop size (pixels) in the texture.
    foot_ew = 2.0 * CAM_ALT_AGL * math.tan(CAM_HFOV_RAD / 2.0)   # ~277 m
    foot_ns = foot_ew * (OUT_H / OUT_W)                          # ~208 m
    crop_w = int(round(foot_ew / mpp_x))
    crop_h = int(round(foot_ns / mpp_y))

    # Grid of tile centres (metres from origin). Include 0 so one tile is
    # centred exactly on the camera footprint. Keep centres inside the box so
    # crops stay within the image.
    max_e = (BOX_EW_M - foot_ew) / 2.0
    max_n = (BOX_NS_M - foot_ns) / 2.0

    def _linspace(n: int, half: float) -> list[float]:
        if n <= 1:
            return [0.0]
        return [(-half + 2 * half * i / (n - 1)) for i in range(n)]

    east_offsets = _linspace(args.cols, max_e)
    north_offsets = _linspace(args.rows, max_n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_per_deg_lon = _m_per_deg_lon(ORIGIN_LAT)

    images = []
    now = datetime.now(timezone.utc).isoformat()
    bounds = {"min_lat": math.inf, "min_lon": math.inf,
              "max_lat": -math.inf, "max_lon": -math.inf}

    def emit(east_m: float, north_m: float, tile_id: str) -> None:
        px = cx_px + east_m / mpp_x
        py = cy_px - north_m / mpp_y                     # image y down = south
        x0 = int(round(px - crop_w / 2.0))
        y0 = int(round(py - crop_h / 2.0))
        x0 = max(0, min(x0, w_px - crop_w))
        y0 = max(0, min(y0, h_px - crop_h))
        crop = img[y0:y0 + crop_h, x0:x0 + crop_w]
        tile = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        lat = ORIGIN_LAT + north_m / M_PER_DEG_LAT
        lon = ORIGIN_LON + east_m / m_per_deg_lon
        fname = f"{tile_id}.jpg"
        cv2.imwrite(str(out_dir / fname), tile)
        images.append({
            "id": tile_id, "filepath": fname,
            "latitude": round(lat, 8), "longitude": round(lon, 8),
            "altitude": ORIGIN_ALT_MSL, "heading": 0,
            "width": OUT_W, "height": OUT_H, "timestamp": now,
        })
        bounds["min_lat"] = min(bounds["min_lat"], lat)
        bounds["max_lat"] = max(bounds["max_lat"], lat)
        bounds["min_lon"] = min(bounds["min_lon"], lon)
        bounds["max_lon"] = max(bounds["max_lon"], lon)

    # Canonical named reference tiles at the exact grid coordinates the
    # localization tests probe (real crops replacing the old synthetic grid_*.jpg).
    named_en = []
    for name, (lat, lon) in GRID_CENTRES.items():
        e = (lon - ORIGIN_LON) * m_per_deg_lon
        n = (lat - ORIGIN_LAT) * M_PER_DEG_LAT
        named_en.append((e, n))
        emit(e, n, name)

    # Regular covering grid for broader live coverage over the textured box.
    # Skip cells that (nearly) coincide with a named tile so the named tile is the
    # unambiguous match at those coordinates (e.g. grid_center at the origin).
    dedup_m = 45.0
    for r, north_m in enumerate(reversed(north_offsets)):   # row 0 = north
        for c, east_m in enumerate(east_offsets):
            if any(math.hypot(east_m - e, north_m - n) < dedup_m for e, n in named_en):
                continue
            emit(east_m, north_m, f"qau_tile_r{r}_c{c}")

    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]

    index = {
        "database": {
            "name": "QAU Campus Reference Database (real satellite tiles)",
            "location": "QAU Campus, Islamabad, Pakistan",
            "source_type": "real_satellite",
            "version": "2.0.0",
            "created": now,
            "image_count": len(images),
            "center": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON, "altitude": ORIGIN_ALT_MSL},
            "bounds": {
                "min_latitude": round(min_lat, 8), "max_latitude": round(max_lat, 8),
                "min_longitude": round(min_lon, 8), "max_longitude": round(max_lon, 8),
            },
        },
        "images": images,
    }
    with (out_dir / "database_index.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(index, fh, sort_keys=False)

    print(f"Texture: {tex_path} ({w_px}x{h_px}, {mpp_x:.4f} m/px)")
    print(f"Camera footprint: {foot_ew:.1f} m x {foot_ns:.1f} m -> crop {crop_w}x{crop_h}px -> {OUT_W}x{OUT_H}")
    print(f"Wrote {len(images)} tiles ({len(GRID_CENTRES)} named + {args.rows}x{args.cols} grid) to {out_dir}")
    print(f"Bounds: lat [{min_lat:.6f}, {max_lat:.6f}]  lon [{min_lon:.6f}, {max_lon:.6f}]")
    print(f"Index:  {out_dir / 'database_index.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

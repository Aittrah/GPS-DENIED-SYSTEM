#!/usr/bin/env python3
"""
Build a georeferenced ground texture for the Gazebo world from the VNS
reference-database tiles.

The VNS reference database (simulation/database/images) is *synthetic*: each
tile is an independent 640x480 image (grid + heading arrow + label + ORB-feature
blobs) tagged with a lat/lon. A real satellite image would not match these
tiles, so to make the drone's downward camera see terrain that the localizer
can actually match, we paint the tiles themselves onto the ground at their true
world positions.

World frame: the world's <spherical_coordinates> origin equals the database
center (lat 33.7470, lon 73.1370). Gazebo ENU => X=East, Y=North, origin (0,0).

Output: a single PNG mapped 1:1 onto a CANVAS_M x CANVAS_M ground box centred at
the origin, so world (x,y) in [-CANVAS_M/2, CANVAS_M/2] maps onto the texture.

Only the five ``grid_*`` tiles are painted. They form a clean, geo-correct base
map: the four corners tile the campus and ``grid_center`` is painted last over
their inner halves, so a downward camera at ~240 m over any grid centre sees a
pristine copy of one reference tile and the localizer gets a strong match.

The 18 building tiles are NOT painted. They are co-located with the grid tiles
(e.g. all four ``qau_admin_*`` tiles share the exact origin), so overlaying their
96 m patches onto the grid corrupted every grid tile and left no region that
matched any single reference entry -- that was the root cause of the
``no_geometric_match`` failures. The building entries still live in the database
for retrieval richness; they are simply not rendered onto the ground.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path("/home/hp/GPS-DENIED-SYSTEM")
INDEX = REPO / "simulation/database/images/database_index.yaml"
IMAGES_DIR = REPO / "simulation/database/images"
OUT_PNG = REPO / "simulation/worlds/materials/textures/qau_campus_map.png"

# World / DB origin (must match world <spherical_coordinates>)
ORIGIN_LAT = 33.7470
ORIGIN_LON = 73.1370

CANVAS_M = 600.0          # ground box is CANVAS_M x CANVAS_M metres
TEX_PX = 2048             # texture resolution (power of two for Ogre)
PX_PER_M = TEX_PX / CANVAS_M

# Footprint each grid tile is painted at (metres) == grid spacing, so the four
# corners tile the campus edge-to-edge and grid_center overlaps their inner
# halves. Building tiles are intentionally not painted (see module docstring).
GRID_FOOTPRINT = (278.0, 222.0)     # (E-W, N-S) == grid spacing

M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(ORIGIN_LAT))

BG = (118, 120, 116)      # BGR mid terrain grey, blends with synthetic tiles


def latlon_to_world(lat: float, lon: float) -> tuple[float, float]:
    x_east = (lon - ORIGIN_LON) * M_PER_DEG_LON
    y_north = (lat - ORIGIN_LAT) * M_PER_DEG_LAT
    return x_east, y_north


def world_to_px(x: float, y: float) -> tuple[int, int]:
    # East -> +u (right); North -> -v (up).
    u = (x + CANVAS_M / 2.0) * PX_PER_M
    v = (CANVAS_M / 2.0 - y) * PX_PER_M
    return int(round(u)), int(round(v))


def paste_centered(canvas: np.ndarray, tile: np.ndarray, cx: int, cy: int,
                   w_px: int, h_px: int) -> None:
    tile = cv2.resize(tile, (w_px, h_px), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - w_px // 2, cy - h_px // 2
    x1, y1 = x0 + w_px, y0 + h_px
    # Clip to canvas
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    tx0, ty0 = cx0 - x0, cy0 - y0
    canvas[cy0:cy1, cx0:cx1] = tile[ty0:ty0 + (cy1 - cy0), tx0:tx0 + (cx1 - cx0)]


def main() -> None:
    with open(INDEX) as f:
        index = yaml.safe_load(f)

    canvas = np.zeros((TEX_PX, TEX_PX, 3), dtype=np.uint8)
    canvas[:, :] = BG

    # Paint only the grid_* tiles -- a clean, geo-correct base map. Building
    # tiles are co-located with the grids and would corrupt them, so they are
    # skipped (see module docstring). Painting order nw, ne, sw, se, center
    # leaves grid_center pristine on top of the corners' inner halves.
    grid_order = ["grid_nw", "grid_ne", "grid_sw", "grid_se", "grid_center"]
    by_id = {im["id"]: im for im in index["images"]}

    placed = []
    for tile_id in grid_order:
        im = by_id.get(tile_id)
        if im is None:
            print(f"  WARN no index entry for {tile_id}")
            continue
        path = IMAGES_DIR / Path(im["filepath"]).name
        tile = cv2.imread(str(path))
        if tile is None:
            print(f"  WARN missing {path}")
            continue
        x, y = latlon_to_world(im["latitude"], im["longitude"])
        u, v = world_to_px(x, y)
        fw, fh = GRID_FOOTPRINT
        paste_centered(canvas, tile, u, v, int(fw * PX_PER_M), int(fh * PX_PER_M))
        placed.append((im["id"], x, y, u, v))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), canvas)

    print(f"Wrote {OUT_PNG} ({TEX_PX}x{TEX_PX}, {CANVAS_M} m, {PX_PER_M:.2f} px/m)")
    print(f"Placed {len(placed)} tiles. Sample world positions (id: E,N m):")
    for tid, x, y, u, v in placed[:8]:
        print(f"  {tid:16s} x={x:7.1f} y={y:7.1f}  px=({u},{v})")
    print(f"M_PER_DEG_LON={M_PER_DEG_LON:.1f}  (cos lat applied)")


if __name__ == "__main__":
    main()

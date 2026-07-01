#!/usr/bin/env python3
"""
Build the QAU ground-plane texture from the real campus satellite photo.

The only satellite image in the repo (``data/satellite/raw/qau_campus_satellite.jpg``)
is a raw Google Earth Pro screenshot, not a clean export: it has the desktop app's
toolbar baked across the top, a small compass badge, a translucent route/waypoint
overlay, a placemark pin + name/coordinate label sitting over campus center, a
secondary placemark pin (sports field), and copyright/wordmark text in the bottom
band. Projected onto the Gazebo ground as-is, those UI elements would show up as
fake, non-physical "features" to the downward camera / ORB matcher.

This script removes the *opaque* artifacts (toolbar, compass badge, label/pin,
copyright, wordmark) via a top-strip crop, ``cv2.inpaint`` for small isolated
blobs (badge/pins), and direct row-band cloning for the two label text lines.
Text turned out to be too dense for inpainting: characters are tightly and
evenly spaced, so cv2.inpaint has no clean local neighborhood to interpolate
from and produces a visible blurred streak across the whole line. Cloning --
copying the pixels from a verified-clean band a fixed offset above the text,
i.e. a "clone stamp" -- uses only real, unaltered satellite pixels from a few
metres away in the same image, so it does not fabricate new ground content.
This leaves the translucent route/waypoint polyline (rows ~90-560, a thin
semi-transparent overlay that threads through real terrain) as a known,
documented limitation: inpainting or cloning over a winding path that size
risks worse artifacts than the thin line itself.

World frame: the world's <spherical_coordinates> origin (lat 33.7470, lon 73.1370)
is the same point the original screenshot's placemark label names, so the cropped
texture is mapped 1:1 onto a box centred on the world origin. Removing only the
top toolbar rows (not a symmetric top+bottom crop) shifts the image's vertical
center by ~9 m versus the placemark -- accepted as negligible for a simulation
ground texture.

Output: a single PNG, aspect ratio preserved from the cleaned source (no stretch
distortion), long side mapped to LONG_SIDE_M metres, written to
simulation/models/qau_ground_plane/materials/textures/qau_satellite.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_JPG = _REPO_ROOT / "data" / "satellite" / "raw" / "qau_campus_satellite.jpg"
OUT_PNG = (
    _REPO_ROOT
    / "simulation"
    / "models"
    / "qau_ground_plane"
    / "materials"
    / "textures"
    / "qau_satellite.png"
)

# Rows 0-24 are the Google Earth Pro toolbar (uniform bright band, full width) --
# sliced off entirely rather than inpainted.
TOOLBAR_ROWS = (0, 25)

# Bounding boxes for the artifacts to remove, each (x0, x1, y0, y1) in source
# pixel coordinates, found by visual inspection of the 755x592 source image.
# Within each box only the artifact pixels (bright text/badge strokes, or the
# pin's saturated hue) are masked, not the full rectangle, so cv2.inpaint fills
# thin strokes from real neighboring terrain instead of smearing a large flat
# patch (which produced visible directional streak artifacts when the whole
# bounding rectangle was masked).
WHITE_TEXT_REGIONS: tuple[tuple[int, int, int, int], ...] = (
    (680, 715, 22, 50),    # compass "N" badge, top right
    (290, 450, 522, 545),  # "Image (c) 2026 Airbus" copyright text
    (670, 755, 565, 592),  # "Google Earth" wordmark, bottom right
)

# Label text lines are dense enough that inpainting streaks (see module
# docstring); clone each line from a verified-clean band CLONE_SHIFT_PX rows
# above instead. Each entry is (x0, x1, y_dst0, y_dst1).
CLONE_BANDS: tuple[tuple[int, int, int, int], ...] = (
    (260, 620, 260, 286),  # "Quaid e Azam University Islamabad" label, line 1
    (260, 620, 288, 308),  # "33.7470, 73.1370" coordinate label, line 2
)
CLONE_SHIFT_PX = 60
# Label text is rendered white-fill with a dark outline/drop-shadow for
# legibility over varied terrain, so both extremes need masking -- bright fill
# and dark outline -- or the outline is left behind as ghost text. Boxes above
# are deliberately cropped tight to just the text glyph rows: real terrain
# (tree shadows, dark roofs) easily has the same dark/bright extremes, so a
# loose box over real content gets over-masked into a large blurry patch.
WHITE_TEXT_GRAY_THRESHOLD = 185
DARK_OUTLINE_GRAY_THRESHOLD = 70

# (x0, x1, y0, y1, hsv_low, hsv_high)
PIN_REGIONS_HSV: tuple[tuple[int, int, int, int, tuple, tuple], ...] = (
    (370, 470, 300, 345, (20, 90, 90), (35, 255, 255)),   # yellow campus pin
    (0, 40, 365, 405, (40, 60, 60), (85, 255, 255)),      # green sports-field pin
    (155, 185, 383, 410, (85, 40, 60), (110, 255, 255)),  # teal university placemark icon
)

MASK_DILATE_PX = 2
INPAINT_RADIUS_PX = 4

# Long edge of the cleaned image maps to this many metres, matching the existing
# campus ground-box convention in simulation/scripts/build_ground_texture.py.
LONG_SIDE_M = 600.0
LONG_SIDE_PX = 2048  # output texture resolution on the long edge (power-of-two-ish for Ogre)


def _artifact_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for x0, x1, y0, y1 in WHITE_TEXT_REGIONS:
        roi = gray[y0:y1, x0:x1]
        bright_or_dark = (roi > WHITE_TEXT_GRAY_THRESHOLD) | (roi < DARK_OUTLINE_GRAY_THRESHOLD)
        mask[y0:y1, x0:x1] |= bright_or_dark.astype(np.uint8) * 255

    for x0, x1, y0, y1, lo, hi in PIN_REGIONS_HSV:
        roi = hsv[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= cv2.inRange(roi, lo, hi)

    kernel = np.ones((MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1), np.uint8)
    return cv2.dilate(mask, kernel)


def _clone_text_bands(img: np.ndarray) -> np.ndarray:
    cloned = img.copy()
    for x0, x1, y0, y1 in CLONE_BANDS:
        src0, src1 = y0 - CLONE_SHIFT_PX, y1 - CLONE_SHIFT_PX
        cloned[y0:y1, x0:x1] = img[src0:src1, x0:x1]
    return cloned


def _clean_source(img: np.ndarray) -> np.ndarray:
    cloned = _clone_text_bands(img)
    mask = _artifact_mask(cloned)
    cleaned = cv2.inpaint(cloned, mask, INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    top, bottom = TOOLBAR_ROWS
    return cleaned[bottom:, :]


def main() -> int:
    if not SOURCE_JPG.exists():
        print(
            f"Source satellite image not found: {SOURCE_JPG}\n"
            "This script only cleans an existing real photo -- it will not "
            "fabricate a substitute. Place a real QAU campus satellite/aerial "
            "image at this path and re-run.",
            file=sys.stderr,
        )
        return 1

    img = cv2.imread(str(SOURCE_JPG))
    if img is None:
        print(f"Failed to load (corrupt or unsupported): {SOURCE_JPG}", file=sys.stderr)
        return 1

    cleaned = _clean_source(img)
    h, w = cleaned.shape[:2]

    long_side_px = max(h, w)
    scale = LONG_SIDE_PX / long_side_px
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    texture = cv2.resize(cleaned, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    m_per_px = LONG_SIDE_M / max(out_w, out_h)
    footprint_e_m = out_w * m_per_px
    footprint_n_m = out_h * m_per_px

    mean_color_bgr = cleaned.reshape(-1, 3).mean(axis=0)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), texture)

    print(f"Source: {SOURCE_JPG} ({img.shape[1]}x{img.shape[0]})")
    n_inpainted = len(WHITE_TEXT_REGIONS) + len(PIN_REGIONS_HSV)
    print(
        f"Cleaned (toolbar cropped, {len(CLONE_BANDS)} text band(s) cloned, "
        f"{n_inpainted} region(s) inpainted): {w}x{h}"
    )
    print(f"Wrote {OUT_PNG} ({out_w}x{out_h} px, {m_per_px:.4f} m/px)")
    print(f"Footprint: {footprint_e_m:.1f} m (E-W) x {footprint_n_m:.1f} m (N-S)")
    print(
        "Fallback base-plane color (BGR, 0-255): "
        f"{mean_color_bgr[0]:.1f} {mean_color_bgr[1]:.1f} {mean_color_bgr[2]:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

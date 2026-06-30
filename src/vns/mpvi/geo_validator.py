"""Geo-reference validation for satellite images."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_GEOTIFF_SUFFIXES = {".tif", ".tiff", ".geotiff"}
_RASTER_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_image(path: Path) -> dict[str, Any]:
    """Validate a satellite image file and extract geo-reference metadata.

    Returns a dict with keys:
        valid (bool)        — file is a supported format and readable
        format (str)        — "GeoTIFF", "JPEG", "PNG", or "unknown"
        has_georef (bool)   — CRS and bounding box are present
        bbox (tuple|None)   — (west, south, east, north) in WGS84 degrees
        crs (str|None)      — CRS string e.g. "EPSG:4326"
        warning (str|None)  — advisory message (non-fatal)
        error (str|None)    — error description when valid=False
    """
    result: dict[str, Any] = {
        "valid": False,
        "format": "unknown",
        "has_georef": False,
        "bbox": None,
        "crs": None,
        "warning": None,
        "error": None,
    }

    suffix = path.suffix.lower()

    if suffix in _GEOTIFF_SUFFIXES:
        _validate_geotiff(path, result)
    elif suffix in _RASTER_SUFFIXES:
        _validate_raster(path, suffix, result)
    else:
        result["error"] = f"Unsupported file format: '{suffix}'. Accepted: .tif .tiff .geotiff .jpg .jpeg .png"

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_geotiff(path: Path, result: dict[str, Any]) -> None:
    result["format"] = "GeoTIFF"
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            crs = src.crs
            bounds = src.bounds  # BoundingBox(left, bottom, right, top)
            if crs is None:
                result["valid"] = True
                result["warning"] = "GeoTIFF has no CRS — image will be loaded without geo-reference."
                return
            result["has_georef"] = True
            result["crs"] = crs.to_string()
            # bounds are in the file's native CRS; reproject to WGS84 if needed
            try:
                from rasterio.warp import transform_bounds  # type: ignore
                west, south, east, north = transform_bounds(
                    crs, "EPSG:4326",
                    bounds.left, bounds.bottom, bounds.right, bounds.top,
                )
                result["bbox"] = (west, south, east, north)
            except Exception:
                # If reprojection fails, store native bounds
                result["bbox"] = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            result["valid"] = True
    except Exception as exc:
        result["error"] = f"Cannot read GeoTIFF: {exc}"


def _validate_raster(path: Path, suffix: str, result: dict[str, Any]) -> None:
    fmt_map = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    result["format"] = fmt_map.get(suffix, "unknown")
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as img:
            img.verify()  # raises if file is corrupt
        result["valid"] = True
        result["has_georef"] = False
        result["warning"] = (
            f"{result['format']} images have no embedded geo-reference. "
            "Image will be loaded without coordinate metadata."
        )
    except Exception as exc:
        result["error"] = f"Cannot read image: {exc}"

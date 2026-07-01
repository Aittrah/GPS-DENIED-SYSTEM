#!/usr/bin/env python3
"""Stdlib-only helpers for open-house diagnostics."""

from __future__ import annotations

import ast
import json
import math
import struct
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Load JSONL rows, ignoring malformed trailing lines."""
    rows: list[dict[str, Any]] = []
    invalid_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, invalid_lines


def _parse_npy_header(blob: bytes) -> tuple[dict[str, Any], int]:
    if blob[:6] != b"\x93NUMPY":
        raise ValueError("Unsupported .npy header.")

    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        start = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", blob[8:12])[0]
        start = 12
    else:
        raise ValueError(f"Unsupported .npy version: {major}")

    header_text = blob[start : start + header_len].decode("latin1").strip()
    header = ast.literal_eval(header_text)
    if not isinstance(header, dict):
        raise ValueError("Invalid .npy header payload.")
    return header, start + header_len


def _shape_size(shape: Any) -> int:
    if shape in {(), None}:
        return 1
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def load_npy_blob(blob: bytes) -> Any:
    """Load the limited .npy dtypes this task needs."""
    header, offset = _parse_npy_header(blob)
    if header.get("fortran_order"):
        raise ValueError("Fortran-order arrays are not supported.")

    descr = str(header["descr"])
    shape = header.get("shape", ())
    count = _shape_size(shape)
    data = blob[offset:]

    if descr.startswith("<U"):
        text = data.decode("utf-32le").rstrip("\x00")
        return text

    if descr in {"<f4", "<f8"}:
        width = 4 if descr == "<f4" else 8
        code = "f" if descr == "<f4" else "d"
        expected = count * width
        values = [item[0] for item in struct.iter_unpack(f"<{code}", data[:expected])]
        return values if shape != () else values[0]

    if descr in {"|u1", "<u1"}:
        values = list(data[:count])
        return values if shape != () else values[0]

    raise ValueError(f"Unsupported .npy dtype: {descr}")


def load_database_metadata(path: str | Path) -> dict[str, Any]:
    db_path = Path(path)
    with zipfile.ZipFile(db_path) as archive:
        metadata = load_npy_blob(archive.read("__metadata__.npy"))
    if not isinstance(metadata, str):
        raise ValueError("Database metadata is not a string payload.")
    payload = json.loads(metadata)
    if not isinstance(payload, dict):
        raise ValueError("Database metadata JSON is not an object.")
    return payload


def resolve_db_source_path(db_path: str | Path, source_path: str | Path) -> Path:
    source = Path(source_path)
    if source.is_absolute():
        return source
    return Path(db_path).resolve().parent / source


def load_bovw_histograms(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], list[float]]]:
    db_path = Path(path)
    meta = metadata or load_database_metadata(db_path)
    entries = meta.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Database metadata is missing its entries list.")

    loaded: list[tuple[dict[str, Any], list[float]]] = []
    with zipfile.ZipFile(db_path) as archive:
        archive_names = set(archive.namelist())
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            histogram_key = entry.get("bovw_histogram_key")
            if not isinstance(histogram_key, str):
                continue
            archive_key = histogram_key if histogram_key in archive_names else f"{histogram_key}.npy"
            histogram = load_npy_blob(archive.read(archive_key))
            if not isinstance(histogram, list):
                raise ValueError(f"Histogram {histogram_key} is malformed.")
            loaded.append((entry, [float(value) for value in histogram]))
    return loaded


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


@dataclass(frozen=True)
class PngChannelStats:
    width: int
    height: int
    mean_r: float
    mean_g: float
    mean_b: float
    green_pixel_ratio: float

    @property
    def is_green_dominant(self) -> bool:
        return self.green_pixel_ratio > 0.55 or (
            self.mean_g > self.mean_r * 1.15 and self.mean_g > self.mean_b * 1.15
        )


def png_channel_stats(path: str | Path) -> PngChannelStats:
    """Compute simple RGB stats from a non-interlaced 8-bit PNG."""
    png_path = Path(path)
    raw = png_path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{png_path} is not a PNG file.")

    offset = 8
    width = height = bit_depth = color_type = interlace = None
    idat_chunks: list[bytes] = []
    while offset < len(raw):
        chunk_len = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_data = raw[offset + 8 : offset + 8 + chunk_len]
        offset += 12 + chunk_len
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"{png_path} is missing IHDR data.")
    if bit_depth != 8 or interlace != 0:
        raise ValueError(f"{png_path} uses an unsupported PNG mode.")

    if color_type == 2:
        bytes_per_pixel = 3
    elif color_type == 6:
        bytes_per_pixel = 4
    elif color_type == 0:
        bytes_per_pixel = 1
    else:
        raise ValueError(f"{png_path} uses unsupported color type {color_type}.")

    row_size = width * bytes_per_pixel
    data = zlib.decompress(b"".join(idat_chunks))
    prev = bytearray(row_size)
    cursor = 0
    sum_r = 0
    sum_g = 0
    sum_b = 0
    green_pixels = 0
    pixel_count = width * height

    for _ in range(height):
        filter_type = data[cursor]
        cursor += 1
        row = bytearray(data[cursor : cursor + row_size])
        cursor += row_size

        for index in range(row_size):
            left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = prev[index]
            up_left = prev[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                pass
            elif filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + up) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (row[index] + _paeth(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"{png_path} uses unknown PNG filter {filter_type}.")

        for pixel in range(width):
            base = pixel * bytes_per_pixel
            if color_type == 0:
                red = green = blue = row[base]
            else:
                red = row[base]
                green = row[base + 1]
                blue = row[base + 2]
            sum_r += red
            sum_g += green
            sum_b += blue
            if green > 80 and green > red * 1.10 and green > blue * 1.10:
                green_pixels += 1

        prev = row

    return PngChannelStats(
        width=width,
        height=height,
        mean_r=sum_r / pixel_count,
        mean_g=sum_g / pixel_count,
        mean_b=sum_b / pixel_count,
        green_pixel_ratio=green_pixels / pixel_count,
    )

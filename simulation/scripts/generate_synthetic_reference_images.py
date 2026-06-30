#!/usr/bin/env python3
"""Generate the legacy synthetic reference-image baseline for offline tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zlib

import cv2
import numpy as np

from reference_image_utils import (
    CapturedImage,
    generate_database_index,
    load_reference_config,
    utc_now_iso,
    validate_coverage,
)


def resolve_cli_path(path_value: str, *, script_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (script_dir / candidate).resolve()


def create_synthetic_image(
    latitude: float,
    longitude: float,
    heading: float,
    label: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    lat_norm = (latitude - 33.74) / 0.02
    lon_norm = (longitude - 73.13) / 0.02

    r = int(128 + 127 * lat_norm) % 256
    g = int(128 + 127 * lon_norm) % 256
    b = int(128 + 127 * (lat_norm + lon_norm) / 2.0) % 256

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [b, g, r]

    for x in range(0, width, 40):
        cv2.line(image, (x, 0), (x, height), (255, 255, 255), 1)
    for y in range(0, height, 40):
        cv2.line(image, (0, y), (width, y), (255, 255, 255), 1)

    center_x, center_y = width // 2, height // 2
    heading_rad = np.radians(heading)
    arrow_len = 50
    end_x = int(center_x + arrow_len * np.sin(heading_rad))
    end_y = int(center_y - arrow_len * np.cos(heading_rad))
    cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)

    cv2.putText(
        image,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"({latitude:.4f}, {longitude:.4f})",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    np.random.seed(zlib.adler32(label.encode("utf-8")))
    for _ in range(20):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(80, height - 50)
        radius = np.random.randint(5, 20)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(image, (x, y), radius, color, -1)

    return image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic reference images for offline VNS tests."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../database/qau_reference_metadata.yaml",
        help="Path to reference metadata YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../database/images/synthetic",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing synthetic output directory",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = resolve_cli_path(args.config, script_dir=script_dir)
    output_dir = resolve_cli_path(args.output, script_dir=script_dir)

    config, reference_points = load_reference_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        collisions = [
            output_dir / f"{point.id}.jpg"
            for point in reference_points
            if (output_dir / f"{point.id}.jpg").exists()
        ]
        if collisions:
            print(
                "Refusing to overwrite existing synthetic reference images. "
                "Use --overwrite or choose a fresh --output directory."
            )
            return 1

    captured_images: list[CapturedImage] = []
    for index, point in enumerate(reference_points, start=1):
        print(f"[{index}/{len(reference_points)}] generating {point.id}")
        image = create_synthetic_image(
            point.latitude,
            point.longitude,
            point.heading,
            point.id,
            width=args.width,
            height=args.height,
        )
        filepath = output_dir / f"{point.id}.jpg"
        if not cv2.imwrite(str(filepath), image):
            print(f"Failed to write {filepath}", file=sys.stderr)
            return 1
        captured_images.append(
            CapturedImage(
                id=point.id,
                filepath=str(filepath),
                latitude=point.latitude,
                longitude=point.longitude,
                altitude=point.altitude,
                heading=point.heading,
                timestamp=utc_now_iso(),
                width=image.shape[1],
                height=image.shape[0],
                extra={
                    "description": point.description,
                    **point.extra,
                    "source_type": "synthetic",
                    "capture_status": "synthetic_only",
                    "manual_capture_required": True,
                    "generator": "generate_synthetic_reference_images.py",
                },
            )
        )

    index_path = generate_database_index(
        captured_images,
        output_dir,
        config,
        database_source_type="synthetic",
    )
    coverage = validate_coverage(captured_images, config)

    print(f"Database index saved to: {index_path}")
    print(f"Coverage: {coverage}")
    print(f"Synthetic reference images saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Reference Image Capture Script for VNS Simulation

This script captures geotagged reference images from the Gazebo simulation
environment for building the visual navigation reference database.

Usage:
    python capture_reference_images.py --config qau_reference_metadata.yaml --output ./images

Requirements:
    - ROS2 Humble
    - Gazebo simulation running with VNS drone
    - Camera topic publishing
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


@dataclass
class ReferencePoint:
    """A reference point for image capture."""
    id: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    description: str


@dataclass
class CapturedImage:
    """Metadata for a captured reference image."""
    id: str
    filepath: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    timestamp: str
    width: int
    height: int


def load_reference_points(config_path: str) -> List[ReferencePoint]:
    """Load reference points from YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    points = []
    for point_data in config.get('reference_points', []):
        points.append(ReferencePoint(
            id=point_data['id'],
            latitude=point_data['latitude'],
            longitude=point_data['longitude'],
            altitude=point_data['altitude'],
            heading=point_data['heading'],
            description=point_data.get('description', '')
        ))
    
    return points


def create_synthetic_image(point: ReferencePoint, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic reference image for testing.
    
    In actual use, this would capture from Gazebo camera topic.
    For offline testing, generates a distinctive pattern based on location.
    """
    # Create base image with location-based color
    lat_norm = (point.latitude - 33.74) / 0.02
    lon_norm = (point.longitude - 73.13) / 0.02
    
    # Generate distinctive colors based on position
    r = int(128 + 127 * lat_norm) % 256
    g = int(128 + 127 * lon_norm) % 256
    b = int(128 + 127 * (lat_norm + lon_norm) / 2) % 256
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [b, g, r]
    
    # Add grid pattern for feature detection
    grid_spacing = 40
    for i in range(0, width, grid_spacing):
        cv2.line(image, (i, 0), (i, height), (255, 255, 255), 1)
    for j in range(0, height, grid_spacing):
        cv2.line(image, (0, j), (width, j), (255, 255, 255), 1)
    
    # Add distinctive markers based on heading
    center_x, center_y = width // 2, height // 2
    heading_rad = np.radians(point.heading)
    arrow_len = 50
    end_x = int(center_x + arrow_len * np.sin(heading_rad))
    end_y = int(center_y - arrow_len * np.cos(heading_rad))
    cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
    
    # Add location text
    text = f"{point.id}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    coords = f"({point.latitude:.4f}, {point.longitude:.4f})"
    cv2.putText(image, coords, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add some random features for ORB detection
    np.random.seed(hash(point.id) % (2**32))
    for _ in range(20):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(80, height - 50)
        size = np.random.randint(5, 20)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(image, (x, y), size, color, -1)
    
    return image


def save_image_with_metadata(
    image: np.ndarray,
    point: ReferencePoint,
    output_dir: Path
) -> CapturedImage:
    """Save image and return metadata."""
    timestamp = datetime.utcnow().isoformat()
    filename = f"{point.id}.jpg"
    filepath = output_dir / filename
    
    cv2.imwrite(str(filepath), image)
    
    return CapturedImage(
        id=point.id,
        filepath=str(filepath),
        latitude=point.latitude,
        longitude=point.longitude,
        altitude=point.altitude,
        heading=point.heading,
        timestamp=timestamp,
        width=image.shape[1],
        height=image.shape[0]
    )


def generate_database_index(
    captured_images: List[CapturedImage],
    output_dir: Path,
    config_path: str
) -> None:
    """Generate database index file from captured images."""
    output_dir = output_dir.resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    index = {
        'database': {
            'name': config['database']['name'],
            'version': config['database']['version'],
            'created': datetime.utcnow().isoformat(),
            'location': config['database']['location'],
            'center': config['database']['center'],
            'bounds': config['database']['bounds'],
            'image_count': len(captured_images)
        },
        'images': []
    }
    
    for img in captured_images:
        image_path = Path(img.filepath).resolve()
        relative_path = image_path.relative_to(output_dir).as_posix()
        index['images'].append({
            'id': img.id,
            'filepath': relative_path,
            'latitude': img.latitude,
            'longitude': img.longitude,
            'altitude': img.altitude,
            'heading': img.heading,
            'timestamp': img.timestamp,
            'width': img.width,
            'height': img.height
        })
    
    index_path = output_dir / 'database_index.yaml'
    with open(index_path, 'w') as f:
        yaml.dump(index, f, default_flow_style=False)
    
    print(f"Database index saved to: {index_path}")


def validate_coverage(
    captured_images: List[CapturedImage],
    config_path: str
) -> dict:
    """Validate that captured images provide adequate coverage."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    bounds = config['database']['bounds']
    
    # Calculate coverage statistics
    lats = [img.latitude for img in captured_images]
    lons = [img.longitude for img in captured_images]
    
    coverage = {
        'total_images': len(captured_images),
        'latitude_range': [min(lats), max(lats)],
        'longitude_range': [min(lons), max(lons)],
        'bounds_coverage': {
            'north': max(lats) >= bounds['max_latitude'] - 0.001,
            'south': min(lats) <= bounds['min_latitude'] + 0.001,
            'east': max(lons) >= bounds['max_longitude'] - 0.001,
            'west': min(lons) <= bounds['min_longitude'] + 0.001
        },
        'headings_covered': len(set(img.heading for img in captured_images))
    }
    
    # Check for gaps (simplified)
    coverage['adequate'] = (
        coverage['total_images'] >= 10 and
        all(coverage['bounds_coverage'].values())
    )
    
    return coverage


def main():
    parser = argparse.ArgumentParser(
        description='Capture reference images for VNS database'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../database/qau_reference_metadata.yaml',
        help='Path to reference metadata YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../database/images',
        help='Output directory for captured images'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic images (for testing without Gazebo)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Image width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Image height'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    config_path = (script_dir / args.config).resolve()
    output_dir = (script_dir / args.output).resolve()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading reference points from: {config_path}")
    reference_points = load_reference_points(str(config_path))
    print(f"Found {len(reference_points)} reference points")
    
    captured_images = []
    
    if args.synthetic:
        print("Generating synthetic reference images...")
        for i, point in enumerate(reference_points):
            print(f"  [{i+1}/{len(reference_points)}] {point.id}: {point.description}")
            image = create_synthetic_image(point, args.width, args.height)
            captured = save_image_with_metadata(image, point, output_dir)
            captured_images.append(captured)
    else:
        print("ROS2 camera capture not implemented in this version.")
        print("Use --synthetic flag to generate test images.")
        sys.exit(1)
    
    # Generate database index
    generate_database_index(captured_images, output_dir, str(config_path))
    
    # Validate coverage
    print("\nValidating coverage...")
    coverage = validate_coverage(captured_images, str(config_path))
    print(f"  Total images: {coverage['total_images']}")
    print(f"  Latitude range: {coverage['latitude_range']}")
    print(f"  Longitude range: {coverage['longitude_range']}")
    print(f"  Bounds coverage: {coverage['bounds_coverage']}")
    print(f"  Headings covered: {coverage['headings_covered']}")
    print(f"  Adequate coverage: {coverage['adequate']}")
    
    print(f"\nReference images saved to: {output_dir}")


if __name__ == '__main__':
    main()

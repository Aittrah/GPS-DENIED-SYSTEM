#!/usr/bin/env python3
"""
Build Reference Database for VNS Simulation

This script processes captured reference images and builds a feature database
for visual navigation. It extracts ORB features and creates a spatial index.

Usage:
    python build_reference_database.py --input ./images --output qau_campus.vnsdb

Requirements:
    - OpenCV with ORB feature detector
    - PyYAML for configuration
"""

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


@dataclass
class DatabaseEntry:
    """A single entry in the reference database."""
    id: str
    source_path: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    capture_time: str
    feature_count: int
    feature_algorithm: str
    keypoints: np.ndarray
    descriptors: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class GeoBounds:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within bounds."""
        return (self.min_lat <= lat <= self.max_lat and
                self.min_lon <= lon <= self.max_lon)
    
    def expand(self, lat: float, lon: float) -> None:
        """Expand bounds to include a point."""
        self.min_lat = min(self.min_lat, lat)
        self.max_lat = max(self.max_lat, lat)
        self.min_lon = min(self.min_lon, lon)
        self.max_lon = max(self.max_lon, lon)


@dataclass
class ReferenceDatabase:
    """Reference image database for visual navigation."""
    version: str
    name: str
    created: str
    algorithm: str
    bounds: GeoBounds
    entries: Dict[str, DatabaseEntry] = field(default_factory=dict)
    
    @property
    def entry_count(self) -> int:
        return len(self.entries)
    
    def add_entry(self, entry: DatabaseEntry) -> None:
        """Add an entry to the database."""
        self.entries[entry.id] = entry
        self.bounds.expand(entry.latitude, entry.longitude)
    
    def query_region(
        self,
        lat: float,
        lon: float,
        radius_deg: float
    ) -> List[DatabaseEntry]:
        """Find entries within radius of a point (simple implementation)."""
        results = []
        for entry in self.entries.values():
            dlat = entry.latitude - lat
            dlon = entry.longitude - lon
            dist = np.sqrt(dlat**2 + dlon**2)
            if dist <= radius_deg:
                results.append(entry)
        return results


def extract_orb_features(
    image_path: str,
    max_features: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ORB features from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if keypoints is None or len(keypoints) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 32)
    
    # Convert keypoints to numpy array
    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    if descriptors is None:
        descriptors = np.array([]).reshape(0, 32)
    
    return kp_array, descriptors


def load_image_index(index_path: str) -> List[dict]:
    """Load image index from YAML file."""
    with open(index_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('images', [])


def build_database(
    index_path: str,
    max_features: int = 500,
    algorithm: str = "ORB"
) -> ReferenceDatabase:
    """Build reference database from image index."""
    images = load_image_index(index_path)
    
    if not images:
        raise ValueError("No images found in index")
    
    # Initialize database with first image bounds
    first = images[0]
    db = ReferenceDatabase(
        version="1.0.0",
        name="QAU Campus Reference Database",
        created=datetime.utcnow().isoformat(),
        algorithm=algorithm,
        bounds=GeoBounds(
            min_lat=first['latitude'],
            max_lat=first['latitude'],
            min_lon=first['longitude'],
            max_lon=first['longitude']
        )
    )
    
    print(f"Processing {len(images)} images...")
    
    for i, img_data in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_data['id']}", end=" ")
        
        try:
            keypoints, descriptors = extract_orb_features(
                img_data['filepath'],
                max_features
            )
            
            entry = DatabaseEntry(
                id=img_data['id'],
                source_path=img_data['filepath'],
                latitude=img_data['latitude'],
                longitude=img_data['longitude'],
                altitude=img_data['altitude'],
                heading=img_data['heading'],
                capture_time=img_data.get('timestamp', ''),
                feature_count=len(keypoints),
                feature_algorithm=algorithm,
                keypoints=keypoints,
                descriptors=descriptors,
                metadata={
                    'width': img_data.get('width', 0),
                    'height': img_data.get('height', 0)
                }
            )
            
            db.add_entry(entry)
            print(f"- {entry.feature_count} features")
            
        except Exception as e:
            print(f"- ERROR: {e}")
    
    return db


def save_database(db: ReferenceDatabase, output_path: str) -> None:
    """Save database to binary file."""
    # Convert to serializable format
    data = {
        'version': db.version,
        'name': db.name,
        'created': db.created,
        'algorithm': db.algorithm,
        'bounds': {
            'min_lat': db.bounds.min_lat,
            'max_lat': db.bounds.max_lat,
            'min_lon': db.bounds.min_lon,
            'max_lon': db.bounds.max_lon
        },
        'entries': {}
    }
    
    for entry_id, entry in db.entries.items():
        data['entries'][entry_id] = {
            'id': entry.id,
            'source_path': entry.source_path,
            'latitude': entry.latitude,
            'longitude': entry.longitude,
            'altitude': entry.altitude,
            'heading': entry.heading,
            'capture_time': entry.capture_time,
            'feature_count': entry.feature_count,
            'feature_algorithm': entry.feature_algorithm,
            'keypoints': entry.keypoints,
            'descriptors': entry.descriptors,
            'metadata': entry.metadata
        }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Database saved to: {output_path}")


def print_database_stats(db: ReferenceDatabase) -> None:
    """Print database statistics."""
    total_features = sum(e.feature_count for e in db.entries.values())
    avg_features = total_features / db.entry_count if db.entry_count > 0 else 0
    
    print("\n=== Database Statistics ===")
    print(f"Name: {db.name}")
    print(f"Version: {db.version}")
    print(f"Algorithm: {db.algorithm}")
    print(f"Total entries: {db.entry_count}")
    print(f"Total features: {total_features}")
    print(f"Average features per image: {avg_features:.1f}")
    print(f"Geographic bounds:")
    print(f"  Latitude:  {db.bounds.min_lat:.6f} to {db.bounds.max_lat:.6f}")
    print(f"  Longitude: {db.bounds.min_lon:.6f} to {db.bounds.max_lon:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Build VNS reference database from captured images'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../database/images/database_index.yaml',
        help='Path to image index YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../database/qau_campus.vnsdb',
        help='Output database file path'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=500,
        help='Maximum features to extract per image'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ORB',
        choices=['ORB'],
        help='Feature extraction algorithm'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    index_path = script_dir / args.input
    output_path = script_dir / args.output
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Building database from: {index_path}")
    
    db = build_database(
        str(index_path),
        max_features=args.max_features,
        algorithm=args.algorithm
    )
    
    print_database_stats(db)
    
    save_database(db, str(output_path))
    
    print("\nDatabase build complete!")


if __name__ == '__main__':
    main()
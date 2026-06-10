import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    keypoints: np.ndarray  # Shape: (N, 2)
    descriptors: np.ndarray  # Shape: (N, 32)
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

class ReferenceDatabase:
    """Reference image database for visual navigation."""

    def __init__(
        self,
        name: str = "VNS Reference Database",
        version: str = "1.0.0",
        algorithm: str = "ORB",
        created: Optional[str] = None
    ) -> None:
        self.name = name
        self.version = version
        self.created = created or datetime.utcnow().isoformat()
        self.algorithm = algorithm
        self.bounds = GeoBounds(90.0, -90.0, 180.0, -180.0)
        self.entries: Dict[str, DatabaseEntry] = {}

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def add_entry(self, entry: DatabaseEntry) -> None:
        """Add an pre-extracted entry to the database."""
        self.entries[entry.id] = entry
        self.bounds.expand(entry.latitude, entry.longitude)

    def add_image(
        self,
        image_path: str,
        latitude: float = 0.0,
        longitude: float = 0.0,
        altitude: float = 0.0,
        heading: float = 0.0,
        image_id: Optional[str] = None
    ) -> None:
        """
        Extract ORB features and add image to the database.
        Matches the standard README usage: db.add_image("path.jpg")
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img_id = image_id or path.stem
        
        # Extract features
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        if keypoints is None or len(keypoints) == 0:
            kp_array = np.array([]).reshape(0, 2)
            descriptors = np.array([]).reshape(0, 32)
        else:
            kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            
        if descriptors is None:
            descriptors = np.array([]).reshape(0, 32)
            
        entry = DatabaseEntry(
            id=img_id,
            source_path=str(path),
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            heading=heading,
            capture_time=datetime.utcnow().isoformat(),
            feature_count=len(kp_array),
            feature_algorithm=self.algorithm,
            keypoints=kp_array,
            descriptors=descriptors,
            metadata={"width": image.shape[1], "height": image.shape[0]}
        )
        
        self.add_entry(entry)

    def query_region(self, lat: float, lon: float, radius_deg: float) -> List[DatabaseEntry]:
        """Find entries within radius of a point."""
        results = []
        for entry in self.entries.values():
            dlat = entry.latitude - lat
            dlon = entry.longitude - lon
            dist = np.sqrt(dlat**2 + dlon**2)
            if dist <= radius_deg:
                results.append(entry)
        return results

    def save(self, filepath: str) -> None:
        """Save database to binary file (pickle)."""
        data = {
            'version': self.version,
            'name': self.name,
            'created': self.created,
            'algorithm': self.algorithm,
            'bounds': {
                'min_lat': self.bounds.min_lat,
                'max_lat': self.bounds.max_lat,
                'min_lon': self.bounds.min_lon,
                'max_lon': self.bounds.max_lon
            },
            'entries': {}
        }
        
        for entry_id, entry in self.entries.items():
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
            
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "ReferenceDatabase":
        """Load database from a binary file (pickle)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found at: {filepath}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        db = cls(
            name=data.get('name', 'Reference Database'),
            version=data.get('version', '1.0.0'),
            algorithm=data.get('algorithm', 'ORB'),
            created=data.get('created')
        )
        
        bounds_data = data.get('bounds', {})
        db.bounds = GeoBounds(
            min_lat=bounds_data.get('min_lat', 90.0),
            max_lat=bounds_data.get('max_lat', -90.0),
            min_lon=bounds_data.get('min_lon', 180.0),
            max_lon=bounds_data.get('max_lon', -180.0)
        )
        
        for entry_id, e in data.get('entries', {}).items():
            entry = DatabaseEntry(
                id=e['id'],
                source_path=e['source_path'],
                latitude=e['latitude'],
                longitude=e['longitude'],
                altitude=e['altitude'],
                heading=e['heading'],
                capture_time=e.get('capture_time', ''),
                feature_count=e.get('feature_count', 0),
                feature_algorithm=e.get('feature_algorithm', 'ORB'),
                keypoints=e['keypoints'],
                descriptors=e['descriptors'],
                metadata=e.get('metadata', {})
            )
            db.entries[entry_id] = entry
            
        return db

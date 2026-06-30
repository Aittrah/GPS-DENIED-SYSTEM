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
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV not installed. Run: pip3 install opencv-python")
    sys.exit(1)

# Make the `vns` package importable whether or not it is pip-installed, so the
# offline builder and the online BoVWIndex share the SAME histogram routine.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:
    from sklearn.cluster import MiniBatchKMeans

    from vns.database.reference_db import (
        DatabaseEntry as SafeDatabaseEntry,
        GeoBounds as SafeGeoBounds,
        ReferenceDatabase as SafeReferenceDatabase,
    )
    from vns.utils.paths import relativize_to_base, resolve_from_base
    from vns.vision.bovw_retrieval import BoVWIndex, compute_bovw_histogram
except ImportError:
    print("Error: scikit-learn not installed. Run: pip3 install scikit-learn")
    sys.exit(1)

logger = logging.getLogger("vns.build_reference_database")


def resolve_cli_path(path_value: str, *, script_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (script_dir / candidate).resolve()


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
    # BoVW histogram for this image: (K,) float32, L2-normalized. None until a
    # vocabulary is built; stays None for legacy / --no-vocab databases.
    bovw_histogram: Optional[np.ndarray] = None


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
    # BoVW vocabulary (k-means cluster centers), (K, 32) float32. None = no index.
    vocabulary: Optional[np.ndarray] = None
    # Metric the histogram NN index was/should be built with ("cosine" or "l2").
    bovw_metric: str = "cosine"

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


def load_image_index(index_path: str) -> dict:
    """Load the full image index document from YAML."""
    with open(index_path, 'r') as f:
        return yaml.safe_load(f) or {}


def build_database(
    index_path: str,
    max_features: int = 500,
    algorithm: str = "ORB"
) -> ReferenceDatabase:
    """Build reference database from image index."""
    index_file = Path(index_path).resolve()
    index_document = load_image_index(str(index_file))
    images = index_document.get('images', [])
    database_meta = index_document.get('database', {})
    
    if not images:
        raise ValueError("No images found in index")
    
    # Initialize database with first image bounds
    first = images[0]
    db = ReferenceDatabase(
        version="2.0.0",  # 2.0.0 stores the safe archive format
        name=database_meta.get("name", "QAU Campus Reference Database"),
        created=datetime.now(timezone.utc).isoformat(),
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
            image_path = resolve_from_base(img_data['filepath'], index_file.parent)
            keypoints, descriptors = extract_orb_features(
                str(image_path),
                max_features
            )
            metadata = {
                'width': img_data.get('width', 0),
                'height': img_data.get('height', 0),
            }
            if database_meta.get("source_type") is not None:
                metadata.setdefault("source_type", database_meta.get("source_type"))
            for key, value in img_data.items():
                if key in {
                    'id',
                    'filepath',
                    'latitude',
                    'longitude',
                    'altitude',
                    'heading',
                    'timestamp',
                    'width',
                    'height',
                }:
                    continue
                metadata[key] = value

            entry = DatabaseEntry(
                id=img_data['id'],
                source_path=str(image_path),
                latitude=img_data['latitude'],
                longitude=img_data['longitude'],
                altitude=img_data['altitude'],
                heading=img_data['heading'],
                capture_time=img_data.get('timestamp', ''),
                feature_count=len(keypoints),
                feature_algorithm=algorithm,
                keypoints=keypoints,
                descriptors=descriptors,
                metadata=metadata,
            )
            
            db.add_entry(entry)
            print(f"- {entry.feature_count} features")
            
        except Exception as e:
            print(f"- ERROR: {e}")
    
    return db


def build_bovw_vocabulary(
    db: ReferenceDatabase,
    vocab_size: int = 1000,
    metric: str = "cosine",
    random_state: int = 42,
    batch_size: int = 1000,
) -> Optional[dict]:
    """Train a BoVW vocabulary and per-image histograms over the DB's descriptors.

    This is the offline scalability layer: it sits ON TOP of the ORB descriptors
    already computed for each entry and does not change feature extraction.

    ORB descriptors are BINARY (uint8, Hamming space); MiniBatchKMeans clusters
    them with Euclidean distance, so the vocabulary is a known/accepted
    approximation chosen for speed. The SAME Euclidean nearest-word assignment is
    reused at query time via vns.vision.bovw_retrieval.compute_bovw_histogram, so
    the offline and online histograms can never diverge.

    Returns a small stats dict, or None if there were no descriptors to cluster.
    """
    all_desc = [
        e.descriptors
        for e in db.entries.values()
        if e.descriptors is not None and len(e.descriptors) > 0
    ]
    if not all_desc:
        logger.warning("No descriptors available; skipping BoVW vocabulary build.")
        return None

    stacked = np.vstack(all_desc).astype(np.float32)
    total = stacked.shape[0]

    # Clamp K for small / test databases where total descriptors < requested K.
    k = int(vocab_size)
    if total < k:
        logger.warning(
            "Total descriptors (%d) < requested vocab_size (%d); clamping K to %d.",
            total, vocab_size, total,
        )
        k = total

    logger.info(
        "Training BoVW vocabulary: K=%d over %d descriptors (metric=%s, seed=%d).",
        k, total, metric, random_state,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,           # deterministic, reproducible rebuilds
        batch_size=min(batch_size, total),
    )
    kmeans.fit(stacked)
    vocabulary = kmeans.cluster_centers_.astype(np.float32)

    db.vocabulary = vocabulary
    db.bovw_metric = metric

    for entry in db.entries.values():
        hist = compute_bovw_histogram(entry.descriptors, vocabulary, k)
        entry.bovw_histogram = hist
        logger.debug(
            "  %s: histogram nnz=%d / %d", entry.id, int(np.count_nonzero(hist)), k
        )

    logger.info("BoVW vocabulary built: K=%d, %d reference histograms.", k, db.entry_count)
    return {"k": k, "total_descriptors": total, "metric": metric}


def save_database(db: ReferenceDatabase, output_path: str) -> None:
    """Save database to the canonical safe archive format."""
    output_file = Path(output_path).resolve()
    output_dir = output_file.parent
    safe_db = SafeReferenceDatabase(
        name=db.name,
        version=db.version,
        algorithm=db.algorithm,
        created=db.created,
    )
    safe_db.bounds = SafeGeoBounds(
        min_lat=db.bounds.min_lat,
        max_lat=db.bounds.max_lat,
        min_lon=db.bounds.min_lon,
        max_lon=db.bounds.max_lon,
    )
    safe_db.vocabulary = None if db.vocabulary is None else np.asarray(
        db.vocabulary, dtype=np.float32,
    )
    safe_db.bovw_metric = db.bovw_metric

    for entry_id, entry in db.entries.items():
        safe_db.entries[entry_id] = SafeDatabaseEntry(
            id=entry.id,
            source_path=relativize_to_base(entry.source_path, output_dir),
            latitude=entry.latitude,
            longitude=entry.longitude,
            altitude=entry.altitude,
            heading=entry.heading,
            capture_time=entry.capture_time,
            feature_count=entry.feature_count,
            feature_algorithm=entry.feature_algorithm,
            keypoints=np.asarray(entry.keypoints, dtype=np.float32),
            descriptors=np.asarray(entry.descriptors, dtype=np.uint8),
            metadata=dict(entry.metadata),
            bovw_histogram=(
                None
                if entry.bovw_histogram is None
                else np.asarray(entry.bovw_histogram, dtype=np.float32)
            ),
        )

    safe_db.save(str(output_file))
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


def print_bovw_stats(db: ReferenceDatabase) -> None:
    """Print BoVW index statistics, including query (histogram + NN) timing."""
    print("\n=== BoVW Index Statistics ===")
    if db.vocabulary is None:
        print("Not built (run with --build-vocab to enable coarse retrieval).")
        return

    k = int(db.vocabulary.shape[0])
    hists = [e.bovw_histogram for e in db.entries.values() if e.bovw_histogram is not None]
    nnz = [int(np.count_nonzero(h)) for h in hists]

    print(f"Vocabulary size (K): {k}")
    print(f"Reference histograms: {len(hists)}")
    print(f"NN metric: {db.bovw_metric}")
    if nnz:
        print(
            f"Histogram non-zeros: min={min(nnz)} "
            f"mean={sum(nnz) / len(nnz):.1f} max={max(nnz)}"
        )

    # Time the real query path (histogram build + vectorized top-k lookup) by
    # exercising the same BoVWIndex used online. Target: sub-millisecond.
    ids = list(db.entries.keys())
    matrix = np.vstack([db.entries[i].bovw_histogram for i in ids]).astype(np.float32)
    index = BoVWIndex(db.vocabulary, ids, matrix, metric=db.bovw_metric)
    sample = next(iter(db.entries.values()))
    top_k = min(5, len(ids))
    index.query(sample.descriptors, k=top_k)  # warm up
    runs = 200
    start = time.perf_counter()
    for _ in range(runs):
        index.query(sample.descriptors, k=top_k)
    per_query_ms = (time.perf_counter() - start) / runs * 1000.0
    print(
        f"Mean query time (histogram + top-{top_k} lookup): "
        f"{per_query_ms:.3f} ms over {runs} runs"
    )


def load_bovw_defaults(config_path: Path) -> dict:
    """Read build-time BoVW defaults from simulation.yaml's retrieval.bovw section."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    return ((cfg.get('retrieval') or {}).get('bovw') or {})


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
    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument(
        '--build-vocab',
        dest='build_vocab',
        action='store_true',
        default=True,
        help='Build the BoVW index on top of the ORB descriptors (default)'
    )
    vocab_group.add_argument(
        '--no-vocab',
        dest='build_vocab',
        action='store_false',
        help='Skip the BoVW index (legacy ORB-only database)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=None,
        help='BoVW vocabulary size K (default: simulation.yaml retrieval.bovw.vocab_size, else 1000)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=None,
        choices=['cosine', 'l2'],
        help='Histogram NN metric (default: simulation.yaml retrieval.bovw.metric, else cosine)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help='MiniBatchKMeans seed for reproducible vocabularies (default from config, else 42)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print detailed BoVW stats including query timing'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.stats else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    index_path = resolve_cli_path(args.input, script_dir=script_dir)
    output_path = resolve_cli_path(args.output, script_dir=script_dir)

    # Build-time BoVW defaults from simulation.yaml (CLI flags override these).
    bovw_cfg = load_bovw_defaults(
        resolve_cli_path("../config/simulation.yaml", script_dir=script_dir)
    )
    vocab_size = args.vocab_size if args.vocab_size is not None else int(bovw_cfg.get('vocab_size', 1000))
    metric = args.metric if args.metric is not None else str(bovw_cfg.get('metric', 'cosine'))
    random_state = args.random_state if args.random_state is not None else int(bovw_cfg.get('random_state', 42))
    batch_size = int(bovw_cfg.get('batch_size', 1000))

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building database from: {index_path}")

    db = build_database(
        str(index_path),
        max_features=args.max_features,
        algorithm=args.algorithm
    )

    if args.build_vocab:
        build_bovw_vocabulary(
            db,
            vocab_size=vocab_size,
            metric=metric,
            random_state=random_state,
            batch_size=batch_size,
        )

    print_database_stats(db)

    if args.stats:
        print_bovw_stats(db)

    save_database(db, str(output_path))

    print("\nDatabase build complete!")


if __name__ == '__main__':
    main()

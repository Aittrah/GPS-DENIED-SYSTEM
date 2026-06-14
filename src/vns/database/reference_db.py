import json
import logging
import pickle
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("vns.database")

SAFE_ARCHIVE_FORMAT = "vns.safe_database"
SAFE_SCHEMA_VERSION = 1
SAFE_FORMAT = "safe_archive"
LEGACY_FORMAT = "legacy_pickle"
UNKNOWN_FORMAT = "unknown"

MAX_DATABASE_BYTES = 1024 * 1024 * 1024
MAX_ENTRY_COUNT = 100_000
MAX_ARRAY_ROWS = 1_000_000


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
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )

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
        created: Optional[str] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.created = created or datetime.utcnow().isoformat()
        self.algorithm = algorithm
        self.bounds = GeoBounds(90.0, -90.0, 180.0, -180.0)
        self.entries: Dict[str, DatabaseEntry] = {}
        self.vocabulary: Optional[np.ndarray] = None
        self.bovw_metric: str = "cosine"

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @staticmethod
    def _compute_bounds(entries: List[DatabaseEntry]) -> GeoBounds:
        if not entries:
            return GeoBounds(90.0, -90.0, 180.0, -180.0)
        min_lat = min(entry.latitude for entry in entries)
        max_lat = max(entry.latitude for entry in entries)
        min_lon = min(entry.longitude for entry in entries)
        max_lon = max(entry.longitude for entry in entries)
        return GeoBounds(min_lat, max_lat, min_lon, max_lon)

    @staticmethod
    def _legacy_block_error(filepath: str) -> ValueError:
        return ValueError(
            f"Database '{filepath}' uses the insecure legacy pickle format and "
            "is blocked by default. Use `vns database migrate --input ... "
            "--output ...` or ReferenceDatabase.load_legacy_trusted(...) for "
            "an explicit trusted migration step."
        )

    @staticmethod
    def _unsupported_format_error(filepath: str) -> ValueError:
        return ValueError(
            f"Database '{filepath}' is not a supported VNS database archive."
        )

    @staticmethod
    def _ensure_size_limit(path: Path) -> None:
        size = path.stat().st_size
        if size > MAX_DATABASE_BYTES:
            raise ValueError(
                f"Database '{path}' exceeds the maximum supported size of "
                f"{MAX_DATABASE_BYTES} bytes."
            )

    @staticmethod
    def _empty_keypoints() -> np.ndarray:
        return np.empty((0, 2), dtype=np.float32)

    @staticmethod
    def _empty_descriptors() -> np.ndarray:
        return np.empty((0, 32), dtype=np.uint8)

    @staticmethod
    def _coerce_keypoints(
        value: Optional[np.ndarray], entry_id: str, *, context: str
    ) -> np.ndarray:
        if value is None:
            return ReferenceDatabase._empty_keypoints()
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{context}: entry '{entry_id}' keypoints must have shape (N, 2)."
            )
        if arr.shape[0] > MAX_ARRAY_ROWS:
            raise ValueError(
                f"{context}: entry '{entry_id}' keypoints exceed the row limit."
            )
        return arr

    @staticmethod
    def _coerce_descriptors(
        value: Optional[np.ndarray], entry_id: str, *, context: str
    ) -> np.ndarray:
        if value is None:
            return ReferenceDatabase._empty_descriptors()
        arr = np.asarray(value, dtype=np.uint8)
        if arr.ndim != 2 or arr.shape[1] != 32:
            raise ValueError(
                f"{context}: entry '{entry_id}' descriptors must have shape (N, 32)."
            )
        if arr.shape[0] > MAX_ARRAY_ROWS:
            raise ValueError(
                f"{context}: entry '{entry_id}' descriptors exceed the row limit."
            )
        return arr

    @staticmethod
    def _coerce_histogram(
        value: Optional[np.ndarray],
        entry_id: str,
        expected_width: int,
        *,
        context: str,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != expected_width:
            raise ValueError(
                f"{context}: entry '{entry_id}' BoVW histogram must have shape "
                f"({expected_width},)."
            )
        return arr

    @staticmethod
    def _coerce_vocabulary(
        value: Optional[np.ndarray], *, context: str
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 32:
            raise ValueError(
                f"{context}: BoVW vocabulary must have shape (K, 32)."
            )
        if arr.shape[0] == 0 or arr.shape[0] > MAX_ARRAY_ROWS:
            raise ValueError(
                f"{context}: BoVW vocabulary size is out of bounds."
            )
        return arr

    @staticmethod
    def _require_metadata_dict(metadata: Any, *, context: str) -> dict:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{context}: entry metadata must be a JSON object.")
        return metadata

    @classmethod
    def detect_format(cls, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found at: {filepath}")

        cls._ensure_size_limit(path)

        if zipfile.is_zipfile(path):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    raw_metadata = archive["__metadata__"].item()
                    metadata = json.loads(str(raw_metadata))
                if metadata.get("format") == SAFE_ARCHIVE_FORMAT:
                    return SAFE_FORMAT
            except Exception:
                return UNKNOWN_FORMAT
            return UNKNOWN_FORMAT

        try:
            with open(path, "rb") as handle:
                prefix = handle.read(2)
        except OSError:
            return UNKNOWN_FORMAT

        if prefix[:1] == b"\x80":
            return LEGACY_FORMAT
        return UNKNOWN_FORMAT

    def add_entry(self, entry: DatabaseEntry) -> None:
        """Add a pre-extracted entry to the database."""
        self.entries[entry.id] = entry
        self.bounds.expand(entry.latitude, entry.longitude)

    def add_image(
        self,
        image_path: str,
        latitude: float = 0.0,
        longitude: float = 0.0,
        altitude: float = 0.0,
        heading: float = 0.0,
        image_id: Optional[str] = None,
    ) -> None:
        """
        Extract ORB features and add image to the database.
        Matches the standard README usage: db.add_image("path.jpg")
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img_id = image_id or path.stem

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(image, None)

        if keypoints is None or len(keypoints) == 0:
            kp_array = self._empty_keypoints()
            descriptors_array = self._empty_descriptors()
        else:
            kp_array = np.array(
                [[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32
            )
            descriptors_array = (
                self._empty_descriptors()
                if descriptors is None
                else np.asarray(descriptors, dtype=np.uint8)
            )

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
            descriptors=descriptors_array,
            metadata={"width": image.shape[1], "height": image.shape[0]},
        )

        self.add_entry(entry)

    def query_region(
        self, lat: float, lon: float, radius_deg: float
    ) -> List[DatabaseEntry]:
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
        """Save database to a safe archive format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry_documents = []
        archive_arrays: Dict[str, np.ndarray] = {}

        vocabulary = self._coerce_vocabulary(
            self.vocabulary, context="Saving database"
        )
        if vocabulary is not None:
            archive_arrays["__bovw_vocabulary__"] = vocabulary

        for index, entry in enumerate(self.entries.values()):
            context = "Saving database"
            keypoints = self._coerce_keypoints(
                entry.keypoints, entry.id, context=context
            )
            descriptors = self._coerce_descriptors(
                entry.descriptors, entry.id, context=context
            )
            if len(keypoints) != len(descriptors):
                raise ValueError(
                    f"{context}: entry '{entry.id}' keypoint/descriptor count mismatch."
                )
            if entry.feature_count != len(keypoints):
                raise ValueError(
                    f"{context}: entry '{entry.id}' feature_count does not match "
                    "the stored feature arrays."
                )

            keypoints_key = f"entry_{index:06d}_keypoints"
            descriptors_key = f"entry_{index:06d}_descriptors"
            archive_arrays[keypoints_key] = keypoints
            archive_arrays[descriptors_key] = descriptors

            bovw_key = None
            if vocabulary is not None:
                histogram = self._coerce_histogram(
                    entry.bovw_histogram,
                    entry.id,
                    vocabulary.shape[0],
                    context=context,
                )
                if histogram is None:
                    raise ValueError(
                        f"{context}: entry '{entry.id}' is missing its BoVW histogram."
                    )
                bovw_key = f"entry_{index:06d}_bovw_histogram"
                archive_arrays[bovw_key] = histogram
            elif entry.bovw_histogram is not None:
                raise ValueError(
                    f"{context}: entry '{entry.id}' has a BoVW histogram but no "
                    "database vocabulary."
                )

            entry_documents.append(
                {
                    "id": entry.id,
                    "source_path": entry.source_path,
                    "latitude": float(entry.latitude),
                    "longitude": float(entry.longitude),
                    "altitude": float(entry.altitude),
                    "heading": float(entry.heading),
                    "capture_time": entry.capture_time,
                    "feature_count": int(entry.feature_count),
                    "feature_algorithm": entry.feature_algorithm,
                    "metadata": self._require_metadata_dict(
                        entry.metadata, context=context
                    ),
                    "keypoints_key": keypoints_key,
                    "descriptors_key": descriptors_key,
                    "bovw_histogram_key": bovw_key,
                }
            )

        stored_bounds = self._compute_bounds(list(self.entries.values()))
        metadata: Dict[str, Any] = {
            "format": SAFE_ARCHIVE_FORMAT,
            "schema_version": SAFE_SCHEMA_VERSION,
            "version": self.version,
            "name": self.name,
            "created": self.created,
            "algorithm": self.algorithm,
            "bounds": {
                "min_lat": float(stored_bounds.min_lat),
                "max_lat": float(stored_bounds.max_lat),
                "min_lon": float(stored_bounds.min_lon),
                "max_lon": float(stored_bounds.max_lon),
            },
            "entries": entry_documents,
        }
        if vocabulary is not None:
            metadata["bovw"] = {
                "metric": self.bovw_metric,
                "vocabulary_key": "__bovw_vocabulary__",
            }

        archive_arrays["__metadata__"] = np.array(
            json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        )

        with open(path, "wb") as handle:
            np.savez_compressed(handle, **archive_arrays)

    @classmethod
    def load(cls, filepath: str) -> "ReferenceDatabase":
        """Load a database from the safe archive format."""
        file_format = cls.detect_format(filepath)
        if file_format == SAFE_FORMAT:
            return cls._load_safe(filepath)
        if file_format == LEGACY_FORMAT:
            raise cls._legacy_block_error(filepath)
        raise cls._unsupported_format_error(filepath)

    @classmethod
    def load_legacy_trusted(cls, filepath: str) -> "ReferenceDatabase":
        """Load a legacy pickle-backed database from a trusted source only."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found at: {filepath}")

        cls._ensure_size_limit(path)
        try:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
        except Exception as exc:
            raise ValueError(
                f"Failed to load trusted legacy database '{filepath}': {exc}"
            ) from exc

        logger.warning(
            "Loading trusted legacy pickle database: %s. Migrate this file to "
            "the safe archive format as soon as possible.",
            filepath,
        )
        return cls._from_legacy_data(data, filepath=filepath)

    @classmethod
    def migrate_legacy_file(cls, input_path: str, output_path: str) -> "ReferenceDatabase":
        """Convert a trusted legacy pickle database to the safe archive format."""
        db = cls.load_legacy_trusted(input_path)
        db.save(output_path)
        return db

    @classmethod
    def _load_safe(cls, filepath: str) -> "ReferenceDatabase":
        path = Path(filepath)
        cls._ensure_size_limit(path)

        try:
            with np.load(path, allow_pickle=False) as archive:
                raw_metadata = archive["__metadata__"].item()
                metadata = json.loads(str(raw_metadata))
                return cls._from_safe_archive(archive, metadata, filepath=filepath)
        except KeyError as exc:
            raise ValueError(
                f"Database '{filepath}' is missing required archive data: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Database '{filepath}' has invalid metadata JSON: {exc}"
            ) from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to load database '{filepath}': {exc}"
            ) from exc

    @classmethod
    def _from_safe_archive(
        cls, archive: Any, metadata: Dict[str, Any], *, filepath: str
    ) -> "ReferenceDatabase":
        if metadata.get("format") != SAFE_ARCHIVE_FORMAT:
            raise cls._unsupported_format_error(filepath)
        if metadata.get("schema_version") != SAFE_SCHEMA_VERSION:
            raise ValueError(
                f"Database '{filepath}' uses unsupported schema version "
                f"{metadata.get('schema_version')!r}."
            )

        entries_metadata = metadata.get("entries")
        if not isinstance(entries_metadata, list):
            raise ValueError(
                f"Database '{filepath}' metadata is missing the entries list."
            )
        if len(entries_metadata) > MAX_ENTRY_COUNT:
            raise ValueError(
                f"Database '{filepath}' exceeds the maximum supported entry count."
            )

        bounds_data = metadata.get("bounds", {})
        expected_bounds = GeoBounds(
            min_lat=float(bounds_data.get("min_lat", 90.0)),
            max_lat=float(bounds_data.get("max_lat", -90.0)),
            min_lon=float(bounds_data.get("min_lon", 180.0)),
            max_lon=float(bounds_data.get("max_lon", -180.0)),
        )
        db = cls(
            name=metadata.get("name", "Reference Database"),
            version=metadata.get("version", "1.0.0"),
            algorithm=metadata.get("algorithm", "ORB"),
            created=metadata.get("created"),
        )

        bovw_meta = metadata.get("bovw")
        vocabulary = None
        if bovw_meta is not None:
            if not isinstance(bovw_meta, dict):
                raise ValueError(
                    f"Database '{filepath}' contains invalid BoVW metadata."
                )
            vocabulary_key = bovw_meta.get("vocabulary_key")
            if not isinstance(vocabulary_key, str):
                raise ValueError(
                    f"Database '{filepath}' BoVW metadata is missing vocabulary_key."
                )
            vocabulary = cls._coerce_vocabulary(
                archive[vocabulary_key], context=f"Loading database '{filepath}'"
            )
            db.vocabulary = vocabulary
            db.bovw_metric = str(bovw_meta.get("metric", "cosine"))
            if db.bovw_metric not in {"cosine", "l2"}:
                raise ValueError(
                    f"Database '{filepath}' uses unsupported BoVW metric "
                    f"{db.bovw_metric!r}."
                )

        for entry_doc in entries_metadata:
            if not isinstance(entry_doc, dict):
                raise ValueError(
                    f"Database '{filepath}' contains malformed entry metadata."
                )
            entry_id = str(entry_doc["id"])
            keypoints = cls._coerce_keypoints(
                archive[entry_doc["keypoints_key"]],
                entry_id,
                context=f"Loading database '{filepath}'",
            )
            descriptors = cls._coerce_descriptors(
                archive[entry_doc["descriptors_key"]],
                entry_id,
                context=f"Loading database '{filepath}'",
            )
            if len(keypoints) != len(descriptors):
                raise ValueError(
                    f"Loading database '{filepath}': entry '{entry_id}' has a "
                    "keypoint/descriptor count mismatch."
                )

            feature_count = int(entry_doc.get("feature_count", len(keypoints)))
            if feature_count != len(keypoints):
                raise ValueError(
                    f"Loading database '{filepath}': entry '{entry_id}' has an "
                    "invalid feature_count."
                )

            bovw_histogram = None
            bovw_key = entry_doc.get("bovw_histogram_key")
            if vocabulary is not None:
                if not isinstance(bovw_key, str):
                    raise ValueError(
                        f"Loading database '{filepath}': entry '{entry_id}' is "
                        "missing its BoVW histogram."
                    )
                bovw_histogram = cls._coerce_histogram(
                    archive[bovw_key],
                    entry_id,
                    vocabulary.shape[0],
                    context=f"Loading database '{filepath}'",
                )
            elif bovw_key is not None:
                raise ValueError(
                    f"Loading database '{filepath}': entry '{entry_id}' has a "
                    "BoVW histogram without a vocabulary."
                )

            entry = DatabaseEntry(
                id=entry_id,
                source_path=str(entry_doc["source_path"]),
                latitude=float(entry_doc["latitude"]),
                longitude=float(entry_doc["longitude"]),
                altitude=float(entry_doc["altitude"]),
                heading=float(entry_doc["heading"]),
                capture_time=str(entry_doc.get("capture_time", "")),
                feature_count=feature_count,
                feature_algorithm=str(entry_doc.get("feature_algorithm", "ORB")),
                keypoints=keypoints,
                descriptors=descriptors,
                metadata=cls._require_metadata_dict(
                    entry_doc.get("metadata"),
                    context=f"Loading database '{filepath}'",
                ),
                bovw_histogram=bovw_histogram,
            )
            db.add_entry(entry)

        computed_bounds = cls._compute_bounds(list(db.entries.values()))
        if db.entries and computed_bounds != expected_bounds:
            raise ValueError(
                f"Database '{filepath}' bounds do not match the stored entries."
            )
        db.bounds = expected_bounds

        return db

    @classmethod
    def _from_legacy_data(
        cls, data: Dict[str, Any], *, filepath: str
    ) -> "ReferenceDatabase":
        if not isinstance(data, dict):
            raise ValueError(
                f"Legacy database '{filepath}' does not contain a valid payload."
            )

        bounds_data = data.get("bounds", {})
        expected_bounds = GeoBounds(
            min_lat=float(bounds_data.get("min_lat", 90.0)),
            max_lat=float(bounds_data.get("max_lat", -90.0)),
            min_lon=float(bounds_data.get("min_lon", 180.0)),
            max_lon=float(bounds_data.get("max_lon", -180.0)),
        )
        db = cls(
            name=data.get("name", "Reference Database"),
            version=data.get("version", "1.0.0"),
            algorithm=data.get("algorithm", "ORB"),
            created=data.get("created"),
        )

        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError(
                f"Legacy database '{filepath}' contains malformed entries."
            )
        if len(entries) > MAX_ENTRY_COUNT:
            raise ValueError(
                f"Legacy database '{filepath}' exceeds the supported entry count."
            )

        for entry_id, entry_data in entries.items():
            if not isinstance(entry_data, dict):
                raise ValueError(
                    f"Legacy database '{filepath}' contains malformed entry data."
                )
            keypoints = cls._coerce_keypoints(
                entry_data.get("keypoints"),
                str(entry_id),
                context=f"Loading legacy database '{filepath}'",
            )
            descriptors = cls._coerce_descriptors(
                entry_data.get("descriptors"),
                str(entry_id),
                context=f"Loading legacy database '{filepath}'",
            )
            if len(keypoints) != len(descriptors):
                raise ValueError(
                    f"Loading legacy database '{filepath}': entry '{entry_id}' has "
                    "a keypoint/descriptor count mismatch."
                )
            feature_count = int(entry_data.get("feature_count", len(keypoints)))
            if feature_count != len(keypoints):
                raise ValueError(
                    f"Loading legacy database '{filepath}': entry '{entry_id}' has "
                    "an invalid feature_count."
                )

            db.add_entry(DatabaseEntry(
                id=str(entry_data["id"]),
                source_path=str(entry_data["source_path"]),
                latitude=float(entry_data["latitude"]),
                longitude=float(entry_data["longitude"]),
                altitude=float(entry_data["altitude"]),
                heading=float(entry_data["heading"]),
                capture_time=str(entry_data.get("capture_time", "")),
                feature_count=feature_count,
                feature_algorithm=str(entry_data.get("feature_algorithm", "ORB")),
                keypoints=keypoints,
                descriptors=descriptors,
                metadata=cls._require_metadata_dict(
                    entry_data.get("metadata"),
                    context=f"Loading legacy database '{filepath}'",
                ),
            ))

        legacy_bovw = data.get("bovw")
        if legacy_bovw is not None:
            if not isinstance(legacy_bovw, dict):
                raise ValueError(
                    f"Legacy database '{filepath}' contains malformed BoVW data."
                )
            vocabulary = cls._coerce_vocabulary(
                legacy_bovw.get("vocabulary"),
                context=f"Loading legacy database '{filepath}'",
            )
            ids = legacy_bovw.get("ids")
            histograms = legacy_bovw.get("histograms")
            if vocabulary is None or not isinstance(ids, list):
                raise ValueError(
                    f"Legacy database '{filepath}' contains incomplete BoVW data."
                )
            hist_matrix = np.asarray(histograms, dtype=np.float32)
            if hist_matrix.ndim != 2 or hist_matrix.shape != (
                len(ids),
                vocabulary.shape[0],
            ):
                raise ValueError(
                    f"Legacy database '{filepath}' contains invalid BoVW histograms."
                )
            db.vocabulary = vocabulary
            db.bovw_metric = str(legacy_bovw.get("metric", "cosine"))
            if db.bovw_metric not in {"cosine", "l2"}:
                raise ValueError(
                    f"Legacy database '{filepath}' uses unsupported BoVW metric "
                    f"{db.bovw_metric!r}."
                )
            for index, entry_id in enumerate(ids):
                key = str(entry_id)
                if key not in db.entries:
                    raise ValueError(
                        f"Legacy database '{filepath}' BoVW references unknown "
                        f"entry '{key}'."
                    )
                db.entries[key].bovw_histogram = hist_matrix[index]

        computed_bounds = cls._compute_bounds(list(db.entries.values()))
        if bounds_data and db.entries and computed_bounds != expected_bounds:
            raise ValueError(
                f"Legacy database '{filepath}' bounds do not match the stored entries."
            )
        db.bounds = computed_bounds if db.entries else expected_bounds

        return db

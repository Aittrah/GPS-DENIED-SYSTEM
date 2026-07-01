from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from ..perception.feature_extractor import DINOv2Extractor, ORBVerifier
from ..preprocessing.calibration import CameraCalibration, load_camera_calibration
from ..preprocessing.satellite_preprocessor import SatellitePreprocessor
from ..preprocessing.uav_preprocessor import UAVPreprocessor
from ..preprocessing.patch_generator import PatchGenerator
from ..core.geo_utils import GeoPoint, geo_to_ned, ned_to_geo

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    # Fail soft, but keep import-time errors from crashing the whole app/tests.
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore
    print("[FAISSDatabase] faiss is not installed (pip install faiss-cpu). Build/query disabled.")

logger = logging.getLogger("vns.database.faiss")


class FAISSDatabase:
    """
    Two-stage satellite patch retrieval:
      Stage 1 — DINOv2 (384-dim) + FAISS IndexFlatIP for semantic retrieval.
      Stage 2 — ORBVerifier for geometric confirmation.
    """

    DESCRIPTOR_DIM = 384  # DINOv2 ViT-S/14 output dimension

    def __init__(
        self,
        index_path: str = "data/database/faiss.index",
        meta_path: str = "data/database/metadata.pkl",
        *,
        query_calibration_path: str | None = None,
        undistort_queries: bool = False,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.index: Optional[object] = None
        self.metadata: list[dict] = []
        self._undistort_queries = bool(undistort_queries)
        self._query_calibration: CameraCalibration | None = None

        self.extractor = DINOv2Extractor()
        self.verifier = ORBVerifier()

        if query_calibration_path is not None:
            self._query_calibration = load_camera_calibration(query_calibration_path)
            logger.info(
                "Loaded optional UAV query calibration from %s",
                query_calibration_path,
            )

        self._init_index()

    # ── private ──────────────────────────────────────────────────────────────

    def _init_index(self) -> None:
        if not _FAISS_AVAILABLE:
            return
        self.index = faiss.IndexFlatIP(self.DESCRIPTOR_DIM)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows in-place and return the array (float32)."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    # ── public — build ────────────────────────────────────────────────────────

    def build(self,
              satellite_image_path: str,
              lat: float,
              lon: float,
              alt: float = 550.0,
              progress_callback=None) -> int:
        """
        Full pipeline:
          1. SatellitePreprocessor → 512×512
          2. PatchGenerator → 256×256 patches (stride 128)
          3. DINOv2Extractor.extract_batch → (N, 384)
          4. L2-normalise → faiss.index.add()
          5. Persist metadata (patch geo + array for ORB verify)
          6. Save to disk
        Returns number of descriptors added.
        """
        if not _FAISS_AVAILABLE:
            print("[FAISSDatabase] faiss not available — build skipped")
            return 0
        if not self.extractor.available:
            print("[FAISSDatabase] DINOv2 not available — build skipped")
            return 0

        preprocessor = SatellitePreprocessor()
        pg = PatchGenerator()
        source_name = Path(satellite_image_path).name

        processed = preprocessor.load_and_preprocess(satellite_image_path)
        if processed is None:
            return 0

        origin = GeoPoint(lat, lon, alt)
        center_ned = geo_to_ned(origin, origin)
        meters_per_pixel = max(0.1, alt * 0.001)

        patches = pg.generate_patches(
            image=processed,
            image_center=center_ned,
            meters_per_pixel=meters_per_pixel,
            source="satellite",
            base_id=Path(satellite_image_path).stem,
        )

        total = len(patches)
        patch_arrays = [p.data for p in patches]

        if progress_callback:
            progress_callback(0, total)

        descriptors = self.extractor.extract_batch(patch_arrays, batch_size=4)
        if descriptors is None or len(descriptors) == 0:
            return 0

        descriptors = self._normalize(descriptors)
        self.index.add(descriptors)

        n_cols = max(1, (processed.shape[1] - 256) // 128 + 1)
        for i, patch in enumerate(patches):
            geo = ned_to_geo(patch.center_position, origin)
            self.metadata.append({
                "patch_id":     patch.patch_id,
                "lat":          geo.latitude,
                "lon":          geo.longitude,
                "alt":          geo.altitude,
                "row":          i // n_cols,
                "col":          i % n_cols,
                "source_image": source_name,
                "patch_array":  patch.data,
            })
            if progress_callback:
                progress_callback(i + 1, total)

        self.save()
        return len(patches)

    # ── public — query ────────────────────────────────────────────────────────

    def query(self,
              uav_image: np.ndarray,
              top_k: int = 5,
              min_confidence: float = 0.3) -> list[dict]:
        """
        Stage 1 — DINOv2 retrieval via FAISS.
        Stage 2 — ORB geometric verification.
        Returns list[dict] sorted by confidence descending.
        """
        if not _FAISS_AVAILABLE or self.index is None:
            return []
        if not self.extractor.available:
            return []
        if self.index.ntotal == 0:
            return []

        preprocessor = UAVPreprocessor(
            calibration=self._query_calibration,
            undistort=self._undistort_queries,
        )
        pg = PatchGenerator()

        processed = preprocessor.preprocess_frame(uav_image)
        if processed is None:
            return []

        from ..core.geo_utils import NEDPoint
        center_ned = NEDPoint(0.0, 0.0, 0.0)
        uav_patches = pg.generate_patches(
            image=processed,
            image_center=center_ned,
            meters_per_pixel=0.5,
            source="uav",
            base_id="uav_query",
        )
        if not uav_patches:
            return []

        n_uav = len(uav_patches)
        uav_arrays = [p.data for p in uav_patches]
        uav_descs = self.extractor.extract_batch(uav_arrays, batch_size=4)
        if uav_descs is None:
            return []

        uav_descs = self._normalize(uav_descs)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(uav_descs, k)

        # Vote: count how many UAV patches retrieved each metadata entry
        vote_map: dict[int, float] = {}
        for row_d, row_i in zip(distances, indices):
            for dist, idx in zip(row_d, row_i):
                if idx < 0:
                    continue
                vote_map[idx] = vote_map.get(idx, 0.0) + float(dist)

        # Sort by accumulated inner-product score (higher = better)
        ranked = sorted(vote_map.items(), key=lambda x: x[1], reverse=True)
        top_candidates = ranked[:top_k]

        results = []
        for meta_idx, score in top_candidates:
            meta = self.metadata[meta_idx]
            vote_count = sum(
                1 for row_i in indices
                if meta_idx in row_i
            )
            # Stage 2 — ORB verification (use first uav patch as query)
            orb_result = self.verifier.verify(
                uav_arrays[0], meta["patch_array"]
            )
            raw_conf = (vote_count / n_uav) * orb_result["confidence"]
            confidence = max(0.0, min(1.0, raw_conf))

            if confidence < min_confidence and not orb_result["verified"]:
                continue

            results.append({
                "patch_id":     meta["patch_id"],
                "lat":          meta["lat"],
                "lon":          meta["lon"],
                "alt":          meta["alt"],
                "confidence":   round(confidence, 4),
                "vote_count":   vote_count,
                "orb_verified": orb_result["verified"],
                "source_image": meta["source_image"],
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if not _FAISS_AVAILABLE or self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> bool:
        if not _FAISS_AVAILABLE:
            return False
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception as exc:
            print(f"[FAISSDatabase] load failed: {exc}")
            return False

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        index_mb = (
            self.index_path.stat().st_size / (1024 * 1024)
            if self.index_path.exists() else 0.0
        )
        unique_images = len({m["source_image"] for m in self.metadata})
        return {
            "total_descriptors": self.index.ntotal if self.index else 0,
            "unique_patches":    len(self.metadata),
            "unique_images":     unique_images,
            "index_size_mb":     round(index_mb, 3),
            "descriptor_dim":    self.DESCRIPTOR_DIM,
            "is_loaded":         self.index is not None and self.index.ntotal > 0,
            "model_name":        self.extractor.model_name,
        }

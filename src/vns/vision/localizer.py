"""VisualLocalizer: orchestrates the six-stage localization pipeline."""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from vns.database.reference_db import ReferenceDatabase
from vns.vision.extractor import FeatureExtractor
from vns.vision.pose_recovery import PoseRecovery
from vns.vision.preprocessor import Preprocessor
from vns.vision.retrieval import RetrievalIndex
from vns.vision.types import LocalizationResult, VerificationResult
from vns.vision.verification import GeometricVerifier

logger = logging.getLogger("vns.vision.localizer")


class VisualLocalizer:
    """End-to-end visual localization against a reference database.

    Pipeline stages:
        1. Preprocess (resize / grayscale / CLAHE)
        2. ORB feature extraction
        3. Coarse FLANN/LSH retrieval
        4. Geometric verification (ratio test + RANSAC homography)
        5. Best-match selection
        6. Pose recovery
    """

    def __init__(self, config: dict, database: ReferenceDatabase) -> None:
        preproc_cfg = config.get("preprocessing", {})
        feat_cfg = config.get("feature_extraction", {})
        match_cfg = config.get("matching", {})
        retrieval_cfg = config.get("retrieval", {})

        self._database = database
        self._camera_config: Dict = config.get("camera", {})
        self._geo_origin: Dict[str, float] = config.get("geo_reference", {})
        self._confidence_threshold: float = match_cfg.get(
            "confidence_threshold", 0.6,
        )
        self._top_k: int = retrieval_cfg.get("top_k", 5)
        camera_matrix = np.array(
            [
                [self._camera_config.get("fx", 554.25), 0.0, self._camera_config.get("cx", 320.0)],
                [0.0, self._camera_config.get("fy", 554.25), self._camera_config.get("cy", 240.0)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        distortion = np.asarray(
            self._camera_config.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )

        self._preprocessor = Preprocessor(
            target_width=preproc_cfg.get("target_width", 640),
            target_height=preproc_cfg.get("target_height", 480),
            clahe_clip_limit=preproc_cfg.get("clahe_clip_limit", 2.0),
            clahe_grid_size=preproc_cfg.get("clahe_grid_size", 8),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            calibration_size=(
                int(self._camera_config.get("width", 640)),
                int(self._camera_config.get("height", 480)),
            ),
            undistort=preproc_cfg.get("undistort", True),
        )
        self._extractor = FeatureExtractor(
            algorithm=feat_cfg.get("algorithm", "ORB"),
            max_features=feat_cfg.get("max_features", 500),
            scale_factor=feat_cfg.get("scale_factor", 1.2),
            n_levels=feat_cfg.get("n_levels", 8),
        )
        self._retrieval = RetrievalIndex()
        self._verifier = GeometricVerifier(
            ratio_test_threshold=match_cfg.get("ratio_test_threshold", 0.75),
            min_matches=match_cfg.get("min_matches", 10),
        )
        self._pose_recovery = PoseRecovery()

        entries = list(database.entries.values())
        self._retrieval.build_index(entries)
        logger.info(
            "VisualLocalizer ready: %d entries, top_k=%d",
            len(entries),
            self._top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def localize(
        self,
        frame: np.ndarray,
        altitude: float,
        heading_deg: float,
    ) -> LocalizationResult:
        """Run the full localization pipeline on a camera frame.

        Args:
            frame: Raw camera image (BGR or grayscale).
            altitude: Drone altitude in metres MSL.
            heading_deg: Drone heading in degrees (clockwise from north).

        Returns:
            A :class:`LocalizationResult` — always returned, never raises.
            Check ``result.success`` before consuming the pose.
        """
        ts = time.time()

        # 1 — Preprocess
        try:
            processed = self._preprocessor.process(frame)
        except Exception as exc:
            logger.warning("Preprocessing failed: %s", exc)
            return self._fail("preprocessing_error", ts)

        # 2 — Feature extraction
        kp, des = self._extractor.detect_and_compute(processed)
        if len(kp) == 0:
            return self._fail("no_features", ts)
        logger.debug("Extracted %d features", len(kp))

        # 3 — Coarse retrieval
        candidates = self._retrieval.query(des, top_k=self._top_k)
        if not candidates:
            return self._fail("no_retrieval_candidates", ts)
        logger.debug(
            "Retrieval: %d candidates %s",
            len(candidates),
            [(cid, cnt) for cid, cnt in candidates],
        )

        # 4 — Geometric verification
        results: List[VerificationResult] = []
        for entry_id, _vote_count in candidates:
            entry = self._database.entries.get(entry_id)
            if entry is None:
                continue
            vr = self._verifier.verify(kp, des, entry)
            if vr is not None:
                results.append(vr)

        if not results:
            return self._fail("no_geometric_match", ts)

        # 5 — Best-match selection
        best = max(results, key=lambda r: r.inlier_count)
        if best.confidence < self._confidence_threshold:
            return self._fail(
                f"low_confidence ({best.confidence:.2f} < "
                f"{self._confidence_threshold})",
                ts,
                confidence=best.confidence,
                inlier_count=best.inlier_count,
                matched_ref_id=best.entry_id,
            )

        # 6 — Pose recovery
        entry = self._database.entries[best.entry_id]
        pose = self._pose_recovery.recover(
            best,
            entry,
            self._camera_config,
            altitude,
            heading_deg,
            self._geo_origin,
        )
        if pose is None:
            return self._fail(
                "pose_recovery_failed",
                ts,
                confidence=best.confidence,
                inlier_count=best.inlier_count,
                matched_ref_id=best.entry_id,
            )

        geodetic, ned, yaw_rad = pose

        logger.info(
            "Localized: ref=%s conf=%.2f inliers=%d lat=%.6f lon=%.6f",
            best.entry_id,
            best.confidence,
            best.inlier_count,
            geodetic[0],
            geodetic[1],
        )

        return LocalizationResult(
            success=True,
            confidence=best.confidence,
            inlier_count=best.inlier_count,
            matched_ref_id=best.entry_id,
            reason="ok",
            pose_ned=ned,
            pose_geodetic=geodetic,
            yaw_rad=yaw_rad,
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fail(
        self,
        reason: str,
        timestamp: float,
        confidence: float = 0.0,
        inlier_count: int = 0,
        matched_ref_id: Optional[str] = None,
    ) -> LocalizationResult:
        logger.debug("Localization failed: %s", reason)
        return LocalizationResult(
            success=False,
            confidence=confidence,
            inlier_count=inlier_count,
            matched_ref_id=matched_ref_id,
            reason=reason,
            pose_ned=None,
            pose_geodetic=None,
            yaw_rad=0.0,
            timestamp=timestamp,
        )

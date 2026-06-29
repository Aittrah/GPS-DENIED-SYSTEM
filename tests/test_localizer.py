"""Integration tests for the VisualLocalizer pipeline."""

import cv2
import numpy as np
import pytest

from vns.vision.bovw_retrieval import BoVWIndex
from vns.vision.localizer import VisualLocalizer
from vns.vision.types import LocalizationResult


class TestVisualLocalizer:

    def test_localize_identical_frame(self, sample_config, sample_database, textured_image):
        """Querying with the same image used to build the DB should succeed."""
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(textured_image, altitude=580.0, heading_deg=0.0)

        assert isinstance(result, LocalizationResult)
        assert result.success
        assert result.confidence > 0
        assert result.inlier_count > 0
        assert result.matched_ref_id is not None
        assert result.pose_geodetic is not None
        assert result.pose_ned is not None
        assert result.reason == "ok"

    def test_localize_shifted_frame(self, sample_config, sample_database, textured_image):
        """A slightly shifted version of a DB image should still localize."""
        shifted = np.roll(textured_image, 10, axis=1)
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(shifted, altitude=580.0, heading_deg=0.0)

        assert result.success
        assert result.pose_geodetic is not None
        lat, lon, alt = result.pose_geodetic
        assert 33.0 < lat < 34.0
        assert 73.0 < lon < 74.0

    def test_localize_bgr_input(self, sample_config, sample_database, textured_image):
        """The pipeline should handle BGR colour frames."""
        bgr = cv2.cvtColor(textured_image, cv2.COLOR_GRAY2BGR)
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(bgr, altitude=580.0, heading_deg=0.0)
        assert result.success

    def test_localize_blank_frame(self, sample_config, sample_database):
        """A featureless black frame should fail gracefully."""
        blank = np.zeros((480, 640), dtype=np.uint8)
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(blank, altitude=580.0, heading_deg=0.0)

        assert not result.success
        assert result.pose_geodetic is None
        assert result.pose_ned is None
        assert result.reason in ("no_features", "no_retrieval_candidates", "no_geometric_match")

    def test_localize_unrelated_scene(self, sample_config, sample_database):
        """A random-noise image unrelated to the DB should fail."""
        rng = np.random.RandomState(999)
        noise = rng.randint(0, 256, (480, 640), dtype=np.uint8)
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(noise, altitude=580.0, heading_deg=0.0)

        assert not result.success or result.confidence < 0.5

    def test_localize_resized_input(self, sample_config, sample_database, textured_image):
        """Input at a different resolution should be resized and still work."""
        big = cv2.resize(textured_image, (1280, 960))
        loc = VisualLocalizer(sample_config, sample_database)
        result = loc.localize(big, altitude=580.0, heading_deg=0.0)
        assert isinstance(result, LocalizationResult)

    def test_result_fields_always_present(self, sample_config, sample_database, textured_image):
        """Every result must have all fields regardless of success."""
        loc = VisualLocalizer(sample_config, sample_database)

        success_result = loc.localize(textured_image, 580.0, 0.0)
        blank_result = loc.localize(np.zeros((480, 640), dtype=np.uint8), 580.0, 0.0)

        for r in (success_result, blank_result):
            assert isinstance(r.success, bool)
            assert isinstance(r.confidence, float)
            assert isinstance(r.inlier_count, int)
            assert isinstance(r.reason, str)
            assert isinstance(r.yaw_rad, float)
            assert isinstance(r.timestamp, float)

    def test_localize_with_bovw_mode(
        self,
        sample_bovw_config,
        sample_bovw_database,
        textured_image,
    ):
        loc = VisualLocalizer(sample_bovw_config, sample_bovw_database)
        result = loc.localize(textured_image, altitude=580.0, heading_deg=0.0)

        assert loc._retrieval.backend_name == "bovw"
        assert result.success
        assert result.matched_ref_id == "synth_001"

    def test_bovw_query_path_is_used(
        self,
        monkeypatch,
        sample_bovw_config,
        sample_bovw_database,
        textured_image,
    ):
        calls = {"count": 0}
        original_query = BoVWIndex.query

        def spy_query(self, descriptors, k=5):
            calls["count"] += 1
            return original_query(self, descriptors, k=k)

        monkeypatch.setattr(BoVWIndex, "query", spy_query)

        loc = VisualLocalizer(sample_bovw_config, sample_bovw_database)
        result = loc.localize(textured_image, altitude=580.0, heading_deg=0.0)

        assert result.success
        assert calls["count"] == 1

    def test_bovw_missing_index_raises(self, sample_bovw_config, sample_database):
        with pytest.raises(ValueError, match="BoVW retrieval selected"):
            VisualLocalizer(sample_bovw_config, sample_database)

    def test_bovw_can_fallback_to_flann(
        self,
        sample_bovw_fallback_config,
        sample_database,
        textured_image,
    ):
        loc = VisualLocalizer(sample_bovw_fallback_config, sample_database)
        result = loc.localize(textured_image, altitude=580.0, heading_deg=0.0)

        assert loc._retrieval.backend_name == "flann"
        assert result.success

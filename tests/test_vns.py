import json
import math
import os
import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np
import cv2
import yaml

from vns.config.config_manager import ConfigManager
from vns.database.reference_db import ReferenceDatabase, DatabaseEntry
from vns.utils.coordinates import (
    geodetic_to_enu,
    enu_to_geodetic,
    geodetic_to_ecef,
    ecef_to_geodetic,
)
from vns.vision.extractor import FeatureExtractor
from vns.vision.matcher import FeatureMatcher
from vns.core.gnss_monitor import GnssMonitor, GnssState
from vns.core.blender import PositionBlender
from vns.validation.accuracy_report import AccuracyThresholds, load_and_generate_report

class TestVnsCoordinates(unittest.TestCase):
    """Test coordinate conversions (WGS-84)."""

    def test_ecef_geodetic_roundtrip(self):
        # Coordinates around Islamabad (QAU Campus)
        lat, lon, alt = 33.747, 73.137, 580.0
        
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat_rt, lon_rt, alt_rt = ecef_to_geodetic(x, y, z)
        
        self.assertAlmostEqual(lat, lat_rt, places=5)
        self.assertAlmostEqual(lon, lon_rt, places=5)
        self.assertAlmostEqual(alt, alt_rt, places=2)

    def test_enu_roundtrip(self):
        origin_lat, origin_lon, origin_alt = 33.747, 73.137, 550.0
        target_lat, target_lon, target_alt = 33.748, 73.138, 580.0
        
        e, n, u = geodetic_to_enu(
            target_lat, target_lon, target_alt,
            origin_lat, origin_lon, origin_alt
        )
        
        # Test that ENU is sane
        self.assertTrue(e > 0) # East is positive
        self.assertTrue(n > 0) # North is positive
        self.assertTrue(u > 0) # Up is positive (alt: 580 > 550)
        
        lat_rt, lon_rt, alt_rt = enu_to_geodetic(
            e, n, u,
            origin_lat, origin_lon, origin_alt
        )
        
        self.assertAlmostEqual(target_lat, lat_rt, places=6)
        self.assertAlmostEqual(target_lon, lon_rt, places=6)
        self.assertAlmostEqual(target_alt, alt_rt, places=2)

class TestVnsConfig(unittest.TestCase):
    """Test ConfigManager parsing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        self.config_data = {
            "camera": {
                "width": 640,
                "height": 480,
                "fx": 550.0
            },
            "matching": {
                "confidence_threshold": 0.6
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_and_get(self):
        cfg = ConfigManager.load(self.config_path)
        self.assertEqual(cfg.get("camera.width"), 640)
        self.assertEqual(cfg.get("camera.fx"), 550.0)
        self.assertEqual(cfg.get("matching.confidence_threshold"), 0.6)
        self.assertEqual(cfg.get("nonexistent.field", "default"), "default")

class TestVnsDatabase(unittest.TestCase):
    """Test ReferenceDatabase functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.vnsdb")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_database_indexing(self):
        db = ReferenceDatabase(name="Test DB")
        
        # Add pre-computed entry
        entry = DatabaseEntry(
            id="test_01",
            source_path="mock.jpg",
            latitude=33.7470,
            longitude=73.1370,
            altitude=580.0,
            heading=0.0,
            capture_time="2024-01-01",
            feature_count=100,
            feature_algorithm="ORB",
            keypoints=np.random.rand(100, 2) * 500,
            descriptors=np.random.randint(0, 256, (100, 32), dtype=np.uint8)
        )
        db.add_entry(entry)
        
        self.assertEqual(db.entry_count, 1)
        self.assertAlmostEqual(db.bounds.min_lat, 33.7470)
        
        # Save and Load
        db.save(self.db_path)
        db_loaded = ReferenceDatabase.load(self.db_path)
        
        self.assertEqual(db_loaded.name, "Test DB")
        self.assertEqual(db_loaded.entry_count, 1)
        self.assertEqual(db_loaded.entries["test_01"].feature_count, 100)
        
        # Geographic query
        results = db_loaded.query_region(33.7471, 73.1371, 0.001)
        self.assertEqual(len(results), 1)
        
        results_far = db_loaded.query_region(34.0, 74.0, 0.001)
        self.assertEqual(len(results_far), 0)

class TestVnsVision(unittest.TestCase):
    """Test feature extraction and matching modules."""

    def test_extractor(self):
        # Create a simple synthetic image with circles and lines
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(img, (320, 240), 50, 255, -1)
        cv2.line(img, (100, 100), (500, 400), 255, 5)
        
        extractor = FeatureExtractor(max_features=200)
        kp, des = extractor.detect_and_compute(img)
        
        self.assertTrue(len(kp) > 0)
        self.assertEqual(kp.shape[1], 2)
        self.assertEqual(des.shape[1], 32)

class TestGnssMonitor(unittest.TestCase):
    """Test GNSS health classifier."""

    def test_state_transitions(self):
        monitor = GnssMonitor(degraded_hdop=5.0, degraded_satellites=4, denied_timeout_seconds=1.0)
        
        # Starts denied
        self.assertEqual(monitor.state, GnssState.DENIED)
        
        # Healthy GNSS update
        state = monitor.update(has_fix=True, num_satellites=8, hdop=1.2, timestamp=100.0)
        self.assertEqual(state, GnssState.HEALTHY)
        
        # Degraded update
        state = monitor.update(has_fix=True, num_satellites=3, hdop=1.2, timestamp=101.0)
        self.assertEqual(state, GnssState.DEGRADED)
        
        # Degraded due to HDOP
        state = monitor.update(has_fix=True, num_satellites=8, hdop=6.0, timestamp=102.0)
        self.assertEqual(state, GnssState.DEGRADED)
        
        # Completely Denied fix loss
        state = monitor.update(has_fix=False, num_satellites=8, hdop=1.2, timestamp=103.0)
        self.assertEqual(state, GnssState.DENIED)
        
        # Timeout detection
        monitor.update(has_fix=True, num_satellites=8, hdop=1.2, timestamp=104.0)
        monitor.check_timeout(current_time=104.5)
        self.assertEqual(monitor.state, GnssState.HEALTHY)
        
        # After 1.5 seconds (timeout threshold is 1.0)
        monitor.check_timeout(current_time=106.0)
        self.assertEqual(monitor.state, GnssState.DENIED)

class TestPositionBlender(unittest.TestCase):
    """Test transition blending and failsafe logic."""

    def test_blending_modes(self):
        blender = PositionBlender(
            uncertainty_failsafe_threshold=10.0,
            visual_timeout_seconds=5.0,
            fusion_blend_duration=4.0,
            position_continuity_threshold=5.0
        )
        
        gps_pos = (33.747, 73.137, 580.0)
        vis_pos = (33.74701, 73.13701, 580.0)
        
        # GPS healthy mode
        pos, mode = blender.blend(gps_pos, None, GnssState.HEALTHY, current_time=100.0)
        self.assertEqual(mode, "GPS")
        self.assertEqual(pos, gps_pos)
        
        # Switch to vision - start transition
        pos, mode = blender.blend(gps_pos, vis_pos, GnssState.DENIED, current_time=101.0)
        self.assertEqual(mode, "BLENDING")
        
        # Mid-point blending (elapsed 2.0s of 4.0s blend duration => 50% blend)
        pos, mode = blender.blend(gps_pos, vis_pos, GnssState.DENIED, current_time=103.0)
        self.assertEqual(mode, "BLENDING")
        self.assertAlmostEqual(pos[0], 0.5 * gps_pos[0] + 0.5 * vis_pos[0])
        
        # Finished transition (elapsed 5.0s > 4.0s)
        pos, mode = blender.blend(gps_pos, vis_pos, GnssState.DENIED, current_time=106.0)
        self.assertEqual(mode, "VISION")
        self.assertEqual(pos, vis_pos)
        
        # Visual signal dropout timeout
        # Advance time by 6.0 seconds (>5.0 timeout)
        pos, mode = blender.blend(gps_pos, None, GnssState.DENIED, current_time=113.0)
        self.assertEqual(mode, "FAILSAFE")

class TestVnsValidation(unittest.TestCase):
    """Test accuracy reporting logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "ground_truth_test.jsonl")
        
        # Generate dummy log samples
        samples = [
            {"timestamp": 100.0, "horizontal_error": 1.5, "vertical_error": 0.5, "heading_error": 2.0, "mode": "VISION"},
            {"timestamp": 101.0, "horizontal_error": 2.0, "vertical_error": 0.8, "heading_error": 3.0, "mode": "VISION"},
            {"timestamp": 102.0, "horizontal_error": 2.5, "vertical_error": 1.2, "heading_error": 4.0, "mode": "VISION"}
        ]
        
        with open(self.log_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_report_generation(self):
        thresholds = AccuracyThresholds(
            max_mean_horizontal_error_m=3.0,
            max_p95_horizontal_error_m=5.0,
            max_mean_vertical_error_m=2.0,
            max_mean_heading_error_deg=10.0
        )
        
        report = load_and_generate_report(self.log_file, self.temp_dir, thresholds)
        
        self.assertEqual(report.total_samples, 3)
        self.assertEqual(report.duration_seconds, 2.0)
        self.assertAlmostEqual(report.horizontal_error["mean"], 2.0)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.failures), 0)
        
        # Confirm report output files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "report_ground_truth_test.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "report_ground_truth_test.md")))

if __name__ == "__main__":
    unittest.main()

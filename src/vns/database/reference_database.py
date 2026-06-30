import sqlite3
import pickle
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..preprocessing.satellite_preprocessor import SatellitePreprocessor
from ..preprocessing.patch_generator import PatchGenerator
from ..core.geo_utils import GeoPoint, geo_to_ned, ned_to_geo

ORB_PARAMS = {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8}

_PATCH_SAVE_DIR = Path("data/patches/satellite")


def _kp_to_list(kps):
    return [(k.pt[0], k.pt[1], k.size, k.angle,
             k.response, k.octave, k.class_id) for k in kps]


def _list_to_kp(data):
    kps = []
    for d in data:
        k = cv2.KeyPoint(x=d[0], y=d[1], size=d[2], angle=d[3],
                         response=d[4], octave=int(d[5]), class_id=int(d[6]))
        kps.append(k)
    return kps


class ReferenceDatabase:

    def __init__(self, db_path: str = "data/database/reference.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _PATCH_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS patches (
                    patch_id      TEXT PRIMARY KEY,
                    lat           REAL,
                    lon           REAL,
                    alt           REAL,
                    row_idx       INTEGER,
                    col_idx       INTEGER,
                    image_path    TEXT,
                    descriptors   BLOB,
                    keypoints     BLOB,
                    feature_count INTEGER,
                    source_image  TEXT,
                    created_at    TEXT
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_latlon ON patches(lat, lon)"
            )
            con.commit()

    def build(self,
              satellite_image_path: str,
              lat: float,
              lon: float,
              alt: float = 550.0,
              progress_callback=None) -> int:
        preprocessor = SatellitePreprocessor()
        pg = PatchGenerator()
        orb = cv2.ORB_create(**ORB_PARAMS)

        processed = preprocessor.load_and_preprocess(satellite_image_path)
        if processed is None:
            return 0

        origin = GeoPoint(lat, lon, alt)
        center_ned = geo_to_ned(origin, origin)
        meters_per_pixel = (alt * 0.001) if alt > 0 else 0.5
        source_name = Path(satellite_image_path).name

        patches = pg.generate_patches(
            image=processed,
            image_center=center_ned,
            meters_per_pixel=meters_per_pixel,
            source="satellite",
            base_id=Path(satellite_image_path).stem,
        )

        total = len(patches)
        stored = 0
        now = datetime.datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as con:
            for idx, patch in enumerate(patches):
                kps, descs = orb.detectAndCompute(patch.data, None)

                if descs is None or len(kps) == 0:
                    if progress_callback:
                        progress_callback(idx + 1, total)
                    continue

                geo = ned_to_geo(patch.center_position, origin)

                patch_img_path = _PATCH_SAVE_DIR / f"{patch.patch_id}.png"
                cv2.imwrite(str(patch_img_path), patch.data)

                row_idx = idx // max(1, int(
                    (processed.shape[1] - 256) / 128 + 1))
                col_idx = idx % max(1, int(
                    (processed.shape[1] - 256) / 128 + 1))

                con.execute(
                    """INSERT OR REPLACE INTO patches
                       (patch_id, lat, lon, alt,
                        row_idx, col_idx, image_path,
                        descriptors, keypoints, feature_count,
                        source_image, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        patch.patch_id,
                        geo.latitude, geo.longitude, geo.altitude,
                        row_idx, col_idx,
                        str(patch_img_path),
                        pickle.dumps(descs),
                        pickle.dumps(_kp_to_list(kps)),
                        len(kps),
                        source_name,
                        now,
                    )
                )
                stored += 1

                if progress_callback:
                    progress_callback(idx + 1, total)

            con.commit()

        return stored

    def query(self,
              uav_patch: np.ndarray,
              top_k: int = 5,
              lat_center: Optional[float] = None,
              lon_center: Optional[float] = None,
              radius_deg: float = 0.01) -> list:
        orb = cv2.ORB_create(**ORB_PARAMS)
        kps, descs = orb.detectAndCompute(uav_patch, None)

        if descs is None or len(kps) == 0:
            return []

        with sqlite3.connect(self.db_path) as con:
            if lat_center is not None and lon_center is not None:
                rows = con.execute(
                    """SELECT patch_id, lat, lon, alt, image_path,
                              descriptors, feature_count
                       FROM patches
                       WHERE lat BETWEEN ? AND ?
                         AND lon BETWEEN ? AND ?""",
                    (lat_center - radius_deg, lat_center + radius_deg,
                     lon_center - radius_deg, lon_center + radius_deg)
                ).fetchall()
            else:
                rows = con.execute(
                    """SELECT patch_id, lat, lon, alt, image_path,
                              descriptors, feature_count
                       FROM patches"""
                ).fetchall()

        if not rows:
            return []

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        results = []

        for row in rows:
            patch_id, p_lat, p_lon, p_alt, img_path, desc_blob, feat_count = row
            ref_descs = pickle.loads(desc_blob)

            if ref_descs is None or len(ref_descs) == 0:
                continue

            try:
                matches = matcher.match(descs, ref_descs)
            except cv2.error:
                continue

            if not matches:
                continue

            matches = sorted(matches, key=lambda m: m.distance)

            # ratio-style filter on sorted matches (threshold=0.75 of max range)
            good = [m for m in matches if m.distance < 0.75 * 256]
            if not good:
                good = matches[:max(1, len(matches) // 4)]

            avg_dist = sum(m.distance for m in good) / len(good)
            confidence = max(0.0, 1.0 - avg_dist / 256.0)

            results.append({
                "patch_id":    patch_id,
                "lat":         p_lat,
                "lon":         p_lon,
                "alt":         p_alt,
                "confidence":  confidence,
                "match_count": len(good),
                "distance":    avg_dist,
                "image_path":  img_path,
            })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0.0

        with sqlite3.connect(self.db_path) as con:
            total = con.execute(
                "SELECT COUNT(*) FROM patches"
            ).fetchone()[0]

            unique = con.execute(
                "SELECT COUNT(DISTINCT source_image) FROM patches"
            ).fetchone()[0]

            bounds = con.execute(
                "SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM patches"
            ).fetchone()

            last = con.execute(
                "SELECT MAX(created_at) FROM patches"
            ).fetchone()[0]

        return {
            "total_patches":  total,
            "unique_images":  unique,
            "lat_range":      (bounds[0], bounds[1]) if bounds[0] else (None, None),
            "lon_range":      (bounds[2], bounds[3]) if bounds[2] else (None, None),
            "last_built":     last,
            "db_size_mb":     round(db_size, 3),
        }

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM patches")
            con.commit()

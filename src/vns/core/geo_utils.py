# src/vns/core/geo_utils.py
"""
Single place where WGS84 <-> NED conversion happens.
NO other module should do coordinate math.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GeoPoint:
    """WGS84 coordinates — only used at system boundaries"""
    latitude: float   # degrees
    longitude: float  # degrees  
    altitude: float   # meters MSL

@dataclass(frozen=True)
class NEDPoint:
    """Local NED frame — used internally everywhere"""
    north: float  # meters from origin
    east: float   # meters from origin
    down: float   # meters (negative = up)

# QAU Campus origin
QAU_ORIGIN = GeoPoint(33.7470, 73.1370, 550.0)

def geo_to_ned(point: GeoPoint, origin: GeoPoint = QAU_ORIGIN) -> NEDPoint:
    """Convert WGS84 to local NED meters"""
    lat_rad = np.radians(origin.latitude)
    dlat = np.radians(point.latitude - origin.latitude)
    dlon = np.radians(point.longitude - origin.longitude)
    R = 6378137.0  # Earth radius meters
    north = dlat * R
    east = dlon * R * np.cos(lat_rad)
    down = -(point.altitude - origin.altitude)
    return NEDPoint(north, east, down)

def ned_to_geo(point: NEDPoint, origin: GeoPoint = QAU_ORIGIN) -> GeoPoint:
    """Convert local NED meters to WGS84"""
    R = 6378137.0
    lat_rad = np.radians(origin.latitude)
    dlat = point.north / R
    dlon = point.east / (R * np.cos(lat_rad))
    return GeoPoint(
        latitude=origin.latitude + np.degrees(dlat),
        longitude=origin.longitude + np.degrees(dlon),
        altitude=origin.altitude - point.down
    )
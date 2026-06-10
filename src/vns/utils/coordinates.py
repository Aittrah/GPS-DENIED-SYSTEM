import math
from typing import Tuple

# WGS-84 ellipsoid constants
WGS84_A = 6378137.0         # semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # eccentricity squared
WGS84_E_PRIME2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # second eccentricity squared

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert geodetic coordinates (lat, lon, alt) to ECEF (X, Y, Z)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    
    return x, y, z

def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates (X, Y, Z) to geodetic (lat, lon, alt) using Bowring's method."""
    p = math.sqrt(x**2 + y**2)
    if p < 1e-6:
        # Near pole
        lat_deg = 90.0 if z >= 0 else -90.0
        lon_deg = 0.0
        alt_m = abs(z) - WGS84_B
        return lat_deg, lon_deg, alt_m
        
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    
    lat = math.atan2(
        z + WGS84_E_PRIME2 * WGS84_B * math.sin(theta)**3,
        p - WGS84_E2 * WGS84_A * math.cos(theta)**3
    )
    
    lon = math.atan2(y, x)
    
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p / math.cos(lat) - n
    
    return math.degrees(lat), math.degrees(lon), alt

def ecef_to_enu(
    x: float, y: float, z: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float
) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to local ENU (East, North, Up) coordinates."""
    ref_x, ref_y, ref_z = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z
    
    lat_rad = math.radians(origin_lat_deg)
    lon_rad = math.radians(origin_lon_deg)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return e, n, u

def enu_to_ecef(
    e: float, n: float, u: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float
) -> Tuple[float, float, float]:
    """Convert local ENU coordinates to ECEF coordinates."""
    ref_x, ref_y, ref_z = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    
    lat_rad = math.radians(origin_lat_deg)
    lon_rad = math.radians(origin_lon_deg)
    
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    
    return ref_x + dx, ref_y + dy, ref_z + dz

def geodetic_to_enu(
    lat_deg: float, lon_deg: float, alt_m: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float
) -> Tuple[float, float, float]:
    """Directly convert geodetic coordinates to local ENU coordinates."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, origin_lat_deg, origin_lon_deg, origin_alt_m)

def enu_to_geodetic(
    e: float, n: float, u: float,
    origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float
) -> Tuple[float, float, float]:
    """Directly convert local ENU coordinates to geodetic coordinates."""
    x, y, z = enu_to_ecef(e, n, u, origin_lat_deg, origin_lon_deg, origin_alt_m)
    return ecef_to_geodetic(x, y, z)

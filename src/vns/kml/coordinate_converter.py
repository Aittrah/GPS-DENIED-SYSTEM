# src/vns/kml/coordinate_converter.py
"""
Converts between coordinate systems:
- WGS84 (lat/lon) — what GPS and KML use
- UTM (meters)    — what path planning needs
- Pixel           — what map display needs
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class UTMPoint:
    easting:  float   # meters East
    northing: float   # meters North
    zone_num: int
    zone_let: str


@dataclass
class PixelPoint:
    x: int
    y: int


class CoordinateConverter:
    """
    Converts lat/lon ↔ UTM ↔ pixels.
    No external libraries needed — pure math.
    """

    def latlon_to_utm(
        self,
        lat: float,
        lon: float
    ) -> UTMPoint:
        """Convert WGS84 lat/lon to UTM coordinates."""
        # UTM zone calculation
        zone_num = int((lon + 180) / 6) + 1

        # Special zones for Norway and Svalbard
        if 56 <= lat < 64:
            if 3 <= lon < 12:
                zone_num = 32

        # Zone letter
        zone_let = self._utm_zone_letter(lat)

        # Convert to radians
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)

        # Central meridian of zone
        lon0_r = math.radians((zone_num - 1) * 6 - 180 + 3)

        # WGS84 constants
        a  = 6378137.0
        f  = 1 / 298.257223563
        b  = a * (1 - f)
        e2 = 1 - (b/a)**2
        e  = math.sqrt(e2)
        n  = (a - b) / (a + b)

        A0 = 1 - e2/4 - 3*e2**2/64 - 5*e2**3/256
        A2 = 3/8 * (e2 + e2**2/4 + 15*e2**3/128)
        A4 = 15/256 * (e2**2 + 3*e2**3/4)
        A6 = 35*e2**3/3072

        M = a * (
            A0*lat_r - A2*math.sin(2*lat_r)
            + A4*math.sin(4*lat_r) - A6*math.sin(6*lat_r)
        )

        sin_lat = math.sin(lat_r)
        cos_lat = math.cos(lat_r)
        tan_lat = math.tan(lat_r)

        N   = a / math.sqrt(1 - e2 * sin_lat**2)
        T   = tan_lat**2
        C   = (e2 / (1 - e2)) * cos_lat**2
        A_c = cos_lat * (lon_r - lon0_r)

        easting = 0.9996 * N * (
            A_c
            + (1 - T + C) * A_c**3 / 6
            + (5 - 18*T + T**2 + 72*C - 58*(e2/(1-e2)))
            * A_c**5 / 120
        ) + 500000.0

        northing = 0.9996 * (
            M + N * tan_lat * (
                A_c**2 / 2
                + (5 - T + 9*C + 4*C**2) * A_c**4 / 24
                + (61 - 58*T + T**2 + 600*C - 330*(e2/(1-e2)))
                * A_c**6 / 720
            )
        )

        if lat < 0:
            northing += 10000000.0

        return UTMPoint(
            easting=easting,
            northing=northing,
            zone_num=zone_num,
            zone_let=zone_let
        )

    def utm_to_latlon(
        self,
        utm: UTMPoint
    ) -> tuple[float, float]:
        """Convert UTM back to lat/lon."""
        x = utm.easting  - 500000.0
        y = utm.northing

        if utm.zone_let < 'N':
            y -= 10000000.0

        lon0 = math.radians(
            (utm.zone_num - 1) * 6 - 180 + 3
        )

        a  = 6378137.0
        f  = 1 / 298.257223563
        b  = a * (1 - f)
        e2 = 1 - (b/a)**2
        k0 = 0.9996

        M  = y / k0
        mu = M / (a * (
            1 - e2/4 - 3*e2**2/64 - 5*e2**3/256
        ))

        e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

        phi1 = (
            mu
            + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
            + (151*e1**3/96) * math.sin(6*mu)
        )

        sin1 = math.sin(phi1)
        cos1 = math.cos(phi1)
        tan1 = math.tan(phi1)

        N1 = a / math.sqrt(1 - e2 * sin1**2)
        R1 = a * (1-e2) / (1 - e2 * sin1**2)**1.5
        T1 = tan1**2
        C1 = e2 / (1-e2) * cos1**2
        D  = x / (N1 * k0)

        lat = phi1 - (N1 * tan1 / R1) * (
            D**2/2
            - (5 + 3*T1 + 10*C1 - 4*C1**2
               - 9*e2/(1-e2)) * D**4/24
            + (61 + 90*T1 + 298*C1 + 45*T1**2
               - 252*e2/(1-e2) - 3*C1**2) * D**6/720
        )

        lon = lon0 + (
            D
            - (1 + 2*T1 + C1) * D**3/6
            + (5 - 2*C1 + 28*T1 - 3*C1**2
               + 8*e2/(1-e2) + 24*T1**2) * D**5/120
        ) / cos1

        return math.degrees(lat), math.degrees(lon)

    def latlon_to_pixel(
        self,
        lat: float,
        lon: float,
        bounds_north: float,
        bounds_south: float,
        bounds_west:  float,
        bounds_east:  float,
        img_width:    int,
        img_height:   int
    ) -> PixelPoint:
        """
        Convert lat/lon to pixel position on displayed map image.
        Used to draw polygon, waypoints, UAV position on screen.
        """
        x = int(
            (lon - bounds_west)
            / (bounds_east - bounds_west)
            * img_width
        )
        y = int(
            (bounds_north - lat)
            / (bounds_north - bounds_south)
            * img_height
        )
        x = max(0, min(x, img_width  - 1))
        y = max(0, min(y, img_height - 1))
        return PixelPoint(x, y)

    def pixel_to_latlon(
        self,
        px: int,
        py: int,
        bounds_north: float,
        bounds_south: float,
        bounds_west:  float,
        bounds_east:  float,
        img_width:    int,
        img_height:   int
    ) -> tuple[float, float]:
        """
        Convert pixel click position to lat/lon.
        Used when user clicks Start Point or Goal Point.
        """
        lon = bounds_west + (px / img_width) * (
            bounds_east - bounds_west
        )
        lat = bounds_north - (py / img_height) * (
            bounds_north - bounds_south
        )
        return lat, lon

    def _utm_zone_letter(self, lat: float) -> str:
        letters = "CDEFGHJKLMNPQRSTUVWXX"
        idx = int((lat + 80) / 8)
        return letters[max(0, min(idx, 20))]
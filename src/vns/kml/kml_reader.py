# src/vns/kml/kml_reader.py
"""
Handles everything related to KML files:
- Parsing polygon coordinates
- Extracting bounds
- Converting to different formats
"""
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class PolygonData:
    """All data extracted from a KML polygon."""
    name:        str
    coordinates: list[tuple[float, float]]  # list of (lat, lon)
    center_lat:  float
    center_lon:  float
    north:       float   # bounding box
    south:       float
    east:        float
    west:        float


class KMLReader:
    """
    Reads Google Earth Pro KML files.
    Extracts polygon coordinates and bounds.
    """

    # KML uses this namespace
    KML_NS = "{http://www.opengis.net/kml/2.2}"

    def read_kml(self, kml_path: str) -> Optional[PolygonData]:
        """
        Parse KML file and extract polygon data.
        Returns PolygonData or None if parsing fails.
        """
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()

            # Try with namespace first
            coords = self._find_coordinates(root, self.KML_NS)

            # If not found try without namespace
            if not coords:
                coords = self._find_coordinates(root, "")

            if not coords:
                print(f"No polygon coordinates found in {kml_path}")
                return None

            # Parse coordinate string
            points = self._parse_coord_string(coords)
            if len(points) < 3:
                print("Polygon needs at least 3 points")
                return None

            # Extract name
            name = self._find_name(root) or Path(kml_path).stem

            # Calculate bounds
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]

            return PolygonData(
                name=name,
                coordinates=points,
                center_lat=sum(lats) / len(lats),
                center_lon=sum(lons) / len(lons),
                north=max(lats),
                south=min(lats),
                east=max(lons),
                west=min(lons)
            )

        except Exception as e:
            print(f"KML parse error: {e}")
            return None

    def _find_coordinates(
        self,
        root: ET.Element,
        ns: str
    ) -> Optional[str]:
        """Find coordinates element in KML tree."""
        # Try Polygon path first
        paths = [
            f".//{ns}Polygon/{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates",
            f".//{ns}coordinates",
        ]
        for path in paths:
            elem = root.find(path)
            if elem is not None and elem.text:
                return elem.text.strip()
        return None

    def _find_name(self, root: ET.Element) -> Optional[str]:
        """Extract placemark name."""
        ns = self.KML_NS
        for path in [f".//{ns}Placemark/{ns}name",
                     ".//name"]:
            elem = root.find(path)
            if elem is not None and elem.text:
                return elem.text.strip()
        return None

    def _parse_coord_string(
        self,
        coord_str: str
    ) -> list[tuple[float, float]]:
        """
        Parse KML coordinate string.
        KML format: lon,lat,alt lon,lat,alt ...
        We return: list of (lat, lon)
        """
        points = []
        for token in coord_str.split():
            token = token.strip()
            if not token:
                continue
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    # alt = float(parts[2]) if len(parts) > 2 else 0
                    points.append((lat, lon))
                except ValueError:
                    continue
        return points

    def save_as_json(
        self,
        polygon: PolygonData,
        output_path: str
    ) -> str:
        """Save polygon data as JSON for later use."""
        data = asdict(polygon)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path

    def load_from_json(self, json_path: str) -> Optional[PolygonData]:
        """Load previously saved polygon data."""
        try:
            with open(json_path) as f:
                data = json.load(f)
            return PolygonData(**data)
        except Exception as e:
            print(f"JSON load error: {e}")
            return None
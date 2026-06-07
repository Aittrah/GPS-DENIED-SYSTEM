# src/vns/navigation/path_planner.py
"""
A* path planning inside polygon boundary.
Works in UTM coordinates (meters).
"""
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional
from ..kml.coordinate_converter import CoordinateConverter, UTMPoint


@dataclass
class Waypoint:
    lat:       float
    lon:       float
    easting:   float
    northing:  float
    index:     int
    dist_from_prev: float = 0.0


@dataclass(order=True)
class AStarNode:
    f_cost: float
    g_cost: float = field(compare=False)
    pos:    tuple = field(compare=False)
    parent: object = field(compare=False, default=None)


class PathPlanner:
    """
    Generates navigation graph inside polygon.
    Runs A* to find shortest path.
    Exports waypoints to CSV and JSON.
    """

    def __init__(self, grid_spacing_m: float = 10.0):
        self._grid   = grid_spacing_m
        self._conv   = CoordinateConverter()

    def plan_path(
        self,
        start_lat:   float,
        start_lon:   float,
        goal_lat:    float,
        goal_lon:    float,
        polygon_coords: list[tuple[float, float]]
    ) -> Optional[list[Waypoint]]:
        """
        Main entry point.
        Takes start/goal in lat/lon.
        Returns list of Waypoints or None if no path.
        """
        # Convert everything to UTM
        start_utm = self._conv.latlon_to_utm(start_lat, start_lon)
        goal_utm  = self._conv.latlon_to_utm(goal_lat,  goal_lon)
        poly_utm  = [
            self._conv.latlon_to_utm(lat, lon)
            for lat, lon in polygon_coords
        ]

        # Build grid
        grid_nodes = self._build_grid(poly_utm)
        if not grid_nodes:
            print("Could not build grid inside polygon")
            return None

        # Snap start and goal to nearest grid nodes
        start_node = self._nearest_node(
            grid_nodes, start_utm.easting, start_utm.northing
        )
        goal_node  = self._nearest_node(
            grid_nodes, goal_utm.easting, goal_utm.northing
        )

        # Run A*
        path_utm = self._astar(
            grid_nodes, start_node, goal_node, poly_utm
        )

        if not path_utm:
            print("No path found")
            return None

        # Convert path back to lat/lon waypoints
        waypoints = []
        prev_e, prev_n = None, None

        for i, (e, n) in enumerate(path_utm):
            lat, lon = self._conv.utm_to_latlon(
                UTMPoint(e, n, start_utm.zone_num,
                         start_utm.zone_let)
            )
            dist = 0.0
            if prev_e is not None:
                dist = math.sqrt(
                    (e - prev_e)**2 + (n - prev_n)**2
                )
            waypoints.append(Waypoint(
                lat=lat, lon=lon,
                easting=e, northing=n,
                index=i,
                dist_from_prev=dist
            ))
            prev_e, prev_n = e, n

        return waypoints

    def _build_grid(
        self,
        poly_utm: list
    ) -> list[tuple[float, float]]:
        """Build grid of points inside polygon."""
        if not poly_utm:
            return []

        e_vals = [p.easting  for p in poly_utm]
        n_vals = [p.northing for p in poly_utm]
        e_min, e_max = min(e_vals), max(e_vals)
        n_min, n_max = min(n_vals), max(n_vals)

        nodes = []
        e = e_min
        while e <= e_max:
            n = n_min
            while n <= n_max:
                if self._point_in_polygon(e, n, poly_utm):
                    nodes.append((e, n))
                n += self._grid
            e += self._grid

        return nodes

    def _point_in_polygon(
        self,
        px: float,
        py: float,
        polygon: list
    ) -> bool:
        """Ray casting algorithm."""
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi = polygon[i].easting
            yi = polygon[i].northing
            xj = polygon[j].easting
            yj = polygon[j].northing
            if ((yi > py) != (yj > py) and
                    px < (xj-xi) * (py-yi) / (yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def _nearest_node(
        self,
        nodes: list,
        e: float,
        n: float
    ) -> tuple[float, float]:
        """Find closest grid node to a point."""
        return min(
            nodes,
            key=lambda node: math.sqrt(
                (node[0]-e)**2 + (node[1]-n)**2
            )
        )

    def _astar(
        self,
        nodes:      list,
        start:      tuple,
        goal:       tuple,
        polygon:    list
    ) -> Optional[list[tuple]]:
        """A* search on grid nodes."""
        node_set = set(nodes)

        def heuristic(a, b):
            return math.sqrt(
                (a[0]-b[0])**2 + (a[1]-b[1])**2
            )

        def neighbors(node):
            e, n = node
            candidates = [
                (e+self._grid, n),
                (e-self._grid, n),
                (e, n+self._grid),
                (e, n-self._grid),
                (e+self._grid, n+self._grid),
                (e-self._grid, n+self._grid),
                (e+self._grid, n-self._grid),
                (e-self._grid, n-self._grid),
            ]
            return [
                c for c in candidates
                if c in node_set
            ]

        open_heap = []
        heapq.heappush(
            open_heap,
            AStarNode(0, 0, start, None)
        )

        came_from = {start: None}
        g_cost    = {start: 0.0}
        visited   = set()

        while open_heap:
            current = heapq.heappop(open_heap)

            if current.pos in visited:
                continue
            visited.add(current.pos)

            if (abs(current.pos[0] - goal[0]) <= self._grid
                    and abs(current.pos[1] - goal[1]) <= self._grid):
                # Reconstruct path
                path = []
                pos = current.pos
                while pos is not None:
                    path.append(pos)
                    pos = came_from.get(pos)
                return list(reversed(path))

            for nb in neighbors(current.pos):
                new_g = g_cost[current.pos] + heuristic(
                    current.pos, nb
                )
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    f = new_g + heuristic(nb, goal)
                    heapq.heappush(
                        open_heap,
                        AStarNode(f, new_g, nb, current.pos)
                    )
                    came_from[nb] = current.pos

        return None

    def export_waypoints_csv(
        self,
        waypoints: list[Waypoint],
        output_path: str
    ) -> str:
        """Export waypoints as CSV."""
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "latitude", "longitude",
                "easting_m", "northing_m",
                "dist_from_prev_m"
            ])
            for wp in waypoints:
                writer.writerow([
                    wp.index,
                    round(wp.lat, 8),
                    round(wp.lon, 8),
                    round(wp.easting, 2),
                    round(wp.northing, 2),
                    round(wp.dist_from_prev, 2)
                ])
        print(f"Waypoints CSV saved: {output_path}")
        return output_path

    def export_waypoints_json(
        self,
        waypoints: list[Waypoint],
        output_path: str
    ) -> str:
        """Export waypoints as JSON."""
        import json
        from dataclasses import asdict
        data = {
            "total_waypoints": len(waypoints),
            "total_distance_m": sum(
                w.dist_from_prev for w in waypoints
            ),
            "waypoints": [asdict(w) for w in waypoints]
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Waypoints JSON saved: {output_path}")
        return output_path
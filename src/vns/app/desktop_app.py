# src/vns/desktop_app.py
"""
VNS — GNSS-Free Navigation System
Professional Desktop Application
Final Year Project 2026
Muhammad Ahsan & Aittrah Sardar
IIT Quaid-i-Azam University Islamabad
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import sys
import json
import csv
import math
import heapq
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Ensure project root is on sys.path regardless of how the app is launched
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.vns.app.google_earth import (
    DEFAULT_FOCUS_KML_NAME,
    GoogleEarthLaunchResult,
    open_google_earth_at_location,
)

logger = logging.getLogger("vns.app.desktop")

try:
    from src.vns.database.reference_database import ReferenceDatabase
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

try:
    from src.vns.database.faiss_database import FAISSDatabase
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolygonData:
    name:        str
    coordinates: list
    center_lat:  float
    center_lon:  float
    north:       float
    south:       float
    east:        float
    west:        float


@dataclass
class Waypoint:
    index:    int
    lat:      float
    lon:      float
    easting:  float = 0.0
    northing: float = 0.0
    dist_m:   float = 0.0


@dataclass(order=True)
class ANode:
    f:      float
    g:      float  = field(compare=False)
    pos:    tuple  = field(compare=False)
    parent: object = field(compare=False, default=None)


# ─────────────────────────────────────────────────────────────────────────────
# KML READER
# ─────────────────────────────────────────────────────────────────────────────

class KMLReader:

    NS = "{http://www.opengis.net/kml/2.2}"

    def read(self, path: str) -> Optional[PolygonData]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            coords_str = None
            for ns in [self.NS, ""]:
                for p in [
                    f".//{ns}Polygon/{ns}outerBoundaryIs"
                    f"/{ns}LinearRing/{ns}coordinates",
                    f".//{ns}coordinates",
                ]:
                    el = root.find(p)
                    if el is not None and el.text:
                        coords_str = el.text.strip()
                        break
                if coords_str:
                    break

            if not coords_str:
                return None

            points = []
            for token in coords_str.split():
                parts = token.strip().split(",")
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        points.append((lat, lon))
                    except ValueError:
                        continue

            if len(points) < 3:
                return None

            name = "Mission Area"
            for ns in [self.NS, ""]:
                el = root.find(f".//{ns}name")
                if el is not None and el.text:
                    name = el.text.strip()
                    break

            lats = [p[0] for p in points]
            lons = [p[1] for p in points]

            return PolygonData(
                name=name,
                coordinates=points,
                center_lat=sum(lats) / len(lats),
                center_lon=sum(lons) / len(lons),
                north=max(lats), south=min(lats),
                east=max(lons),  west=min(lons),
            )
        except Exception as e:
            print(f"KML error: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

class CoordConverter:

    def latlon_to_utm(self, lat: float, lon: float):
        zone = int((lon + 180) / 6) + 1
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        lon0  = math.radians((zone - 1) * 6 - 180 + 3)

        a  = 6378137.0
        f  = 1 / 298.257223563
        b  = a * (1 - f)
        e2 = 1 - (b / a) ** 2
        k0 = 0.9996

        N  = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
        T  = math.tan(lat_r) ** 2
        C  = e2 / (1 - e2) * math.cos(lat_r) ** 2
        A_ = math.cos(lat_r) * (lon_r - lon0)

        M = a * (
            (1 - e2/4 - 3*e2**2/64) * lat_r
            - (3*e2/8 + 3*e2**2/32) * math.sin(2*lat_r)
            + (15*e2**2/256) * math.sin(4*lat_r)
        )

        E = (k0 * N * (
            A_ + (1-T+C)*A_**3/6
            + (5-18*T+T**2)*A_**5/120
        ) + 500000)

        Nv = k0 * (
            M + N * math.tan(lat_r) * (
                A_**2/2
                + (5-T+9*C+4*C**2)*A_**4/24
                + (61-58*T+T**2)*A_**6/720
            )
        )

        if lat < 0:
            Nv += 10000000
        return E, Nv, zone

    def latlon_to_pixel(
        self, lat, lon, n, s, w, e, iw, ih
    ):
        x = int((lon - w) / (e - w) * iw)
        y = int((n - lat) / (n - s) * ih)
        return (
            max(0, min(x, iw - 1)),
            max(0, min(y, ih - 1)),
        )

    def pixel_to_latlon(
        self, px, py, n, s, w, e, iw, ih
    ):
        lon = w + (px / iw) * (e - w)
        lat = n - (py / ih) * (n - s)
        return lat, lon

    def haversine_m(
        self,
        lat1, lon1,
        lat2, lon2
    ) -> float:
        R  = 6371000
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a  = (
            math.sin(dp/2)**2
            + math.cos(p1) * math.cos(p2)
            * math.sin(dl/2)**2
        )
        return R * 2 * math.atan2(
            math.sqrt(a), math.sqrt(1 - a)
        )


# ─────────────────────────────────────────────────────────────────────────────
# A* PATH PLANNER
# ─────────────────────────────────────────────────────────────────────────────

class PathPlanner:

    def __init__(self, spacing: float = 15.0):
        self.spacing = spacing
        self.conv    = CoordConverter()

    def plan(
        self,
        s_lat, s_lon,
        g_lat, g_lon,
        poly: list
    ) -> Optional[list]:

        sE, sN, zone = self.conv.latlon_to_utm(s_lat, s_lon)
        gE, gN, _    = self.conv.latlon_to_utm(g_lat, g_lon)
        pUTM = [
            self.conv.latlon_to_utm(la, lo)[:2]
            for la, lo in poly
        ]

        grid = self._build_grid(pUTM)
        if not grid:
            return None

        start = self._snap(grid, sE, sN)
        goal  = self._snap(grid, gE, gN)
        path  = self._astar(set(grid), start, goal)
        if not path:
            return None

        wps  = []
        pE = pN = None
        for i, (e, n) in enumerate(path):
            dlat = (n - sN) / 111320
            dlon = (e - sE) / (
                111320 * math.cos(math.radians(s_lat))
            )
            lat = s_lat + dlat
            lon = s_lon + dlon
            d   = 0.0
            if pE is not None:
                d = math.sqrt(
                    (e-pE)**2 + (n-pN)**2
                )
            wps.append(Waypoint(i, lat, lon, e, n, d))
            pE, pN = e, n

        return wps

    def _build_grid(self, poly):
        if not poly:
            return []
        es = [p[0] for p in poly]
        ns = [p[1] for p in poly]
        nodes = []
        e = min(es)
        while e <= max(es):
            n = min(ns)
            while n <= max(ns):
                if self._pip(e, n, poly):
                    nodes.append((e, n))
                n += self.spacing
            e += self.spacing
        return nodes

    def _pip(self, px, py, poly):
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > py) != (yj > py):
                if px < (xj-xi)*(py-yi)/(yj-yi)+xi:
                    inside = not inside
            j = i
        return inside

    def _snap(self, nodes, e, n):
        return min(
            nodes,
            key=lambda nd: math.sqrt(
                (nd[0]-e)**2 + (nd[1]-n)**2
            )
        )

    def _astar(self, grid, start, goal):
        def h(a, b):
            return math.sqrt(
                (a[0]-b[0])**2 + (a[1]-b[1])**2
            )

        def neighbors(pos):
            e, n = pos
            s = self.spacing
            cands = [
                (e+s,n),(e-s,n),(e,n+s),(e,n-s),
                (e+s,n+s),(e-s,n+s),
                (e+s,n-s),(e-s,n-s),
            ]
            return [c for c in cands if c in grid]

        heap  = []
        heapq.heappush(heap, ANode(0, 0, start))
        came  = {start: None}
        gcost = {start: 0.0}
        seen  = set()

        while heap:
            cur = heapq.heappop(heap)
            if cur.pos in seen:
                continue
            seen.add(cur.pos)

            if h(cur.pos, goal) <= self.spacing * 1.5:
                path = []
                pos  = cur.pos
                while pos is not None:
                    path.append(pos)
                    pos = came.get(pos)
                return list(reversed(path))

            for nb in neighbors(cur.pos):
                ng = gcost[cur.pos] + h(cur.pos, nb)
                if nb not in gcost or ng < gcost[nb]:
                    gcost[nb] = ng
                    f = ng + h(nb, goal)
                    heapq.heappush(
                        heap, ANode(f, ng, nb, cur.pos)
                    )
                    came[nb] = cur.pos
        return None


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

class T:
    BG_DARK   = "#0d1117"
    BG_PANEL  = "#161b22"
    BG_CARD   = "#21262d"
    BG_INPUT  = "#0d1117"
    BG_HOVER  = "#30363d"

    ACCENT    = "#58a6ff"
    SUCCESS   = "#3fb950"
    WARNING   = "#d29922"
    DANGER    = "#f85149"
    PURPLE    = "#bc8cff"
    WHITE     = "#e6edf3"

    TEXT_PRI  = "#e6edf3"
    TEXT_SEC  = "#8b949e"
    TEXT_MUTED= "#484f58"
    BORDER    = "#30363d"

    FONT_HEAD = ("Segoe UI", 11, "bold")
    FONT_BODY = ("Segoe UI", 10)
    FONT_SML  = ("Segoe UI", 9)
    FONT_MONO = ("Consolas", 9)
    FONT_LBL  = ("Segoe UI", 9)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

class ProfButton(tk.Button):

    STYLES = {
        "primary": ("#1f6feb", "#388bfd"),
        "success": ("#238636", "#2ea043"),
        "danger":  ("#b91c1c", "#f85149"),
        "ghost":   ("#21262d", "#30363d"),
        "purple":  ("#6e40c9", "#9461fb"),
        "warning": ("#9e6a03", "#d29922"),
    }

    def __init__(
        self, parent, text, cmd,
        style="primary", icon="", **kw
    ):
        bg, hov = self.STYLES.get(
            style, self.STYLES["primary"]
        )
        label = f"{icon}  {text}" if icon else text
        super().__init__(
            parent, text=label, command=cmd,
            bg=bg, fg=T.TEXT_PRI,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0,
            padx=14, pady=8,
            cursor="hand2",
            activebackground=hov,
            activeforeground=T.TEXT_PRI,
            **kw
        )
        self.bind("<Enter>", lambda e: self.config(bg=hov))
        self.bind("<Leave>", lambda e: self.config(bg=bg))
        self._bg  = bg
        self._hov = hov


class StatusBadge(tk.Label):

    COLORS = {
        "idle":    (T.TEXT_SEC,  T.BG_HOVER),
        "ok":      (T.BG_DARK,   T.SUCCESS),
        "working": (T.BG_DARK,   T.WARNING),
        "error":   (T.BG_DARK,   T.DANGER),
        "info":    (T.BG_DARK,   T.ACCENT),
    }

    def __init__(self, parent, text="IDLE"):
        fg, bg = self.COLORS["idle"]
        super().__init__(
            parent,
            text=f"  {text}  ",
            font=("Segoe UI", 8, "bold"),
            fg=fg, bg=bg,
            relief=tk.FLAT,
            padx=4, pady=2
        )

    def set(self, text, color="idle"):
        fg, bg = self.COLORS.get(
            color, self.COLORS["idle"]
        )
        self.config(
            text=f"  {text}  ", fg=fg, bg=bg
        )


def make_entry(parent, default="", w=16,
               hi_color=None):
    e = tk.Entry(
        parent,
        bg=T.BG_INPUT, fg=T.TEXT_PRI,
        insertbackground=T.ACCENT,
        font=("Consolas", 10),
        relief=tk.FLAT, bd=0, width=w,
        highlightthickness=1,
        highlightcolor=hi_color or T.ACCENT,
        highlightbackground=T.BORDER,
    )
    e.insert(0, default)
    return e


def section_label(parent, text, color=T.ACCENT):
    f = tk.Frame(parent, bg=T.BG_PANEL)
    f.pack(fill=tk.X, padx=12, pady=(16, 4))
    tk.Label(
        f, text=f" {text[:2]} ",
        font=("Segoe UI", 8, "bold"),
        fg=T.BG_DARK, bg=color,
        padx=3, pady=1
    ).pack(side=tk.LEFT)
    tk.Label(
        f, text=f"  {text[3:]}",
        font=("Segoe UI", 10, "bold"),
        fg=T.TEXT_PRI, bg=T.BG_PANEL
    ).pack(side=tk.LEFT)
    tk.Frame(
        parent, bg=T.BORDER, height=1
    ).pack(fill=tk.X, padx=12)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class VNSApp:
    KML_DIR = _PROJECT_ROOT / "data" / "kml"
    GENERATED_KML_NAMES = {
        "fly_to.kml",
        DEFAULT_FOCUS_KML_NAME,
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(
            "VNS  —  GNSS-Free Navigation System"
        )
        self.root.geometry("1500x880")
        self.root.minsize(1100, 650)
        self.root.configure(bg=T.BG_DARK)

        # ── State ──────────────────────────────────────
        self._polygon:    Optional[PolygonData] = None
        self._satellite:  Optional[np.ndarray]  = None
        self._start:      Optional[tuple]        = None
        self._goal:       Optional[tuple]        = None
        self._waypoints:  Optional[list]         = None
        self._click_mode: Optional[str]          = None
        self._img_bounds: Optional[dict]         = None
        self._last_kml_t: float                  = 0.0

        # ── Helpers ────────────────────────────────────
        self._kml  = KMLReader()
        self._conv = CoordConverter()
        self._plan = PathPlanner(spacing=15.0)

        self._build_ui()
        self._start_kml_watcher()

    # ═════════════════════════════════════════════════
    # UI  BUILD
    # ═════════════════════════════════════════════════

    def _build_ui(self):
        self._build_navbar()
        body = tk.Frame(self.root, bg=T.BG_DARK)
        body.pack(fill=tk.BOTH, expand=True,
                  padx=10, pady=(0, 10))
        self._build_sidebar(body)
        self._build_map_area(body)

    # ── Navbar ────────────────────────────────────────

    def _build_navbar(self):
        nav = tk.Frame(
            self.root, bg=T.BG_PANEL, height=54
        )
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)
        tk.Frame(
            self.root, bg=T.BORDER, height=1
        ).pack(fill=tk.X)

        tk.Label(
            nav,
            text="⬡  VNS",
            font=("Segoe UI", 15, "bold"),
            fg=T.ACCENT, bg=T.BG_PANEL
        ).pack(side=tk.LEFT, padx=(18, 4), pady=12)

        tk.Label(
            nav,
            text="GNSS-Free Navigation System",
            font=("Segoe UI", 10),
            fg=T.TEXT_SEC, bg=T.BG_PANEL
        ).pack(side=tk.LEFT, pady=12)

        # Right side
        right = tk.Frame(nav, bg=T.BG_PANEL)
        right.pack(side=tk.RIGHT, padx=18)

        self._sys_badge = StatusBadge(right, "READY")
        self._sys_badge.pack(side=tk.RIGHT,
                              padx=(10, 0))

        tk.Label(
            right,
            text="Muhammad Ahsan  &  Aittrah Sardar  |"
                 "  IIT QAU  |  FYP 2026",
            font=("Segoe UI", 9),
            fg=T.TEXT_MUTED, bg=T.BG_PANEL
        ).pack(side=tk.RIGHT)

    # ── Sidebar ───────────────────────────────────────

    def _build_sidebar(self, parent):
        outer = tk.Frame(
            parent, bg=T.BG_PANEL, width=348
        )
        outer.pack(side=tk.LEFT, fill=tk.Y,
                   padx=(0, 10), pady=10)
        outer.pack_propagate(False)

        cvs = tk.Canvas(
            outer, bg=T.BG_PANEL,
            highlightthickness=0, bd=0
        )
        sb = tk.Scrollbar(
            outer, orient="vertical",
            command=cvs.yview
        )
        self._sf = tk.Frame(cvs, bg=T.BG_PANEL)

        self._sf.bind(
            "<Configure>",
            lambda e: cvs.configure(
                scrollregion=cvs.bbox("all")
            )
        )
        cvs.create_window(
            (0, 0), window=self._sf,
            anchor="nw", width=348
        )
        cvs.configure(yscrollcommand=sb.set)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH,
                 expand=True)

        cvs.bind_all(
            "<MouseWheel>",
            lambda e: cvs.yview_scroll(
                -1*(e.delta//120), "units"
            )
        )

        self._fill_sidebar(self._sf)

    def _fill_sidebar(self, p):

        def gap(h=8):
            tk.Frame(p, bg=T.BG_PANEL,
                     height=h).pack(fill=tk.X)

        def card(title=None):
            f = tk.Frame(
                p, bg=T.BG_CARD, relief=tk.FLAT
            )
            f.pack(fill=tk.X, padx=12, pady=4)
            if title:
                tk.Label(
                    f, text=title,
                    font=("Segoe UI", 8, "bold"),
                    fg=T.TEXT_MUTED, bg=T.BG_CARD
                ).pack(anchor="w", padx=14,
                       pady=(10, 0))
            return f

        def row_pair(frame, lbl1, e1, lbl2, e2):
            r = tk.Frame(frame, bg=T.BG_CARD)
            r.pack(fill=tk.X, padx=14, pady=3)
            tk.Label(
                r, text=lbl1, width=4,
                font=T.FONT_LBL,
                fg=T.TEXT_MUTED, bg=T.BG_CARD
            ).pack(side=tk.LEFT)
            e1.pack(side=tk.LEFT, fill=tk.X,
                    expand=True, ipady=5,
                    padx=(4, 10))
            tk.Label(
                r, text=lbl2, width=4,
                font=T.FONT_LBL,
                fg=T.TEXT_MUTED, bg=T.BG_CARD
            ).pack(side=tk.LEFT)
            e2.pack(side=tk.LEFT, fill=tk.X,
                    expand=True, ipady=5,
                    padx=(4, 0))

        # ══ STEP 01 ═══════════════════════════════════
        section_label(p, "01  Area of Interest",
                      T.ACCENT)
        gap(4)

        c1 = card()
        ib = tk.Frame(c1, bg=T.BG_CARD)
        ib.pack(fill=tk.X, padx=14, pady=(10, 4))

        # START row
        tk.Label(
            ib,
            text="▶  START POINT",
            font=("Segoe UI", 8, "bold"),
            fg=T.SUCCESS, bg=T.BG_CARD
        ).pack(anchor="w", pady=(0, 4))

        sr = tk.Frame(ib, bg=T.BG_CARD)
        sr.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            sr, text="Lat", width=4,
            font=T.FONT_LBL,
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT)
        self._e_slat = make_entry(
            sr, "33.7470", hi_color=T.SUCCESS
        )
        self._e_slat.pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, ipady=5, padx=(4, 0)
        )

        sr2 = tk.Frame(ib, bg=T.BG_CARD)
        sr2.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            sr2, text="Lon", width=4,
            font=T.FONT_LBL,
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT)
        self._e_slon = make_entry(
            sr2, "73.1370", hi_color=T.SUCCESS
        )
        self._e_slon.pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, ipady=5, padx=(4, 0)
        )

        # END row
        tk.Label(
            ib,
            text="■  END POINT",
            font=("Segoe UI", 8, "bold"),
            fg=T.DANGER, bg=T.BG_CARD
        ).pack(anchor="w", pady=(0, 4))

        er = tk.Frame(ib, bg=T.BG_CARD)
        er.pack(fill=tk.X, pady=(0, 3))

        tk.Label(
            er, text="Lat", width=4,
            font=T.FONT_LBL,
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT)
        self._e_elat = make_entry(
            er, "33.7550", hi_color=T.DANGER
        )
        self._e_elat.pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, ipady=5, padx=(4, 0)
        )

        er2 = tk.Frame(ib, bg=T.BG_CARD)
        er2.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            er2, text="Lon", width=4,
            font=T.FONT_LBL,
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT)
        self._e_elon = make_entry(
            er2, "73.1460", hi_color=T.DANGER
        )
        self._e_elon.pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, ipady=5, padx=(4, 0)
        )

        # Buttons row
        br = tk.Frame(ib, bg=T.BG_CARD)
        br.pack(fill=tk.X, pady=(0, 6))

        ProfButton(
            br, "Plot Points",
            self._on_plot_points,
            style="success", icon="📍"
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True, padx=(0, 4))

        ProfButton(
            br, "Clear",
            self._on_clear_points,
            style="ghost", icon="✕"
        ).pack(side=tk.LEFT)

        ProfButton(
            ib,
            "Open in Google Earth Pro",
            self._on_open_ge,
            style="primary", icon="🌍"
        ).pack(fill=tk.X)

        self._pts_info = tk.Label(
            ib, text="",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            justify=tk.LEFT
        )
        self._pts_info.pack(
            anchor="w", pady=(8, 0)
        )

        tk.Label(
            ib,
            text=(
                "① Google Earth opens at your location\n"
                "② Add → Polygon  →  draw boundary\n"
                "③ Right-click → Save Place As  →  KML"
            ),
            font=("Segoe UI", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            justify=tk.LEFT
        ).pack(anchor="w", pady=(8, 10))

        # ══ STEP 02 ═══════════════════════════════════
        section_label(p, "02  Import KML",
                      T.SUCCESS)
        gap(4)

        c2 = card()
        f2 = tk.Frame(c2, bg=T.BG_CARD)
        f2.pack(fill=tk.X, padx=14, pady=12)

        # Auto-watch
        wr = tk.Frame(f2, bg=T.BG_CARD)
        wr.pack(fill=tk.X, pady=(0, 8))

        self._watch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            wr,
            text="Auto-detect when KML is saved",
            variable=self._watch_var,
            bg=T.BG_CARD, fg=T.TEXT_SEC,
            selectcolor=T.BG_INPUT,
            activebackground=T.BG_CARD,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        ProfButton(
            f2, "Browse KML File",
            self._on_browse_kml,
            style="ghost", icon="📂"
        ).pack(fill=tk.X, pady=(0, 6))

        self._kml_info = tk.Label(
            f2,
            text="Waiting for KML file...",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_INPUT,
            justify=tk.LEFT,
            wraplength=270,
            anchor="w"
        )
        self._kml_info.pack(
            fill=tk.X, padx=1, pady=1,
            ipadx=8, ipady=6
        )

        gap(4)

        ProfButton(
            f2, "Load Satellite Image",
            self._on_load_sat,
            style="ghost", icon="🛰️"
        ).pack(fill=tk.X, pady=(8, 4))

        self._img_info = tk.Label(
            f2, text="No image loaded",
            font=("Segoe UI", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._img_info.pack(
            anchor="w", pady=(0, 4)
        )

        # ══ STEP 03 ═══════════════════════════════════
        section_label(
            p, "03  Start & Goal on Map",
            T.WARNING
        )
        gap(4)

        c3 = card()
        f3 = tk.Frame(c3, bg=T.BG_CARD)
        f3.pack(fill=tk.X, padx=14, pady=12)

        # Start button + badge
        sr3 = tk.Frame(f3, bg=T.BG_CARD)
        sr3.pack(fill=tk.X, pady=(0, 4))

        self._btn_start = ProfButton(
            sr3, "Set Start on Map",
            self._on_set_start,
            style="success", icon="📍"
        )
        self._btn_start.pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self._badge_start = StatusBadge(
            sr3, "NOT SET"
        )
        self._badge_start.pack(
            side=tk.RIGHT, padx=(8, 0)
        )

        self._lbl_start = tk.Label(
            f3, text="—",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._lbl_start.pack(
            anchor="w", pady=(0, 8)
        )

        # Goal button + badge
        gr3 = tk.Frame(f3, bg=T.BG_CARD)
        gr3.pack(fill=tk.X, pady=(0, 4))

        self._btn_goal = ProfButton(
            gr3, "Set Goal on Map",
            self._on_set_goal,
            style="danger", icon="🎯"
        )
        self._btn_goal.pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self._badge_goal = StatusBadge(
            gr3, "NOT SET"
        )
        self._badge_goal.pack(
            side=tk.RIGHT, padx=(8, 0)
        )

        self._lbl_goal = tk.Label(
            f3, text="—",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._lbl_goal.pack(
            anchor="w", pady=(0, 4)
        )

        tk.Label(
            f3,
            text="Click on map after pressing button above",
            font=("Segoe UI", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(anchor="w")

        # ══ STEP 04 ═══════════════════════════════════
        section_label(
            p, "04  Path Planning",
            T.PURPLE
        )
        gap(4)

        c4 = card()
        f4 = tk.Frame(c4, bg=T.BG_CARD)
        f4.pack(fill=tk.X, padx=14, pady=12)

        gr4 = tk.Frame(f4, bg=T.BG_CARD)
        gr4.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            gr4, text="Grid spacing (m):",
            font=T.FONT_LBL,
            fg=T.TEXT_SEC, bg=T.BG_CARD
        ).pack(side=tk.LEFT)

        self._spacing_var = tk.StringVar(value="15")
        tk.Spinbox(
            gr4,
            from_=5, to=100,
            increment=5,
            textvariable=self._spacing_var,
            width=6,
            bg=T.BG_INPUT, fg=T.TEXT_PRI,
            buttonbackground=T.BG_HOVER,
            font=("Consolas", 10),
            relief=tk.FLAT
        ).pack(side=tk.RIGHT)

        ProfButton(
            f4, "Run A* Path Planning",
            self._on_run_plan,
            style="purple", icon="⚡"
        ).pack(fill=tk.X)

        self._path_info = tk.Label(
            f4, text="No path planned yet",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._path_info.pack(
            anchor="w", pady=(8, 0)
        )

        # ══ STEP 05 ═══════════════════════════════════
        section_label(
            p, "05  Export Waypoints",
            T.TEXT_SEC
        )
        gap(4)

        c5 = card()
        f5 = tk.Frame(c5, bg=T.BG_CARD)
        f5.pack(fill=tk.X, padx=14, pady=12)

        er5 = tk.Frame(f5, bg=T.BG_CARD)
        er5.pack(fill=tk.X)

        ProfButton(
            er5, "Export CSV",
            self._on_export_csv,
            style="ghost", icon="📊"
        ).pack(
            side=tk.LEFT, fill=tk.X,
            expand=True, padx=(0, 4)
        )

        ProfButton(
            er5, "Export JSON",
            self._on_export_json,
            style="ghost", icon="📋"
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True)

        self._export_info = tk.Label(
            f5, text="",
            font=("Segoe UI", 8),
            fg=T.SUCCESS, bg=T.BG_CARD
        )
        self._export_info.pack(
            anchor="w", pady=(6, 0)
        )

        # ══ STEP 06 — DATABASE ════════════════════════
        section_label(
            p, "06  Database  (DINOv2+FAISS)",
            T.PURPLE
        )
        gap(4)

        c6 = card()
        f6 = tk.Frame(c6, bg=T.BG_CARD)
        f6.pack(fill=tk.X, padx=14, pady=12)

        # Lat / Lon / Alt inputs
        for lbl, attr, default in [
            ("Lat", "_db_lat", "33.7470"),
            ("Lon", "_db_lon", "73.1370"),
            ("Alt", "_db_alt", "550.0"),
        ]:
            row = tk.Frame(f6, bg=T.BG_CARD)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, width=4,
                     font=T.FONT_LBL,
                     fg=T.TEXT_MUTED,
                     bg=T.BG_CARD).pack(side=tk.LEFT)
            e = make_entry(row, default, w=14)
            e.pack(side=tk.LEFT, fill=tk.X,
                   expand=True, ipady=4, padx=(4, 0))
            setattr(self, attr, e)

        gap(6)

        ProfButton(
            f6, "Select Satellite Image",
            self._on_db_select_image,
            style="ghost", icon="🛰️"
        ).pack(fill=tk.X, pady=(0, 3))

        self._db_img_lbl = tk.Label(
            f6, text="No image selected",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            wraplength=270, justify=tk.LEFT
        )
        self._db_img_lbl.pack(anchor="w",
                               pady=(0, 6))

        br6 = tk.Frame(f6, bg=T.BG_CARD)
        br6.pack(fill=tk.X, pady=(0, 4))

        ProfButton(
            br6, "Build Database",
            self._on_build_faiss_db,
            style="primary", icon="🗄️"
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True, padx=(0, 3))

        ProfButton(
            br6, "Load",
            self._on_load_faiss_db,
            style="ghost", icon="↑"
        ).pack(side=tk.LEFT)

        self._db_progress = tk.Label(
            f6, text="",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._db_progress.pack(anchor="w",
                               pady=(0, 4))

        self._db_status = tk.Label(
            f6, text="Status: not built",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            justify=tk.LEFT, wraplength=270
        )
        self._db_status.pack(anchor="w",
                             pady=(0, 6))

        ProfButton(
            f6, "Clear Database",
            self._on_clear_faiss_db,
            style="danger", icon="🗑️"
        ).pack(fill=tk.X)

        gap(12)

        # ══ STEP 07 — MATCH UAV IMAGE ══════════════════
        section_label(
            p, "07  Match UAV Image",
            T.WARNING
        )
        gap(4)

        c7 = card()
        f7 = tk.Frame(c7, bg=T.BG_CARD)
        f7.pack(fill=tk.X, padx=14, pady=12)

        ProfButton(
            f7, "Select UAV Image",
            self._on_match_select_uav,
            style="ghost", icon="📷"
        ).pack(fill=tk.X, pady=(0, 3))

        self._match_img_lbl = tk.Label(
            f7, text="No UAV image selected",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            wraplength=270, justify=tk.LEFT
        )
        self._match_img_lbl.pack(anchor="w",
                                  pady=(0, 6))

        ProfButton(
            f7, "Run Matching",
            self._on_run_matching,
            style="purple", icon="⚡"
        ).pack(fill=tk.X, pady=(0, 6))

        self._match_results = tk.Label(
            f7, text="No results yet",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            justify=tk.LEFT, wraplength=270
        )
        self._match_results.pack(anchor="w")

        gap(24)

        # Internal state for DB / match tabs
        self._db_sat_path: Optional[str] = None
        self._uav_match_path: Optional[str] = None
        self._faiss_db: Optional[object] = None

    # ── Map Area ──────────────────────────────────────

    def _build_map_area(self, parent):
        wrap = tk.Frame(
            parent, bg=T.BG_PANEL
        )
        wrap.pack(
            side=tk.LEFT, fill=tk.BOTH,
            expand=True, pady=10
        )

        # Toolbar
        tb = tk.Frame(
            wrap, bg=T.BG_CARD, height=38
        )
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)

        tk.Label(
            tb, text="  MAP VIEW",
            font=("Segoe UI", 9, "bold"),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT, padx=8)

        self._coord_lbl = tk.Label(
            tb, text="",
            font=("Consolas", 9),
            fg=T.ACCENT, bg=T.BG_CARD
        )
        self._coord_lbl.pack(
            side=tk.LEFT, padx=16
        )

        self._mode_badge = StatusBadge(
            tb, "VIEW"
        )
        self._mode_badge.pack(
            side=tk.RIGHT, padx=10
        )

        tk.Frame(
            wrap, bg=T.BORDER, height=1
        ).pack(fill=tk.X)

        # Canvas
        self._map_frame = tk.Frame(
            wrap, bg="#090d12"
        )
        self._map_frame.pack(
            fill=tk.BOTH, expand=True
        )

        self._canvas = tk.Label(
            self._map_frame,
            bg="#090d12",
            cursor="crosshair"
        )
        self._canvas.pack(
            fill=tk.BOTH, expand=True
        )
        self._canvas.bind(
            "<Button-1>", self._on_map_click
        )
        self._canvas.bind(
            "<Motion>", self._on_hover
        )

        # Empty state overlay
        self._empty = tk.Label(
            self._map_frame,
            text=(
                "⬡\n\n"
                "No satellite image loaded\n\n"
                "Enter coordinates  →  Open Google Earth Pro\n"
                "Draw polygon  →  Save as KML\n"
                "Load satellite image to begin"
            ),
            font=("Segoe UI", 12),
            fg=T.TEXT_MUTED,
            bg="#090d12",
            justify=tk.CENTER
        )
        self._empty.place(
            relx=0.5, rely=0.5, anchor="center"
        )

        # Status bar
        sb2 = tk.Frame(
            wrap, bg=T.BG_CARD, height=26
        )
        sb2.pack(fill=tk.X)
        sb2.pack_propagate(False)

        self._status_l = tk.Label(
            sb2, text="Ready",
            font=("Consolas", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        )
        self._status_l.pack(
            side=tk.LEFT, padx=10
        )

        self._status_r = tk.Label(
            sb2, text="",
            font=("Consolas", 8),
            fg=T.TEXT_SEC, bg=T.BG_CARD
        )
        self._status_r.pack(
            side=tk.RIGHT, padx=10
        )

        # Console
        con_frame = tk.Frame(
            wrap, bg=T.BG_CARD, height=130
        )
        con_frame.pack(fill=tk.X)
        con_frame.pack_propagate(False)

        tk.Frame(
            con_frame, bg=T.BORDER, height=1
        ).pack(fill=tk.X)

        con_hdr = tk.Frame(
            con_frame, bg=T.BG_CARD
        )
        con_hdr.pack(fill=tk.X, padx=10,
                     pady=(4, 0))

        tk.Label(
            con_hdr, text="CONSOLE",
            font=("Segoe UI", 8, "bold"),
            fg=T.TEXT_MUTED, bg=T.BG_CARD
        ).pack(side=tk.LEFT)

        tk.Button(
            con_hdr, text="Clear",
            font=("Segoe UI", 8),
            fg=T.TEXT_MUTED, bg=T.BG_CARD,
            relief=tk.FLAT, bd=0,
            cursor="hand2",
            command=self._clear_log
        ).pack(side=tk.RIGHT)

        self._log_box = tk.Text(
            con_frame,
            bg=T.BG_DARK, fg="#00d47e",
            font=("Consolas", 8),
            state=tk.DISABLED,
            relief=tk.FLAT, bd=0,
            wrap=tk.WORD
        )
        self._log_box.pack(
            fill=tk.BOTH, expand=True,
            padx=10, pady=(2, 8)
        )

        for tag, col in [
            ("info",    T.ACCENT),
            ("success", T.SUCCESS),
            ("warning", T.WARNING),
            ("error",   T.DANGER),
        ]:
            self._log_box.tag_config(
                tag, foreground=col
            )

    # ═════════════════════════════════════════════════
    # HANDLERS
    # ═════════════════════════════════════════════════

    # ── Plot / Clear points ───────────────────────────

    def _get_mission_coordinates(self) -> tuple[float, float, float, float]:
        try:
            s_lat = float(self._e_slat.get().strip())
            s_lon = float(self._e_slon.get().strip())
            e_lat = float(self._e_elat.get().strip())
            e_lon = float(self._e_elon.get().strip())
        except ValueError as exc:
            raise ValueError(
                "Enter valid start and end coordinates first."
            ) from exc

        if not (-90 <= s_lat <= 90 and -90 <= e_lat <= 90):
            raise ValueError("Latitude must be -90 to 90.")
        if not (-180 <= s_lon <= 180 and -180 <= e_lon <= 180):
            raise ValueError("Longitude must be -180 to 180.")

        return s_lat, s_lon, e_lat, e_lon

    @staticmethod
    def _display_repo_path(path: Path) -> str:
        try:
            return str(path.relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _show_google_earth_launch_failure(
        self,
        center_lat: float,
        center_lon: float,
        launch_result: GoogleEarthLaunchResult,
    ) -> None:
        kml_path = self._display_repo_path(launch_result.kml_path)
        attempted = "\n".join(
            f"  {command}" for command in launch_result.attempted_commands
        ) or "  (none)"
        searched = "\n".join(
            f"  {command}" for command in launch_result.searched_executables
        ) or "  (none)"

        messagebox.showinfo(
            "Google Earth Pro",
            "Could not launch automatically.\n\n"
            "Target latitude/longitude:\n"
            f"  {center_lat:.5f}, {center_lon:.5f}\n\n"
            "Generated KML path:\n"
            f"  {kml_path}\n\n"
            "Attempted launch commands:\n"
            f"{attempted}\n\n"
            "Executable search order:\n"
            f"{searched}\n\n"
            "Reason:\n"
            f"  {launch_result.error_reason}\n\n"
            "Open Google Earth Pro manually, draw the polygon, "
            "and save the KML to data/kml/."
        )

    def _on_plot_points(self):
        try:
            s_lat, s_lon, e_lat, e_lon = self._get_mission_coordinates()
        except ValueError as exc:
            self._log(str(exc), "error")
            return

        self._start = (s_lat, s_lon)
        self._goal  = (e_lat, e_lon)

        dist = self._conv.haversine_m(
            s_lat, s_lon, e_lat, e_lon
        )

        # Update step-03 labels
        self._badge_start.set("SET", "ok")
        self._badge_goal.set("SET", "ok")
        self._lbl_start.config(
            text=f"{s_lat:.6f},  {s_lon:.6f}",
            fg=T.SUCCESS
        )
        self._lbl_goal.config(
            text=f"{e_lat:.6f},  {e_lon:.6f}",
            fg=T.DANGER
        )

        self._pts_info.config(
            text=(
                f"✓  Start ({s_lat:.5f}, {s_lon:.5f})\n"
                f"✓  End   ({e_lat:.5f}, {e_lon:.5f})\n"
                f"   Distance ≈ {dist:.0f} m"
            ),
            fg=T.SUCCESS
        )

        self._log(
            f"Start ({s_lat:.5f}, {s_lon:.5f})  "
            f"End ({e_lat:.5f}, {e_lon:.5f})  "
            f"dist={dist:.0f}m",
            "success"
        )
        self._sys_badge.set("POINTS SET", "ok")

        # Auto-set bounds if no polygon yet
        if (self._img_bounds is None
                and self._satellite is not None):
            pad = 0.005
            self._img_bounds = {
                "north": max(s_lat, e_lat) + pad,
                "south": min(s_lat, e_lat) - pad,
                "east":  max(s_lon, e_lon) + pad,
                "west":  min(s_lon, e_lon) - pad,
            }

        if self._satellite is not None:
            self._render()
        else:
            self._log(
                "Load satellite image to see "
                "points on map",
                "info"
            )

    def _on_clear_points(self):
        self._start  = None
        self._goal   = None
        self._pts_info.config(text="")
        self._badge_start.set("NOT SET", "idle")
        self._badge_goal.set("NOT SET",  "idle")
        self._lbl_start.config(
            text="—", fg=T.TEXT_MUTED
        )
        self._lbl_goal.config(
            text="—", fg=T.TEXT_MUTED
        )
        self._e_slat.delete(0, tk.END)
        self._e_slat.insert(0, "33.7470")
        self._e_slon.delete(0, tk.END)
        self._e_slon.insert(0, "73.1370")
        self._e_elat.delete(0, tk.END)
        self._e_elat.insert(0, "33.7550")
        self._e_elon.delete(0, tk.END)
        self._e_elon.insert(0, "73.1460")
        self._log("Points cleared", "info")
        if self._satellite is not None:
            self._render()

    # ── Google Earth Pro ──────────────────────────────

    def _on_open_ge(self):
        try:
            s_lat, s_lon, e_lat, e_lon = self._get_mission_coordinates()
        except ValueError as exc:
            self._log(str(exc), "warning")
            messagebox.showwarning(
                "Coordinates Required",
                str(exc)
            )
            return

        c_lat = (s_lat + e_lat) / 2
        c_lon = (s_lon + e_lon) / 2
        diff  = max(
            abs(e_lat - s_lat),
            abs(e_lon - s_lon)
        )
        alt   = max(800, int(diff * 111320 * 2.5))
        logger.info(
            "Opening Google Earth at mission center lat=%.5f lon=%.5f",
            c_lat,
            c_lon,
        )
        launch_result = open_google_earth_at_location(
            c_lat,
            c_lon,
            start=(s_lat, s_lon),
            goal=(e_lat, e_lon),
            output_path=self.KML_DIR / DEFAULT_FOCUS_KML_NAME,
            lookat_range_m=alt,
        )

        if launch_result.opened:
            self._log(
                f"Google Earth Pro opened at "
                f"({c_lat:.4f}, {c_lon:.4f})",
                "success"
            )
            logger.info(
                "Google Earth launch succeeded via %s",
                launch_result.executable_path or launch_result.launch_method,
            )
        else:
            self._log(
                f"Google Earth launch failed: {launch_result.error_reason}",
                "error",
            )
            self._show_google_earth_launch_failure(
                c_lat,
                c_lon,
                launch_result,
            )

        self._log(
            "Draw polygon → Right-click → "
            "Save Place As → KML → save to data/kml/",
            "info"
        )
        self._sys_badge.set("WAITING KML", "working")

    # ── KML watcher ───────────────────────────────────

    def _start_kml_watcher(self):
        kml_dir = self.KML_DIR
        kml_dir.mkdir(parents=True, exist_ok=True)

        def watch():
            while True:
                if self._watch_var.get():
                    files = [
                        f for f in kml_dir.glob("*.kml")
                        if f.name not in self.GENERATED_KML_NAMES
                    ]
                    if files:
                        newest = max(
                            files,
                            key=lambda f: f.stat().st_mtime
                        )
                        mt = newest.stat().st_mtime
                        if mt > self._last_kml_t:
                            self._last_kml_t = mt
                            self.root.after(
                                0, self._load_kml,
                                str(newest)
                            )
                time.sleep(2)

        threading.Thread(
            target=watch, daemon=True
        ).start()

    def _on_browse_kml(self):
        p = filedialog.askopenfilename(
            title="Select KML File",
            initialdir=str(self.KML_DIR),
            filetypes=[
                ("KML", "*.kml"),
                ("All", "*.*")
            ]
        )
        if p:
            self._load_kml(p)

    def _load_kml(self, path: str):
        self._log(
            f"Reading: {Path(path).name}", "info"
        )
        poly = self._kml.read(path)
        if poly is None:
            self._log(
                "Could not parse polygon from KML",
                "error"
            )
            self._kml_info.config(
                text="❌  Invalid or empty KML",
                fg=T.DANGER
            )
            return

        self._polygon = poly

        self.KML_DIR.mkdir(
            parents=True, exist_ok=True
        )
        with open(self.KML_DIR / "polygon.json", "w") as f:
            json.dump(asdict(poly), f, indent=2)

        self._kml_info.config(
            text=(
                f"✓  {poly.name}\n"
                f"   {len(poly.coordinates)} vertices\n"
                f"   N {poly.north:.5f}  "
                f"S {poly.south:.5f}\n"
                f"   E {poly.east:.5f}  "
                f"W {poly.west:.5f}"
            ),
            fg=T.SUCCESS
        )
        self._log(
            f"Polygon: {len(poly.coordinates)} pts  "
            f"center ({poly.center_lat:.4f},"
            f"{poly.center_lon:.4f})",
            "success"
        )
        self._sys_badge.set("KML READY", "ok")

        if self._satellite is not None:
            self._render()

    # ── Satellite image ───────────────────────────────

    def _on_load_sat(self):
        p = filedialog.askopenfilename(
            title="Select Satellite Image",
            initialdir="data/satellite_raw",
            filetypes=[
                ("Images",
                 "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All", "*.*"),
            ]
        )
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
            self._log("Cannot read image", "error")
            return

        self._satellite = img
        self._empty.place_forget()
        self._img_info.config(
            text=(
                f"✓  {Path(p).name}  "
                f"({img.shape[1]}×{img.shape[0]})"
            ),
            fg=T.SUCCESS
        )
        self._log(
            f"Image {img.shape[1]}×{img.shape[0]}",
            "success"
        )
        self._render()

    # ── Map click / hover ─────────────────────────────

    def _on_set_start(self):
        if self._satellite is None:
            self._log(
                "Load satellite image first", "warning"
            )
            return
        self._click_mode = "start"
        self._mode_badge.set(
            "CLICK START", "working"
        )
        self._log(
            "Click on map to place START", "info"
        )

    def _on_set_goal(self):
        if self._satellite is None:
            self._log(
                "Load satellite image first", "warning"
            )
            return
        self._click_mode = "goal"
        self._mode_badge.set(
            "CLICK GOAL", "working"
        )
        self._log(
            "Click on map to place GOAL", "info"
        )

    def _on_map_click(self, ev):
        if self._click_mode is None:
            return
        if self._img_bounds is None:
            self._log(
                "Load image / import KML first",
                "warning"
            )
            return

        w  = self._canvas.winfo_width()
        h  = self._canvas.winfo_height()
        b  = self._img_bounds

        lat, lon = self._conv.pixel_to_latlon(
            ev.x, ev.y,
            b["north"], b["south"],
            b["west"],  b["east"],
            w, h
        )
        utm_e, utm_n, _ = self._conv.latlon_to_utm(
            lat, lon
        )

        if self._click_mode == "start":
            self._start = (lat, lon)
            self._badge_start.set("SET", "ok")
            self._lbl_start.config(
                text=f"{lat:.6f},  {lon:.6f}",
                fg=T.SUCCESS
            )
            self._e_slat.delete(0, tk.END)
            self._e_slat.insert(0, f"{lat:.6f}")
            self._e_slon.delete(0, tk.END)
            self._e_slon.insert(0, f"{lon:.6f}")
            self._log(
                f"Start ({lat:.5f}, {lon:.5f})  "
                f"UTM {utm_e:.0f}E {utm_n:.0f}N",
                "success"
            )

        elif self._click_mode == "goal":
            self._goal = (lat, lon)
            self._badge_goal.set("SET", "ok")
            self._lbl_goal.config(
                text=f"{lat:.6f},  {lon:.6f}",
                fg=T.DANGER
            )
            self._e_elat.delete(0, tk.END)
            self._e_elat.insert(0, f"{lat:.6f}")
            self._e_elon.delete(0, tk.END)
            self._e_elon.insert(0, f"{lon:.6f}")
            self._log(
                f"Goal  ({lat:.5f}, {lon:.5f})  "
                f"UTM {utm_e:.0f}E {utm_n:.0f}N",
                "success"
            )

        # Update pts_info if both set
        if self._start and self._goal:
            d = self._conv.haversine_m(
                *self._start, *self._goal
            )
            self._pts_info.config(
                text=(
                    f"✓  Start ({self._start[0]:.5f},"
                    f" {self._start[1]:.5f})\n"
                    f"✓  End   ({self._goal[0]:.5f},"
                    f" {self._goal[1]:.5f})\n"
                    f"   Distance ≈ {d:.0f} m"
                ),
                fg=T.SUCCESS
            )

        self._click_mode = None
        self._mode_badge.set("VIEW", "idle")
        self._render()

    def _on_hover(self, ev):
        if self._img_bounds is None:
            return
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        b = self._img_bounds
        lat, lon = self._conv.pixel_to_latlon(
            ev.x, ev.y,
            b["north"], b["south"],
            b["west"],  b["east"],
            w, h
        )
        self._coord_lbl.config(
            text=f"Lat {lat:.6f}   Lon {lon:.6f}"
        )

    # ── Path planning ─────────────────────────────────

    def _on_run_plan(self):
        if self._polygon is None:
            self._log("Import KML first", "warning")
            return
        if self._start is None:
            self._log("Set start point", "warning")
            return
        if self._goal is None:
            self._log("Set goal point", "warning")
            return

        try:
            sp = float(self._spacing_var.get())
        except ValueError:
            sp = 15.0

        self._plan = PathPlanner(spacing=sp)
        self._log(
            f"A* running  grid={sp}m  "
            f"start={self._start}  "
            f"goal={self._goal}",
            "info"
        )
        self._sys_badge.set("PLANNING", "working")
        self._path_info.config(
            text="Computing path...",
            fg=T.WARNING
        )

        def run():
            wps = self._plan.plan(
                self._start[0], self._start[1],
                self._goal[0],  self._goal[1],
                self._polygon.coordinates
            )
            self.root.after(
                0, self._path_done, wps
            )

        threading.Thread(
            target=run, daemon=True
        ).start()

    def _path_done(self, wps):
        if wps is None:
            self._log(
                "No path found — try larger grid "
                "spacing or different points",
                "error"
            )
            self._path_info.config(
                text="❌  No path found",
                fg=T.DANGER
            )
            self._sys_badge.set("FAILED", "error")
            return

        self._waypoints = wps
        dist = sum(w.dist_m for w in wps)
        self._path_info.config(
            text=(
                f"✓  {len(wps)} waypoints  "
                f"|  {dist:.0f} m total"
            ),
            fg=T.SUCCESS
        )
        self._log(
            f"Path: {len(wps)} waypoints  "
            f"{dist:.0f}m",
            "success"
        )
        self._sys_badge.set("PATH READY", "ok")
        self._status_r.config(
            text=(
                f"{len(wps)} waypoints  |  "
                f"{dist:.0f} m"
            )
        )
        self._render()

    # ── Export ────────────────────────────────────────

    def _on_export_csv(self):
        if not self._waypoints:
            self._log("Plan path first", "warning")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="waypoints.csv",
            initialdir="data/waypoints",
            filetypes=[("CSV", "*.csv")]
        )
        if not p:
            return
        Path(p).parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "index", "latitude", "longitude",
                "easting_m", "northing_m", "dist_m"
            ])
            for wp in self._waypoints:
                w.writerow([
                    wp.index,
                    round(wp.lat, 8),
                    round(wp.lon, 8),
                    round(wp.easting, 2),
                    round(wp.northing, 2),
                    round(wp.dist_m, 2),
                ])
        self._export_info.config(
            text=f"✓  CSV saved: {Path(p).name}"
        )
        self._log(
            f"CSV exported: {Path(p).name}", "success"
        )

    def _on_export_json(self):
        if not self._waypoints:
            self._log("Plan path first", "warning")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="waypoints.json",
            initialdir="data/waypoints",
            filetypes=[("JSON", "*.json")]
        )
        if not p:
            return
        Path(p).parent.mkdir(
            parents=True, exist_ok=True
        )
        data = {
            "total_waypoints": len(self._waypoints),
            "total_distance_m": sum(
                w.dist_m for w in self._waypoints
            ),
            "waypoints": [
                asdict(w) for w in self._waypoints
            ],
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        self._export_info.config(
            text=f"✓  JSON saved: {Path(p).name}"
        )
        self._log(
            f"JSON exported: {Path(p).name}",
            "success"
        )

    # ═════════════════════════════════════════════════
    # MAP RENDERING
    # ═════════════════════════════════════════════════

    def _render(self):
        if self._satellite is None:
            return

        img = self._satellite.copy()
        h, w = img.shape[:2]

        # Set bounds
        if self._polygon:
            b = {
                "north": self._polygon.north,
                "south": self._polygon.south,
                "east":  self._polygon.east,
                "west":  self._polygon.west,
            }
        elif self._start and self._goal:
            pad = 0.005
            b = {
                "north": max(self._start[0],
                             self._goal[0]) + pad,
                "south": min(self._start[0],
                             self._goal[0]) - pad,
                "east":  max(self._start[1],
                             self._goal[1]) + pad,
                "west":  min(self._start[1],
                             self._goal[1]) - pad,
            }
        else:
            b = {
                "north": 33.760, "south": 33.735,
                "east":  73.150, "west":  73.120,
            }
        self._img_bounds = b

        def px(lat, lon):
            return self._conv.latlon_to_pixel(
                lat, lon,
                b["north"], b["south"],
                b["west"],  b["east"],
                w, h
            )

        # ── Polygon ───────────────────────────────────
        if (self._polygon
                and self._polygon.coordinates):
            pts = np.array(
                [list(px(la, lo))
                 for la, lo in
                 self._polygon.coordinates],
                np.int32
            )
            ov = img.copy()
            cv2.fillPoly(
                ov, [pts], (88, 166, 255)
            )
            img = cv2.addWeighted(
                img, 0.82, ov, 0.18, 0
            )
            cv2.polylines(
                img, [pts], True,
                (88, 166, 255), 2, cv2.LINE_AA
            )

        # ── Waypoint path ─────────────────────────────
        if (self._waypoints
                and len(self._waypoints) > 1):
            wpts = [
                list(px(wp.lat, wp.lon))
                for wp in self._waypoints
            ]
            for i in range(1, len(wpts)):
                cv2.line(
                    img,
                    tuple(wpts[i-1]),
                    tuple(wpts[i]),
                    (188, 140, 255), 2,
                    cv2.LINE_AA
                )
            for i, pt in enumerate(wpts):
                if 0 < i < len(wpts)-1:
                    cv2.circle(
                        img, tuple(pt), 4,
                        (188, 140, 255), -1
                    )

        # ── Dashed line start → goal ──────────────────
        if self._start and self._goal:
            p1 = px(*self._start)
            p2 = px(*self._goal)
            for a, b2 in self._dashed(p1, p2):
                cv2.line(
                    img, a, b2,
                    (200, 200, 200), 1,
                    cv2.LINE_AA
                )

        # ── START marker ──────────────────────────────
        if self._start:
            p = list(px(*self._start))
            cv2.circle(
                img, tuple(p), 18,
                (63, 185, 80), 1, cv2.LINE_AA
            )
            cv2.circle(
                img, tuple(p), 13,
                (63, 185, 80), -1
            )
            cv2.circle(
                img, tuple(p), 15,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            self._draw_label(
                img, "START",
                p[0]+20, p[1],
                (63, 185, 80)
            )

        # ── END / GOAL marker ─────────────────────────
        if self._goal:
            p = list(px(*self._goal))
            cv2.circle(
                img, tuple(p), 18,
                (248, 81, 73), 1, cv2.LINE_AA
            )
            cv2.circle(
                img, tuple(p), 13,
                (248, 81, 73), -1
            )
            cv2.circle(
                img, tuple(p), 15,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            self._draw_label(
                img, "END",
                p[0]+20, p[1],
                (248, 81, 73)
            )

        self._show(img)

    def _draw_label(
        self, img, text, x, y, color
    ):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 2
        (tw, th), _ = cv2.getTextSize(
            text, font, scale, thick
        )
        cv2.rectangle(
            img,
            (x-4, y-th-6),
            (x+tw+4, y+4),
            (18, 18, 18), -1
        )
        cv2.putText(
            img, text, (x, y),
            font, scale, color, thick,
            cv2.LINE_AA
        )

    def _dashed(self, p1, p2,
                dash=14, gap=6):
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        L  = math.sqrt(dx*dx+dy*dy)
        if L == 0:
            return []
        ux, uy = dx/L, dy/L
        segs   = []
        d      = 0.0
        while d < L:
            de  = min(d+dash, L)
            ax  = int(p1[0]+ux*d)
            ay  = int(p1[1]+uy*d)
            bx  = int(p1[0]+ux*de)
            by  = int(p1[1]+uy*de)
            segs.append(((ax,ay),(bx,by)))
            d  += dash+gap
        return segs

    def _show(self, img: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10: cw = 900
        if ch < 10: ch = 600
        disp = cv2.resize(
            img, (cw, ch),
            interpolation=cv2.INTER_AREA
        )
        disp  = cv2.cvtColor(
            disp, cv2.COLOR_BGR2RGB
        )
        photo = ImageTk.PhotoImage(
            Image.fromarray(disp)
        )
        self._canvas.config(image=photo, text="")
        self._canvas.image = photo

    # ═════════════════════════════════════════════════
    # CONSOLE
    # ═════════════════════════════════════════════════

    def _log(self, msg: str, level: str = ""):
        ts = time.strftime("%H:%M:%S")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(
            tk.END,
            f"[{ts}]  {msg}\n",
            level if level else ()
        )
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)
        self._status_l.config(
            text=msg[:70] + ("…" if len(msg)>70 else "")
        )

    def _clear_log(self):
        self._log_box.config(state=tk.NORMAL)
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state=tk.DISABLED)

    # ═════════════════════════════════════════════════
    # DATABASE HANDLERS  (FAISS + DINOv2)
    # ═════════════════════════════════════════════════

    def _get_faiss_db(self) -> Optional[object]:
        if not _FAISS_AVAILABLE:
            messagebox.showerror(
                "Import Error",
                "FAISSDatabase could not be loaded.\n"
                "Install: pip install faiss-cpu torch torchvision"
            )
            return None
        if self._faiss_db is None:
            self._faiss_db = FAISSDatabase()
        return self._faiss_db

    def _on_db_select_image(self):
        p = filedialog.askopenfilename(
            title="Select Satellite Image",
            initialdir="data/satellite/raw",
            filetypes=[
                ("Images",
                 "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All", "*.*"),
            ]
        )
        if not p:
            return
        self._db_sat_path = p
        self._db_img_lbl.config(
            text=f"✓  {Path(p).name}", fg=T.SUCCESS
        )
        self._log(f"Satellite image selected: {Path(p).name}", "info")

    def _on_build_faiss_db(self):
        if not self._db_sat_path:
            messagebox.showwarning(
                "No Image", "Select a satellite image first."
            )
            return
        try:
            lat = float(self._db_lat.get().strip())
            lon = float(self._db_lon.get().strip())
            alt = float(self._db_alt.get().strip())
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Lat / Lon / Alt must be numeric."
            )
            return

        db = self._get_faiss_db()
        if db is None:
            return

        self._log(
            f"Building FAISS DB: {Path(self._db_sat_path).name} "
            f"({lat:.4f}, {lon:.4f})",
            "info"
        )
        self._sys_badge.set("BUILDING DB", "working")
        self._db_progress.config(
            text="Building…  0 / ?", fg=T.WARNING
        )

        def _progress(cur, total):
            pct = int(cur / total * 100) if total else 0
            self.root.after(
                0, self._db_progress.config,
                {"text": f"Building…  {cur}/{total}  ({pct}%)"}
            )

        def _run():
            count = db.build(
                satellite_image_path=self._db_sat_path,
                lat=lat, lon=lon, alt=alt,
                progress_callback=_progress,
            )
            self.root.after(0, self._faiss_build_done, count, db)

        threading.Thread(target=_run, daemon=True).start()

    def _faiss_build_done(self, count: int, db) -> None:
        stats = db.get_stats()
        self._db_progress.config(
            text=f"✓  {count} patches  |  {stats['index_size_mb']} MB",
            fg=T.SUCCESS
        )
        self._db_status.config(
            text=(
                f"✓  {stats['total_descriptors']} patches"
                f"  |  DINOv2 {stats['descriptor_dim']}-dim"
                f"  |  {stats['unique_images']} image(s)"
            ),
            fg=T.SUCCESS
        )
        self._log(
            f"FAISS DB built: {count} patches "
            f"({stats['descriptor_dim']}-dim, "
            f"{stats['index_size_mb']} MB)",
            "success"
        )
        self._sys_badge.set("DB READY", "ok")

    def _on_load_faiss_db(self):
        db = self._get_faiss_db()
        if db is None:
            return
        ok = db.load()
        if ok:
            stats = db.get_stats()
            self._db_status.config(
                text=(
                    f"✓  {stats['total_descriptors']} patches"
                    f"  |  DINOv2 {stats['descriptor_dim']}-dim"
                    f"  |  {stats['unique_images']} image(s)"
                ),
                fg=T.SUCCESS
            )
            self._log(
                f"FAISS DB loaded: {stats['total_descriptors']} descriptors",
                "success"
            )
            self._sys_badge.set("DB LOADED", "ok")
        else:
            self._db_status.config(
                text="Load failed — build the database first",
                fg=T.WARNING
            )
            self._log("FAISS DB load failed", "warning")

    def _on_clear_faiss_db(self):
        if not messagebox.askyesno(
            "Clear Database",
            "Delete the FAISS index and metadata from disk?\n"
            "This cannot be undone."
        ):
            return
        db = self._get_faiss_db()
        if db is None:
            return
        for f in (db.index_path, db.meta_path):
            if f.exists():
                f.unlink()
        db.metadata = []
        db._init_index()
        self._faiss_db = None
        self._db_status.config(
            text="Status: cleared", fg=T.WARNING
        )
        self._db_progress.config(text="")
        self._log("FAISS database cleared", "warning")
        self._sys_badge.set("DB CLEARED", "working")

    # ═════════════════════════════════════════════════
    # MATCHING HANDLERS
    # ═════════════════════════════════════════════════

    def _on_match_select_uav(self):
        p = filedialog.askopenfilename(
            title="Select UAV Image",
            initialdir="data/patches/uav",
            filetypes=[
                ("Images",
                 "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All", "*.*"),
            ]
        )
        if not p:
            return
        self._uav_match_path = p
        self._match_img_lbl.config(
            text=f"✓  {Path(p).name}", fg=T.SUCCESS
        )
        self._log(f"UAV image selected: {Path(p).name}", "info")

    def _on_run_matching(self):
        if not self._uav_match_path:
            messagebox.showwarning(
                "No Image", "Select a UAV image first."
            )
            return
        db = self._get_faiss_db()
        if db is None:
            return
        if db.index is None or db.index.ntotal == 0:
            messagebox.showwarning(
                "Empty Database",
                "Build or load the database first."
            )
            return

        import cv2 as _cv2
        uav_img = _cv2.imread(self._uav_match_path)
        if uav_img is None:
            messagebox.showerror(
                "Read Error", "Cannot read UAV image."
            )
            return

        self._match_results.config(
            text="Running…", fg=T.WARNING
        )
        self._sys_badge.set("MATCHING", "working")
        self._log("Running DINOv2 + ORB matching…", "info")

        def _run():
            results = db.query(uav_img, top_k=5, min_confidence=0.0)
            self.root.after(0, self._match_done, results)

        threading.Thread(target=_run, daemon=True).start()

    def _match_done(self, results: list) -> None:
        n_total = len(results)
        n_verified = sum(1 for r in results if r.get("orb_verified"))

        if not results:
            self._match_results.config(
                text="No matches found", fg=T.DANGER
            )
            self._sys_badge.set("NO MATCH", "error")
            self._log("Matching: no results", "error")
            return

        best = results[0]
        lines = [
            f"Stage 1 (DINOv2): {n_total} candidates found",
            f"Stage 2 (ORB):    {n_verified} verified",
            f"Best Match: Lat {best['lat']:.6f},"
            f"  Lon {best['lon']:.6f}",
            f"Confidence: {best['confidence']*100:.1f}%",
        ]
        self._match_results.config(
            text="\n".join(lines), fg=T.SUCCESS
        )
        self._sys_badge.set("MATCHED", "ok")
        self._log(
            f"Match: lat={best['lat']:.5f} "
            f"lon={best['lon']:.5f} "
            f"conf={best['confidence']*100:.1f}%",
            "success"
        )

    # ═════════════════════════════════════════════════
    # RUN
    # ═════════════════════════════════════════════════

    def run(self):
        self._log(
            "VNS ready  —  enter coordinates "
            "and open Google Earth Pro",
            "success"
        )
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    app = VNSApp()
    app.run()

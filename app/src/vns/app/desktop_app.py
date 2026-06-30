# app/src/vns/app/desktop_app.py
"""
VNS - GNSS-Free Navigation System
Desktop Application - Amber Military Theme
Final Year Project 2026
Muhammad Ahsan & Aittrah Sardar
IIT Quaid-i-Azam University Islamabad
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading
import subprocess
import os
import sys
import json
import csv
import math
import heapq
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

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

# =============================================================================
# DATA CLASSES
# =============================================================================

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


# =============================================================================
# KML READER
# =============================================================================

class KMLReader:
    NS = "{http://www.opengis.net/kml/2.2}"

    def read(self, path: str) -> Optional[PolygonData]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            coords_str = None
            for ns in [self.NS, ""]:
                for p in [
                    f".//{ns}Polygon/{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates",
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
                name=name, coordinates=points,
                center_lat=sum(lats)/len(lats),
                center_lon=sum(lons)/len(lons),
                north=max(lats), south=min(lats),
                east=max(lons),  west=min(lons),
            )
        except Exception as e:
            print(f"KML error: {e}")
            return None


# =============================================================================
# COORDINATE CONVERTER
# =============================================================================

class CoordConverter:

    def latlon_to_utm(self, lat: float, lon: float):
        zone  = int((lon + 180) / 6) + 1
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
        M  = a * (
            (1 - e2/4 - 3*e2**2/64) * lat_r
            - (3*e2/8 + 3*e2**2/32) * math.sin(2*lat_r)
            + (15*e2**2/256) * math.sin(4*lat_r)
        )
        E  = (k0 * N * (A_ + (1-T+C)*A_**3/6 + (5-18*T+T**2)*A_**5/120) + 500000)
        Nv = k0 * (M + N * math.tan(lat_r) * (
            A_**2/2 + (5-T+9*C+4*C**2)*A_**4/24 + (61-58*T+T**2)*A_**6/720
        ))
        if lat < 0:
            Nv += 10000000
        return E, Nv, zone

    def latlon_to_pixel(self, lat, lon, n, s, w, e, iw, ih):
        x = int((lon - w) / (e - w) * iw)
        y = int((n - lat) / (n - s) * ih)
        return (max(0, min(x, iw-1)), max(0, min(y, ih-1)))

    def pixel_to_latlon(self, px, py, n, s, w, e, iw, ih):
        lon = w + (px / iw) * (e - w)
        lat = n - (py / ih) * (n - s)
        return lat, lon

    def haversine_m(self, lat1, lon1, lat2, lon2) -> float:
        R  = 6371000
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# =============================================================================
# A* PATH PLANNER
# =============================================================================

class PathPlanner:

    def __init__(self, spacing: float = 15.0):
        self.spacing = spacing
        self.conv    = CoordConverter()

    def plan(self, s_lat, s_lon, g_lat, g_lon, poly: list) -> Optional[list]:
        sE, sN, zone = self.conv.latlon_to_utm(s_lat, s_lon)
        gE, gN, _    = self.conv.latlon_to_utm(g_lat, g_lon)
        pUTM = [self.conv.latlon_to_utm(la, lo)[:2] for la, lo in poly]
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
            dlon = (e - sE) / (111320 * math.cos(math.radians(s_lat)))
            lat  = s_lat + dlat
            lon  = s_lon + dlon
            d    = 0.0
            if pE is not None:
                d = math.sqrt((e-pE)**2 + (n-pN)**2)
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
        return min(nodes, key=lambda nd: math.sqrt((nd[0]-e)**2+(nd[1]-n)**2))

    def _astar(self, grid, start, goal):
        def h(a, b):
            return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
        def neighbors(pos):
            e, n = pos
            s = self.spacing
            cands = [(e+s,n),(e-s,n),(e,n+s),(e,n-s),
                     (e+s,n+s),(e-s,n+s),(e+s,n-s),(e-s,n-s)]
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
                    heapq.heappush(heap, ANode(f, ng, nb, cur.pos))
                    came[nb] = cur.pos
        return None


# =============================================================================
# THEME  -- amber military palette
# =============================================================================

class T:
    BG_DARK    = "#0a0b08"
    BG_PANEL   = "#0f110c"
    BG_CARD    = "#111408"
    BG_INPUT   = "#07080a"
    BG_HOVER   = "#1a1c14"

    AMBER      = "#f0a030"
    AMBER_DIM  = "#5a4a20"
    AMBER_DARK = "#1a1000"
    GREEN      = "#44cc44"
    RED        = "#ff4400"
    BLUE       = "#44aaff"
    PURPLE     = "#9966ff"

    WARNING    = "#f0a030"

    BORDER     = "#2a2200"
    TEXT_PRI   = "#d4a843"
    TEXT_SEC   = "#5a4a20"
    TEXT_MUTED = "#3a3020"

    FONT_LBL   = ("Courier New", 9)


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ProfButton(tk.Button):
    STYLES = {
        "primary": ("#1a1000",  "#f0a030", "#f0a030"),
        "success": ("#0a1f0a",  "#44cc44", "#44cc44"),
        "danger":  ("#1f0a08",  "#ff4400", "#ff4400"),
        "ghost":   ("#111408",  "#5a4a20", "#f0a030"),
        "purple":  ("#12091f",  "#9966ff", "#9966ff"),
    }

    def __init__(self, parent, text, cmd, style="primary", icon="", **kw):
        bg, fg, _ = self.STYLES.get(style, self.STYLES["primary"])
        label = f"{icon}  {text}" if icon else text
        super().__init__(
            parent, text=label, command=cmd,
            bg=bg, fg=fg,
            font=("Courier New", 9, "bold"),
            relief=tk.FLAT, bd=0,
            padx=12, pady=7,
            cursor="hand2",
            activebackground="#1a1000",
            activeforeground="#f0a030",
            highlightthickness=1,
            highlightbackground="#2a2200",
            highlightcolor="#f0a030",
            **kw
        )
        self._fg = fg
        self.bind("<Enter>", lambda e: self.config(highlightbackground="#f0a030", fg="#f0a030"))
        self.bind("<Leave>", lambda e: self.config(highlightbackground="#2a2200", fg=self._fg))


class StatusBadge(tk.Label):
    COLORS = {
        "idle":    ("#5a4a20", "#111408"),
        "ok":      ("#44cc44", "#0a1a0a"),
        "working": ("#f0a030", "#1a1000"),
        "error":   ("#ff4400", "#1a0800"),
        "info":    ("#44aaff", "#08101a"),
    }

    def __init__(self, parent, text="IDLE"):
        fg, bg = self.COLORS["idle"]
        super().__init__(
            parent, text=f"  {text}  ",
            font=("Courier New", 8, "bold"),
            fg=fg, bg=bg,
            relief=tk.FLAT, padx=4, pady=2
        )

    def set(self, text, color="idle"):
        fg, bg = self.COLORS.get(color, self.COLORS["idle"])
        self.config(text=f"  {text}  ", fg=fg, bg=bg)


def make_entry(parent, default="", w=16, hi_color=None):
    e = tk.Entry(
        parent,
        bg="#07080a", fg="#f0a030",
        insertbackground="#f0a030",
        font=("Courier New", 10),
        relief=tk.FLAT, bd=0, width=w,
        highlightthickness=1,
        highlightcolor=hi_color or "#f0a030",
        highlightbackground="#2a2200",
    )
    e.insert(0, default)
    return e


def section_label(parent, text, color=None):
    color = color or "#f0a030"
    f = tk.Frame(parent, bg="#0f110c")
    f.pack(fill=tk.X, padx=10, pady=(14, 3))
    tk.Label(f, text=f"[{text[:2]}]",
             font=("Courier New", 8, "bold"),
             fg=color, bg="#0f110c").pack(side=tk.LEFT)
    tk.Label(f, text=f"  {text[3:]}",
             font=("Courier New", 9, "bold"),
             fg="#d4a843", bg="#0f110c").pack(side=tk.LEFT)
    tk.Frame(parent, bg="#2a2200", height=1).pack(fill=tk.X, padx=10)


# =============================================================================
# SPLASH SCREEN
# =============================================================================

class _SplashScreen:
    STEPS = [
        (500,  "drone"),
        (1000, "uav_label"),
        (1500, "sat_box"),
        (2000, "match_box"),
        (2500, "dr_box"),
        (3000, "pos_box"),
        (3500, "status_bar"),
        (4000, "close"),
    ]

    def __init__(self, root: tk.Tk):
        self._root   = root
        self._closed = False
        self._win    = tk.Toplevel(root)
        self._win.overrideredirect(True)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        self._win.geometry(f"{sw}x{sh}+0+0")
        self._win.configure(bg="#0a0b08")
        self._win.lift()
        self._win.focus_force()
        self._sh = sh

        self._build()
        self._win.bind("<Button-1>", lambda e: self._close())
        for delay, step in self.STEPS:
            self._win.after(delay, self._animate, step)

    def _build(self):
        badge_f = tk.Frame(self._win, bg="#0a0b08")
        badge_f.place(relx=1.0, rely=0.0, anchor="ne", x=-20, y=12)
        tk.Label(badge_f, text="  GPS DENIED  ",
                 font=("Courier New", 9, "bold"), fg="#ff4400", bg="#0a0b08",
                 highlightthickness=1, highlightbackground="#ff4400",
                 padx=4, pady=2).pack(side=tk.LEFT, padx=4)
        tk.Label(badge_f, text="  VISION ACTIVE  ",
                 font=("Courier New", 9, "bold"), fg="#44cc44", bg="#0a0b08",
                 highlightthickness=1, highlightbackground="#44cc44",
                 padx=4, pady=2).pack(side=tk.LEFT, padx=4)

        cy = self._sh // 2
        tk.Label(self._win,
                 text="GNSS-FREE NAVIGATION SYSTEM FOR UAVs",
                 font=("Courier New", 28, "bold"),
                 fg="#f0a030", bg="#0a0b08").place(relx=0.5, y=cy-160, anchor="center")

        tk.Label(self._win,
                 text="Vision-Based Localization  |  Dead Reckoning  |  Sensor Fusion",
                 font=("Courier New", 11),
                 fg="#5a4a20", bg="#0a0b08").place(relx=0.5, y=cy-115, anchor="center")

        self._cvs = tk.Canvas(self._win, width=860, height=110,
                              bg="#0a0b08", highlightthickness=0)
        self._cvs.place(relx=0.5, y=cy-30, anchor="center")

        self._status_bar = tk.Label(self._win,
                                    text="VNS ACTIVE  --  GPS DENIED  --  VISION LOCK",
                                    font=("Courier New", 12, "bold"),
                                    fg="#0a0b08", bg="#0a0b08", padx=20, pady=6)
        self._status_bar.place(relx=0.5, y=cy+110, anchor="center")

        tk.Label(self._win,
                 text="Muhammad Ahsan & Aittrah Sardar  |  Dr. Bushra Almas  |  IIT QAU 2026",
                 font=("Courier New", 9), fg="#5a4a20", bg="#0a0b08"
                 ).place(relx=0.5, rely=1.0, anchor="s", y=-18)

        tk.Label(self._win, text="Click anywhere to continue",
                 font=("Courier New", 8), fg="#3a3020", bg="#0a0b08"
                 ).place(relx=0.5, rely=1.0, anchor="s", y=-4)

    def _animate(self, step):
        if self._closed:
            return
        c = self._cvs
        if step == "drone":
            cx, cy = 30, 55
            c.create_oval(cx-10, cy-10, cx+10, cy+10, outline="#f0a030", width=2)
            for dx, dy in [(-18,-18),(18,-18),(-18,18),(18,18)]:
                c.create_line(cx+dx//2, cy+dy//2, cx+dx, cy+dy, fill="#5a4a20", width=2)
                c.create_oval(cx+dx-4, cy+dy-4, cx+dx+4, cy+dy+4,
                              outline="#5a4a20", fill="#1a1000")
            c.create_text(cx, cy+30, text="DRONE", fill="#5a4a20", font=("Courier New", 8))
        elif step == "uav_label":
            c.create_line(75, 55, 145, 55, fill="#f0a030", width=2, arrow=tk.LAST)
            c.create_rectangle(148, 38, 248, 72, outline="#f0a030", fill="#1a1000")
            c.create_text(198, 55, text="UAV IMAGE", fill="#f0a030",
                          font=("Courier New", 9, "bold"))
        elif step == "sat_box":
            for x in range(248, 318, 8):
                c.create_line(x, 55, min(x+5,318), 55, fill="#5a4a20", width=1)
            c.create_rectangle(318, 30, 448, 80, outline="#5a4a20", fill="#0f0d00")
            c.create_text(383, 47, text="GEO-REF", fill="#d4a843",
                          font=("Courier New", 8, "bold"))
            c.create_text(383, 63, text="SAT IMAGE", fill="#5a4a20",
                          font=("Courier New", 8))
        elif step == "match_box":
            for x in range(448, 518, 8):
                c.create_line(x, 55, min(x+5,518), 55, fill="#5a4a20", width=1)
            c.create_rectangle(518, 30, 638, 80, outline="#f0a030", fill="#1a1000",
                                tags="match_box")
            c.create_text(578, 55, text="IMAGE\nMATCHING", fill="#f0a030",
                          font=("Courier New", 8, "bold"), justify=tk.CENTER)
            self._pulse(c, 0)
        elif step == "dr_box":
            c.create_line(638, 55, 700, 55, fill="#f0a030", width=2, arrow=tk.LAST)
            c.create_rectangle(700, 30, 790, 80, outline="#5a4a20", fill="#0f0d00")
            c.create_text(745, 47, text="DEAD", fill="#d4a843",
                          font=("Courier New", 8, "bold"))
            c.create_text(745, 63, text="RECKONING", fill="#5a4a20",
                          font=("Courier New", 8))
        elif step == "pos_box":
            c.create_line(790, 55, 820, 55, fill="#44cc44", width=2, arrow=tk.LAST)
            c.create_rectangle(820, 22, 858, 88, outline="#44cc44", fill="#0a1a0a")
            c.create_text(839, 42, text="POS", fill="#44cc44",
                          font=("Courier New", 8, "bold"))
            c.create_text(839, 60, text="33.747N", fill="#44cc44",
                          font=("Courier New", 7))
            c.create_text(839, 74, text="73.137E", fill="#44cc44",
                          font=("Courier New", 7))
        elif step == "status_bar":
            self._status_bar.config(bg="#f0a030", fg="#0a0b08")
        elif step == "close":
            self._close()

    def _pulse(self, c, tick):
        if self._closed:
            return
        col = "#f0a030" if tick % 2 == 0 else "#805010"
        c.itemconfig("match_box", outline=col)
        self._win.after(400, self._pulse, c, tick+1)

    def _close(self):
        if self._closed:
            return
        self._closed = True
        self._win.destroy()
        self._root.deiconify()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class VNSApp:

    GE_PATHS = [
        r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
        r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe",
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            r"Google\Google Earth Pro\client\googleearth.exe"
        ),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VNS  --  GNSS-Free Navigation System")
        self.root.geometry("1500x880")
        self.root.minsize(1100, 650)
        self.root.configure(bg="#0a0b08")

        self._polygon:    Optional[PolygonData] = None
        self._satellite:  Optional[np.ndarray]  = None
        self._start:      Optional[tuple]        = None
        self._goal:       Optional[tuple]        = None
        self._waypoints:  Optional[list]         = None
        self._click_mode: Optional[str]          = None
        self._img_bounds: Optional[dict]         = None
        self._last_kml_t: float                  = 0.0

        self._kml  = KMLReader()
        self._conv = CoordConverter()
        self._plan = PathPlanner(spacing=15.0)

        self._db_sat_path:    Optional[str]    = None
        self._uav_match_path: Optional[str]    = None
        self._faiss_db:       Optional[object] = None

        self._workflow_visible = True

        self.root.withdraw()
        _SplashScreen(self.root)

        self._build_ui()
        self._start_kml_watcher()
        self._animate_radar(0)
        self._update_clock()

    # =========================================================================
    # UI BUILD
    # =========================================================================

    def _build_ui(self):
        self._build_topbar()
        tk.Frame(self.root, bg="#2a2200", height=1).pack(fill=tk.X)
        body = tk.Frame(self.root, bg="#0a0b08")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._build_sidebar(body)
        self._build_map_area(body)
        self._build_footer()

    # ── Topbar ────────────────────────────────────────────────────────────────

    def _build_topbar(self):
        nav = tk.Frame(self.root, bg="#0f110c", height=54)
        nav.pack(fill=tk.X)
        nav.pack_propagate(False)

        self._radar_cvs = tk.Canvas(nav, width=36, height=36,
                                    bg="#0f110c", highlightthickness=0)
        self._radar_cvs.pack(side=tk.LEFT, padx=(14, 4), pady=9)
        for r in [16, 11, 6]:
            self._radar_cvs.create_oval(18-r, 18-r, 18+r, 18+r,
                                        outline="#5a4a20", width=1)

        tk.Label(nav, text="VNS",
                 font=("Courier New", 16, "bold"),
                 fg="#f0a030", bg="#0f110c").pack(side=tk.LEFT, pady=12)
        tk.Label(nav, text="  GNSS-Free Navigation System",
                 font=("Courier New", 10),
                 fg="#5a4a20", bg="#0f110c").pack(side=tk.LEFT, pady=12)

        right = tk.Frame(nav, bg="#0f110c")
        right.pack(side=tk.RIGHT, padx=14)

        self._clock_lbl = tk.Label(right, text="",
                                   font=("Courier New", 9),
                                   fg="#5a4a20", bg="#0f110c")
        self._clock_lbl.pack(side=tk.RIGHT, padx=(10, 0))

        tk.Label(right, text="Muhammad Ahsan & Aittrah Sardar  |  FYP 2026",
                 font=("Courier New", 9),
                 fg="#3a3020", bg="#0f110c").pack(side=tk.RIGHT)

        self._sys_badge = StatusBadge(right, "READY")
        self._sys_badge.pack(side=tk.RIGHT, padx=(10, 10))

        mid = tk.Frame(nav, bg="#0f110c")
        mid.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(mid, text="  GPS DENIED  ",
                 font=("Courier New", 8, "bold"),
                 fg="#ff4400", bg="#0f110c",
                 highlightthickness=1, highlightbackground="#ff4400",
                 padx=3, pady=1).pack(side=tk.LEFT, padx=4)

        self._vision_badge = tk.Label(mid, text="  VISION: STANDBY  ",
                 font=("Courier New", 8, "bold"),
                 fg="#5a4a20", bg="#0f110c",
                 highlightthickness=1, highlightbackground="#5a4a20",
                 padx=3, pady=1)
        self._vision_badge.pack(side=tk.LEFT, padx=4)

        self._db_top_badge = tk.Label(mid, text="  DB: NOT LOADED  ",
                 font=("Courier New", 8, "bold"),
                 fg="#5a4a20", bg="#0f110c",
                 highlightthickness=1, highlightbackground="#2a2200",
                 padx=3, pady=1)
        self._db_top_badge.pack(side=tk.LEFT, padx=4)

    def _animate_radar(self, angle: int):
        c = self._radar_cvs
        cx = cy = 18; r = 14
        c.delete("sweep")
        x = cx + r * math.cos(math.radians(angle))
        y = cy - r * math.sin(math.radians(angle))
        c.create_line(cx, cy, x, y, fill="#f0a030", width=2, tags="sweep")
        for i in range(1, 5):
            a2 = angle + i * 12
            x2 = cx + r * math.cos(math.radians(a2))
            y2 = cy - r * math.sin(math.radians(a2))
            col = "#5a4a20" if i < 3 else "#3a3020"
            c.create_line(cx, cy, x2, y2, fill=col, width=1, tags="sweep")
        self.root.after(60, self._animate_radar, (angle - 15) % 360)

    def _update_clock(self):
        self._clock_lbl.config(text=time.strftime("%H:%M:%S UTC", time.gmtime()))
        self.root.after(1000, self._update_clock)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        outer = tk.Frame(parent, bg="#0f110c", width=300)
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=8)
        outer.pack_propagate(False)

        cvs = tk.Canvas(outer, bg="#0f110c", highlightthickness=0, bd=0)
        sb  = tk.Scrollbar(outer, orient="vertical", command=cvs.yview)
        self._sf = tk.Frame(cvs, bg="#0f110c")

        self._sf.bind("<Configure>",
                      lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.create_window((0, 0), window=self._sf, anchor="nw", width=300)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cvs.bind_all("<MouseWheel>",
                     lambda e: cvs.yview_scroll(-1*(e.delta//120), "units"))
        self._fill_sidebar(self._sf)

    def _fill_sidebar(self, p):

        def gap(h=6):
            tk.Frame(p, bg="#0f110c", height=h).pack(fill=tk.X)

        def card():
            f = tk.Frame(p, bg="#111408",
                         highlightthickness=1, highlightbackground="#2a2200")
            f.pack(fill=tk.X, padx=10, pady=3)
            return f

        def rlbl(fr, txt, w=4):
            return tk.Label(fr, text=txt, width=w,
                            font=("Courier New", 9), fg="#5a4a20", bg="#111408")

        # ── SECTION 01 ────────────────────────────────────────────────────────
        section_label(p, "01  Area of Interest", "#f0a030")
        gap(3)
        c1 = card()
        ib = tk.Frame(c1, bg="#111408")
        ib.pack(fill=tk.X, padx=12, pady=(10, 6))

        tk.Label(ib, text="[ START POINT ]",
                 font=("Courier New", 8, "bold"),
                 fg="#44cc44", bg="#111408").pack(anchor="w", pady=(0, 4))

        for attr, default, hi in [("_e_slat","33.7470","#44cc44"),
                                   ("_e_slon","73.1370","#44cc44")]:
            row = tk.Frame(ib, bg="#111408"); row.pack(fill=tk.X, pady=(0, 2))
            rlbl(row, "Lat" if "lat" in attr else "Lon").pack(side=tk.LEFT)
            e = make_entry(row, default, hi_color=hi)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(4,0))
            setattr(self, attr, e)

        tk.Label(ib, text="[ END POINT ]",
                 font=("Courier New", 8, "bold"),
                 fg="#ff4400", bg="#111408").pack(anchor="w", pady=(8, 4))

        for attr, default, hi in [("_e_elat","33.7550","#ff4400"),
                                   ("_e_elon","73.1460","#ff4400")]:
            row = tk.Frame(ib, bg="#111408"); row.pack(fill=tk.X, pady=(0, 2))
            rlbl(row, "Lat" if "lat" in attr else "Lon").pack(side=tk.LEFT)
            e = make_entry(row, default, hi_color=hi)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(4,0))
            setattr(self, attr, e)

        gap(6)
        br = tk.Frame(ib, bg="#111408"); br.pack(fill=tk.X, pady=(0, 6))
        ProfButton(br, "Plot Points", self._on_plot_points,
                   style="success", icon=">").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0,4))
        ProfButton(br, "Clear", self._on_clear_points,
                   style="ghost", icon="x").pack(side=tk.LEFT)

        ProfButton(ib, "Open in Google Earth Pro", self._on_open_ge,
                   style="primary", icon="*").pack(fill=tk.X)

        self._pts_info = tk.Label(ib, text="",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#111408", justify=tk.LEFT)
        self._pts_info.pack(anchor="w", pady=(6, 0))

        tk.Label(ib,
                 text=("1. Google Earth opens at your location\n"
                       "2. Add > Polygon > draw boundary\n"
                       "3. Right-click > Save Place As > KML"),
                 font=("Courier New", 8),
                 fg="#3a3020", bg="#111408", justify=tk.LEFT
                 ).pack(anchor="w", pady=(6, 8))

        # ── SECTION 02 ────────────────────────────────────────────────────────
        section_label(p, "02  Import KML", "#44cc44")
        gap(3)
        c2 = card()
        f2 = tk.Frame(c2, bg="#111408"); f2.pack(fill=tk.X, padx=12, pady=10)

        wr = tk.Frame(f2, bg="#111408"); wr.pack(fill=tk.X, pady=(0,6))
        self._watch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(wr, text="Auto-detect when KML is saved",
                       variable=self._watch_var,
                       bg="#111408", fg="#5a4a20",
                       selectcolor="#07080a",
                       activebackground="#111408",
                       font=("Courier New", 9)).pack(side=tk.LEFT)

        ProfButton(f2, "Browse KML File", self._on_browse_kml,
                   style="ghost", icon="@").pack(fill=tk.X, pady=(0,5))

        self._kml_info = tk.Label(f2, text="Waiting for KML file...",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#07080a",
                                  justify=tk.LEFT, wraplength=250, anchor="w")
        self._kml_info.pack(fill=tk.X, padx=1, pady=1, ipadx=6, ipady=5)

        gap(3)
        ProfButton(f2, "Load Satellite Image", self._on_load_sat,
                   style="ghost", icon="~").pack(fill=tk.X, pady=(6,3))

        self._img_info = tk.Label(f2, text="No image loaded",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#111408")
        self._img_info.pack(anchor="w", pady=(0,3))

        # ── SECTION 03 ────────────────────────────────────────────────────────
        section_label(p, "03  Start & Goal on Map", "#f0a030")
        gap(3)
        c3 = card()
        f3 = tk.Frame(c3, bg="#111408"); f3.pack(fill=tk.X, padx=12, pady=10)

        sr3 = tk.Frame(f3, bg="#111408"); sr3.pack(fill=tk.X, pady=(0,3))
        self._btn_start = ProfButton(sr3, "Set Start on Map", self._on_set_start,
                                     style="success", icon=">")
        self._btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._badge_start = StatusBadge(sr3, "NOT SET")
        self._badge_start.pack(side=tk.RIGHT, padx=(6,0))

        self._lbl_start = tk.Label(f3, text="--",
                                   font=("Courier New", 8),
                                   fg="#5a4a20", bg="#111408")
        self._lbl_start.pack(anchor="w", pady=(0,8))

        gr3 = tk.Frame(f3, bg="#111408"); gr3.pack(fill=tk.X, pady=(0,3))
        self._btn_goal = ProfButton(gr3, "Set Goal on Map", self._on_set_goal,
                                    style="danger", icon="!")
        self._btn_goal.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._badge_goal = StatusBadge(gr3, "NOT SET")
        self._badge_goal.pack(side=tk.RIGHT, padx=(6,0))

        self._lbl_goal = tk.Label(f3, text="--",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#111408")
        self._lbl_goal.pack(anchor="w", pady=(0,3))

        tk.Label(f3, text="Click on map after pressing button above",
                 font=("Courier New", 8),
                 fg="#3a3020", bg="#111408").pack(anchor="w")

        # ── SECTION 04 ────────────────────────────────────────────────────────
        section_label(p, "04  Path Planning", "#9966ff")
        gap(3)
        c4 = card()
        f4 = tk.Frame(c4, bg="#111408"); f4.pack(fill=tk.X, padx=12, pady=10)

        gr4 = tk.Frame(f4, bg="#111408"); gr4.pack(fill=tk.X, pady=(0,8))
        tk.Label(gr4, text="Grid spacing (m):",
                 font=("Courier New", 9), fg="#5a4a20", bg="#111408").pack(side=tk.LEFT)
        self._spacing_var = tk.StringVar(value="15")
        tk.Spinbox(gr4, from_=5, to=100, increment=5,
                   textvariable=self._spacing_var, width=6,
                   bg="#07080a", fg="#f0a030",
                   buttonbackground="#1a1c14",
                   font=("Courier New", 10),
                   relief=tk.FLAT).pack(side=tk.RIGHT)

        ProfButton(f4, "Run A* Path Planning", self._on_run_plan,
                   style="purple", icon="#").pack(fill=tk.X)

        self._path_info = tk.Label(f4, text="No path planned yet",
                                   font=("Courier New", 8),
                                   fg="#5a4a20", bg="#111408")
        self._path_info.pack(anchor="w", pady=(6,0))

        # ── SECTION 05 ────────────────────────────────────────────────────────
        section_label(p, "05  Export Waypoints", "#5a4a20")
        gap(3)
        c5 = card()
        f5 = tk.Frame(c5, bg="#111408"); f5.pack(fill=tk.X, padx=12, pady=10)

        er5 = tk.Frame(f5, bg="#111408"); er5.pack(fill=tk.X)
        ProfButton(er5, "Export CSV", self._on_export_csv,
                   style="ghost", icon="$").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0,4))
        ProfButton(er5, "Export JSON", self._on_export_json,
                   style="ghost", icon="{}").pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._export_info = tk.Label(f5, text="",
                                     font=("Courier New", 8),
                                     fg="#44cc44", bg="#111408")
        self._export_info.pack(anchor="w", pady=(5,0))

        # ── SECTION 06 ────────────────────────────────────────────────────────
        section_label(p, "06  Database  (DINOv2+FAISS)", "#9966ff")
        gap(3)
        c6 = card()
        f6 = tk.Frame(c6, bg="#111408"); f6.pack(fill=tk.X, padx=12, pady=10)

        for lbl, attr, default in [("Lat","_db_lat","33.7470"),
                                    ("Lon","_db_lon","73.1370"),
                                    ("Alt","_db_alt","550.0")]:
            row = tk.Frame(f6, bg="#111408"); row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, width=4,
                     font=("Courier New", 9), fg="#5a4a20", bg="#111408").pack(side=tk.LEFT)
            e = make_entry(row, default, w=14)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(4,0))
            setattr(self, attr, e)

        gap(5)
        ProfButton(f6, "Select Satellite Image", self._on_db_select_image,
                   style="ghost", icon="~").pack(fill=tk.X, pady=(0,3))

        self._db_img_lbl = tk.Label(f6, text="No image selected",
                                    font=("Courier New", 8),
                                    fg="#5a4a20", bg="#111408",
                                    wraplength=250, justify=tk.LEFT)
        self._db_img_lbl.pack(anchor="w", pady=(0,5))

        br6 = tk.Frame(f6, bg="#111408"); br6.pack(fill=tk.X, pady=(0,3))
        ProfButton(br6, "Build Database", self._on_build_faiss_db,
                   style="primary", icon="B").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
        ProfButton(br6, "Load", self._on_load_faiss_db,
                   style="ghost", icon="^").pack(side=tk.LEFT)

        self._db_progress = tk.Label(f6, text="",
                                     font=("Courier New", 8),
                                     fg="#5a4a20", bg="#111408")
        self._db_progress.pack(anchor="w", pady=(0,3))

        self._db_status = tk.Label(f6, text="Status: not built",
                                   font=("Courier New", 8),
                                   fg="#5a4a20", bg="#111408",
                                   justify=tk.LEFT, wraplength=250)
        self._db_status.pack(anchor="w", pady=(0,5))

        ProfButton(f6, "Clear Database", self._on_clear_faiss_db,
                   style="danger", icon="X").pack(fill=tk.X)

        # ── SECTION 07 ────────────────────────────────────────────────────────
        section_label(p, "07  Match UAV Image", "#f0a030")
        gap(3)
        c7 = card()
        f7 = tk.Frame(c7, bg="#111408"); f7.pack(fill=tk.X, padx=12, pady=10)

        ProfButton(f7, "Select UAV Image", self._on_match_select_uav,
                   style="ghost", icon="@").pack(fill=tk.X, pady=(0,3))

        self._match_img_lbl = tk.Label(f7, text="No UAV image selected",
                                       font=("Courier New", 8),
                                       fg="#5a4a20", bg="#111408",
                                       wraplength=250, justify=tk.LEFT)
        self._match_img_lbl.pack(anchor="w", pady=(0,5))

        ProfButton(f7, "Run Matching", self._on_run_matching,
                   style="purple", icon="#").pack(fill=tk.X, pady=(0,5))

        self._match_results = tk.Label(f7, text="No results yet",
                                       font=("Courier New", 8),
                                       fg="#5a4a20", bg="#111408",
                                       justify=tk.LEFT, wraplength=250)
        self._match_results.pack(anchor="w")

        # ── SECTION 08 ────────────────────────────────────────────────────────
        section_label(p, "08  Demo Video", "#44aaff")
        gap(3)
        c8 = card()
        f8 = tk.Frame(c8, bg="#111408"); f8.pack(fill=tk.X, padx=12, pady=10)

        ProfButton(f8, "Watch Demo Video", self._on_play_video,
                   style="ghost", icon=">").pack(fill=tk.X, pady=(0,3))

        self._video_lbl = tk.Label(f8, text="Upload your demo walkthrough",
                                   font=("Courier New", 8),
                                   fg="#3a3020", bg="#111408")
        self._video_lbl.pack(anchor="w", pady=(0,6))

        gap(16)

    # ── Map Area ──────────────────────────────────────────────────────────────

    def _build_map_area(self, parent):
        wrap = tk.Frame(parent, bg="#0f110c")
        wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)

        tb = tk.Frame(wrap, bg="#111408", height=36)
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)

        tk.Label(tb, text="  MAP VIEW",
                 font=("Courier New", 9, "bold"),
                 fg="#5a4a20", bg="#111408").pack(side=tk.LEFT, padx=8)

        self._coord_lbl = tk.Label(tb, text="",
                                   font=("Courier New", 9),
                                   fg="#f0a030", bg="#111408")
        self._coord_lbl.pack(side=tk.LEFT, padx=16)

        tk.Button(tb, text="[ WORKFLOW ]",
                  font=("Courier New", 8, "bold"),
                  fg="#5a4a20", bg="#111408",
                  relief=tk.FLAT, bd=0, cursor="hand2",
                  activebackground="#111408", activeforeground="#f0a030",
                  command=self._toggle_workflow).pack(side=tk.RIGHT, padx=8)

        self._mode_badge = StatusBadge(tb, "VIEW")
        self._mode_badge.pack(side=tk.RIGHT, padx=6)

        tk.Frame(wrap, bg="#2a2200", height=1).pack(fill=tk.X)

        self._map_frame = tk.Frame(wrap, bg="#050800")
        self._map_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Label(self._map_frame, bg="#050800", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Button-1>", self._on_map_click)
        self._canvas.bind("<Motion>",   self._on_hover)

        self._empty = tk.Label(
            self._map_frame,
            text=("VNS\n\nNo satellite image loaded\n\n"
                  "Enter coordinates  >  Open Google Earth Pro\n"
                  "Draw polygon  >  Save as KML\n"
                  "Load satellite image to begin"),
            font=("Courier New", 11),
            fg="#5a4a20", bg="#050800", justify=tk.CENTER,
        )
        self._empty.place(relx=0.5, rely=0.5, anchor="center")

        sb2 = tk.Frame(wrap, bg="#111408", height=24)
        sb2.pack(fill=tk.X)
        sb2.pack_propagate(False)

        self._status_l = tk.Label(sb2, text="Ready",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#111408")
        self._status_l.pack(side=tk.LEFT, padx=10)

        self._status_r = tk.Label(sb2, text="",
                                  font=("Courier New", 8),
                                  fg="#5a4a20", bg="#111408")
        self._status_r.pack(side=tk.RIGHT, padx=10)

        # Workflow panel
        self._workflow_frame = tk.Frame(wrap, bg="#111408", height=140,
                                        highlightthickness=1,
                                        highlightbackground="#2a2200")
        self._workflow_frame.pack(fill=tk.X)
        self._workflow_frame.pack_propagate(False)
        self._build_workflow_panel(self._workflow_frame)

        # Console
        con_frame = tk.Frame(wrap, bg="#111408", height=120)
        con_frame.pack(fill=tk.X)
        con_frame.pack_propagate(False)

        tk.Frame(con_frame, bg="#2a2200", height=1).pack(fill=tk.X)

        con_hdr = tk.Frame(con_frame, bg="#111408")
        con_hdr.pack(fill=tk.X, padx=10, pady=(3,0))

        tk.Label(con_hdr, text="[ CONSOLE ]",
                 font=("Courier New", 8, "bold"),
                 fg="#5a4a20", bg="#111408").pack(side=tk.LEFT)

        tk.Button(con_hdr, text="clear",
                  font=("Courier New", 8),
                  fg="#3a3020", bg="#111408",
                  relief=tk.FLAT, bd=0, cursor="hand2",
                  command=self._clear_log).pack(side=tk.RIGHT)

        self._log_box = tk.Text(con_frame,
                                bg="#07080a", fg="#f0a030",
                                font=("Courier New", 8),
                                state=tk.DISABLED,
                                relief=tk.FLAT, bd=0, wrap=tk.WORD)
        self._log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2,6))

        for tag, col in [("info","#44aaff"),("success","#44cc44"),
                         ("warning","#f0a030"),("error","#ff4400")]:
            self._log_box.tag_config(tag, foreground=col)

    def _build_workflow_panel(self, parent):
        tk.Label(parent, text="[ NAVIGATION PIPELINE ]",
                 font=("Courier New", 8, "bold"),
                 fg="#5a4a20", bg="#111408").pack(anchor="w", padx=12, pady=(6,0))

        self._wf_canvas = tk.Canvas(parent, bg="#111408",
                                    highlightthickness=0, height=100)
        self._wf_canvas.pack(fill=tk.X, padx=12, pady=(2,6))
        self._wf_canvas.bind("<Configure>", lambda e: self._draw_workflow())
        self._draw_workflow()

    def _draw_workflow(self):
        c = self._wf_canvas
        c.delete("all")
        W = c.winfo_width()
        if W < 50:
            return
        steps = [
            ("DRONE",     "UAV\nplatform"),
            ("UAV IMAGE", "Camera\ncapture"),
            ("SAT REF",   "Geo-ref\nsatellite"),
            ("MATCHING",  "DINOv2\n+ ORB"),
            ("DEAD RECK", "IMU\nfusion"),
            ("POSITION",  "Estimated\nlocation"),
        ]
        n = len(steps); bw = 82; bh = 44
        gap = (W - n * bw) / (n + 1)
        last_x2 = None
        for i, (title, sub) in enumerate(steps):
            x1 = gap + i * (bw + gap); x2 = x1 + bw
            y1 = 18; y2 = y1 + bh
            col  = "#44cc44" if i == n-1 else "#f0a030"
            bg_c = "#0a1a0a" if i == n-1 else "#1a1000"
            if last_x2 is not None:
                c.create_line(last_x2, (y1+y2)/2, x1, (y1+y2)/2,
                              fill="#5a4a20", width=1, dash=(4,3),
                              arrow=tk.LAST, arrowshape=(8,10,3))
            c.create_rectangle(x1, y1, x2, y2, outline=col, fill=bg_c, width=1)
            c.create_text((x1+x2)/2, y1+14, text=title, fill=col,
                          font=("Courier New", 7, "bold"))
            c.create_text((x1+x2)/2, y1+30, text=sub, fill="#5a4a20",
                          font=("Courier New", 6), justify=tk.CENTER)
            c.create_text(x1+6, y1+6, text=str(i+1), fill="#5a4a20",
                          font=("Courier New", 6))
            last_x2 = x2

    def _toggle_workflow(self):
        if self._workflow_visible:
            self._workflow_frame.pack_forget()
            self._workflow_visible = False
        else:
            self._workflow_frame.pack(fill=tk.X,
                                      before=self._workflow_frame.master.winfo_children()[-1])
            self._workflow_visible = True
            self.root.after(50, self._draw_workflow)

    def _build_footer(self):
        ft = tk.Frame(self.root, bg="#0f110c", height=26)
        ft.pack(fill=tk.X, side=tk.BOTTOM)
        ft.pack_propagate(False)
        tk.Frame(ft, bg="#2a2200", height=1).pack(fill=tk.X, side=tk.TOP)

        tk.Label(ft, text="  FR-1 to FR-17  |  M1-M6 Modules",
                 font=("Courier New", 8), fg="#3a3020",
                 bg="#0f110c").pack(side=tk.LEFT, padx=6)

        tk.Label(ft, text="GNSS-FREE NAVIGATION SYSTEM  --  IIT QAU 2026",
                 font=("Courier New", 8, "bold"), fg="#5a4a20",
                 bg="#0f110c").pack(side=tk.LEFT, expand=True)

        self._footer_stats = tk.Label(ft, text="DB: 0 patches  |  Matches: 0",
                                      font=("Courier New", 8), fg="#3a3020",
                                      bg="#0f110c")
        self._footer_stats.pack(side=tk.RIGHT, padx=10)

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def _on_plot_points(self):
        try:
            s_lat = float(self._e_slat.get().strip())
            s_lon = float(self._e_slon.get().strip())
            e_lat = float(self._e_elat.get().strip())
            e_lon = float(self._e_elon.get().strip())
        except ValueError:
            self._log("Invalid coordinates -- use decimal format e.g. 33.7470", "error")
            return
        if not (-90 <= s_lat <= 90 and -90 <= e_lat <= 90):
            self._log("Latitude must be -90 to 90", "error"); return
        if not (-180 <= s_lon <= 180 and -180 <= e_lon <= 180):
            self._log("Longitude must be -180 to 180", "error"); return
        self._start = (s_lat, s_lon)
        self._goal  = (e_lat, e_lon)
        dist = self._conv.haversine_m(s_lat, s_lon, e_lat, e_lon)
        self._badge_start.set("SET", "ok")
        self._badge_goal.set("SET", "ok")
        self._lbl_start.config(text=f"{s_lat:.6f},  {s_lon:.6f}", fg="#44cc44")
        self._lbl_goal.config(text=f"{e_lat:.6f},  {e_lon:.6f}", fg="#ff4400")
        self._pts_info.config(
            text=(f"+ Start ({s_lat:.5f}, {s_lon:.5f})\n"
                  f"+ End   ({e_lat:.5f}, {e_lon:.5f})\n"
                  f"  Distance ~ {dist:.0f} m"),
            fg="#44cc44")
        self._log(f"Start ({s_lat:.5f}, {s_lon:.5f})  End ({e_lat:.5f}, {e_lon:.5f})  dist={dist:.0f}m", "success")
        self._sys_badge.set("POINTS SET", "ok")
        if self._img_bounds is None and self._satellite is not None:
            pad = 0.005
            self._img_bounds = {
                "north": max(s_lat,e_lat)+pad, "south": min(s_lat,e_lat)-pad,
                "east":  max(s_lon,e_lon)+pad, "west":  min(s_lon,e_lon)-pad,
            }
        if self._satellite is not None:
            self._render()
        else:
            self._log("Load satellite image to see points on map", "info")

    def _on_clear_points(self):
        self._start = None; self._goal = None
        self._pts_info.config(text="")
        self._badge_start.set("NOT SET", "idle")
        self._badge_goal.set("NOT SET", "idle")
        self._lbl_start.config(text="--", fg="#5a4a20")
        self._lbl_goal.config(text="--", fg="#5a4a20")
        for e, v in [(self._e_slat,"33.7470"),(self._e_slon,"73.1370"),
                     (self._e_elat,"33.7550"),(self._e_elon,"73.1460")]:
            e.delete(0, tk.END); e.insert(0, v)
        self._log("Points cleared", "info")
        if self._satellite is not None:
            self._render()

    def _on_open_ge(self):
        try:
            s_lat = float(self._e_slat.get().strip())
            s_lon = float(self._e_slon.get().strip())
            e_lat = float(self._e_elat.get().strip())
            e_lon = float(self._e_elon.get().strip())
        except ValueError:
            messagebox.showwarning("Coordinates Required",
                                   "Enter valid start and end coordinates first.")
            return
        c_lat = (s_lat + e_lat) / 2; c_lon = (s_lon + e_lon) / 2
        diff  = max(abs(e_lat-s_lat), abs(e_lon-s_lon))
        alt   = max(800, int(diff * 111320 * 2.5))
        kml   = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document><name>VNS Mission Area</name>
    <LookAt><longitude>{c_lon}</longitude><latitude>{c_lat}</latitude>
      <altitude>0</altitude><heading>0</heading><tilt>0</tilt><range>{alt}</range></LookAt>
    <Placemark><name>Start Point</name>
      <Style><IconStyle><color>ff00ff00</color></IconStyle></Style>
      <Point><coordinates>{s_lon},{s_lat},0</coordinates></Point></Placemark>
    <Placemark><name>End Point</name>
      <Style><IconStyle><color>ff0000ff</color></IconStyle></Style>
      <Point><coordinates>{e_lon},{e_lat},0</coordinates></Point></Placemark>
  </Document></kml>"""
        fly = Path("data/kml/fly_to.kml")
        fly.parent.mkdir(parents=True, exist_ok=True)
        fly.write_text(kml)
        ge = None
        for path in self.GE_PATHS:
            if Path(path).exists():
                ge = path; break
        if ge:
            subprocess.Popen([ge, str(fly.absolute())])
            self._log(f"Google Earth Pro opened at ({c_lat:.4f}, {c_lon:.4f})", "success")
        else:
            try:
                os.startfile(str(fly.absolute()))
                self._log("Opened in default KML app", "success")
            except Exception:
                messagebox.showinfo("Google Earth Pro",
                    f"Open Google Earth Pro manually and navigate to:\n"
                    f"  {c_lat:.5f}, {c_lon:.5f}\n\n"
                    "Then draw polygon and save as KML to: data\\kml\\")
        self._log("Draw polygon > Right-click > Save Place As > KML > save to data/kml/", "info")
        self._sys_badge.set("WAITING KML", "working")

    def _start_kml_watcher(self):
        kml_dir = Path("data/kml")
        kml_dir.mkdir(parents=True, exist_ok=True)
        def watch():
            while True:
                if self._watch_var.get():
                    files = [f for f in kml_dir.glob("*.kml") if f.name != "fly_to.kml"]
                    if files:
                        newest = max(files, key=lambda f: f.stat().st_mtime)
                        mt = newest.stat().st_mtime
                        if mt > self._last_kml_t:
                            self._last_kml_t = mt
                            self.root.after(0, self._load_kml, str(newest))
                time.sleep(2)
        threading.Thread(target=watch, daemon=True).start()

    def _on_browse_kml(self):
        p = filedialog.askopenfilename(
            title="Select KML File", initialdir="data/kml",
            filetypes=[("KML","*.kml"),("All","*.*")])
        if p:
            self._load_kml(p)

    def _load_kml(self, path: str):
        self._log(f"Reading: {Path(path).name}", "info")
        poly = self._kml.read(path)
        if poly is None:
            self._log("Could not parse polygon from KML", "error")
            self._kml_info.config(text="[!] Invalid or empty KML", fg="#ff4400")
            return
        self._polygon = poly
        Path("data/kml").mkdir(parents=True, exist_ok=True)
        with open("data/kml/polygon.json", "w") as f:
            json.dump(asdict(poly), f, indent=2)
        self._kml_info.config(
            text=(f"[+] {poly.name}\n"
                  f"    {len(poly.coordinates)} vertices\n"
                  f"    N {poly.north:.5f}  S {poly.south:.5f}\n"
                  f"    E {poly.east:.5f}  W {poly.west:.5f}"),
            fg="#f0a030")
        self._log(f"Polygon: {len(poly.coordinates)} pts  center ({poly.center_lat:.4f},{poly.center_lon:.4f})", "success")
        self._sys_badge.set("KML READY", "ok")
        if self._satellite is not None:
            self._render()

    def _on_load_sat(self):
        p = filedialog.askopenfilename(
            title="Select Satellite Image",
            initialdir="data/satellite_raw",
            filetypes=[("Images","*.jpg *.jpeg *.png *.tif *.tiff"),("All","*.*")])
        if not p: return
        img = cv2.imread(p)
        if img is None:
            self._log("Cannot read image", "error"); return
        self._satellite = img
        self._empty.place_forget()
        self._img_info.config(
            text=f"[+] {Path(p).name}  ({img.shape[1]}x{img.shape[0]})",
            fg="#44cc44")
        self._log(f"Image {img.shape[1]}x{img.shape[0]}", "success")
        self._render()

    def _on_set_start(self):
        if self._satellite is None:
            self._log("Load satellite image first", "warning"); return
        self._click_mode = "start"
        self._mode_badge.set("CLICK START", "working")
        self._log("Click on map to place START", "info")

    def _on_set_goal(self):
        if self._satellite is None:
            self._log("Load satellite image first", "warning"); return
        self._click_mode = "goal"
        self._mode_badge.set("CLICK GOAL", "working")
        self._log("Click on map to place GOAL", "info")

    def _on_map_click(self, ev):
        if self._click_mode is None: return
        if self._img_bounds is None:
            self._log("Load image / import KML first", "warning"); return
        w = self._canvas.winfo_width(); h = self._canvas.winfo_height()
        b = self._img_bounds
        lat, lon = self._conv.pixel_to_latlon(ev.x, ev.y,
                                               b["north"],b["south"],b["west"],b["east"],w,h)
        utm_e, utm_n, _ = self._conv.latlon_to_utm(lat, lon)
        if self._click_mode == "start":
            self._start = (lat, lon)
            self._badge_start.set("SET", "ok")
            self._lbl_start.config(text=f"{lat:.6f},  {lon:.6f}", fg="#44cc44")
            self._e_slat.delete(0, tk.END); self._e_slat.insert(0, f"{lat:.6f}")
            self._e_slon.delete(0, tk.END); self._e_slon.insert(0, f"{lon:.6f}")
            self._log(f"Start ({lat:.5f}, {lon:.5f})  UTM {utm_e:.0f}E {utm_n:.0f}N", "success")
        elif self._click_mode == "goal":
            self._goal = (lat, lon)
            self._badge_goal.set("SET", "ok")
            self._lbl_goal.config(text=f"{lat:.6f},  {lon:.6f}", fg="#ff4400")
            self._e_elat.delete(0, tk.END); self._e_elat.insert(0, f"{lat:.6f}")
            self._e_elon.delete(0, tk.END); self._e_elon.insert(0, f"{lon:.6f}")
            self._log(f"Goal  ({lat:.5f}, {lon:.5f})  UTM {utm_e:.0f}E {utm_n:.0f}N", "success")
        if self._start and self._goal:
            d = self._conv.haversine_m(*self._start, *self._goal)
            self._pts_info.config(
                text=(f"+ Start ({self._start[0]:.5f}, {self._start[1]:.5f})\n"
                      f"+ End   ({self._goal[0]:.5f}, {self._goal[1]:.5f})\n"
                      f"  Distance ~ {d:.0f} m"),
                fg="#44cc44")
        self._click_mode = None
        self._mode_badge.set("VIEW", "idle")
        self._render()

    def _on_hover(self, ev):
        if self._img_bounds is None: return
        w = self._canvas.winfo_width(); h = self._canvas.winfo_height()
        b = self._img_bounds
        lat, lon = self._conv.pixel_to_latlon(ev.x, ev.y,
                                               b["north"],b["south"],b["west"],b["east"],w,h)
        self._coord_lbl.config(text=f"Lat {lat:.6f}   Lon {lon:.6f}")

    def _on_run_plan(self):
        if self._polygon is None:
            self._log("Import KML first", "warning"); return
        if self._start is None:
            self._log("Set start point", "warning"); return
        if self._goal is None:
            self._log("Set goal point", "warning"); return
        try:
            sp = float(self._spacing_var.get())
        except ValueError:
            sp = 15.0
        self._plan = PathPlanner(spacing=sp)
        self._log(f"A* running  grid={sp}m  start={self._start}  goal={self._goal}", "info")
        self._sys_badge.set("PLANNING", "working")
        self._path_info.config(text="Computing path...", fg="#f0a030")
        def run():
            wps = self._plan.plan(self._start[0], self._start[1],
                                  self._goal[0],  self._goal[1],
                                  self._polygon.coordinates)
            self.root.after(0, self._path_done, wps)
        threading.Thread(target=run, daemon=True).start()

    def _path_done(self, wps):
        if wps is None:
            self._log("No path found -- try larger grid spacing or different points", "error")
            self._path_info.config(text="[!] No path found", fg="#ff4400")
            self._sys_badge.set("FAILED", "error"); return
        self._waypoints = wps
        dist = sum(w.dist_m for w in wps)
        self._path_info.config(
            text=f"[+] {len(wps)} waypoints  |  {dist:.0f} m total",
            fg="#44cc44")
        self._log(f"Path: {len(wps)} waypoints  {dist:.0f}m", "success")
        self._sys_badge.set("PATH READY", "ok")
        self._status_r.config(text=f"{len(wps)} waypoints  |  {dist:.0f} m")
        self._render()

    def _on_export_csv(self):
        if not self._waypoints:
            self._log("Plan path first", "warning"); return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="waypoints.csv",
            initialdir="data/waypoints", filetypes=[("CSV","*.csv")])
        if not p: return
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index","latitude","longitude","easting_m","northing_m","dist_m"])
            for wp in self._waypoints:
                w.writerow([wp.index, round(wp.lat,8), round(wp.lon,8),
                             round(wp.easting,2), round(wp.northing,2), round(wp.dist_m,2)])
        self._export_info.config(text=f"[+] CSV saved: {Path(p).name}")
        self._log(f"CSV exported: {Path(p).name}", "success")

    def _on_export_json(self):
        if not self._waypoints:
            self._log("Plan path first", "warning"); return
        p = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="waypoints.json",
            initialdir="data/waypoints", filetypes=[("JSON","*.json")])
        if not p: return
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_waypoints": len(self._waypoints),
            "total_distance_m": sum(w.dist_m for w in self._waypoints),
            "waypoints": [asdict(w) for w in self._waypoints],
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        self._export_info.config(text=f"[+] JSON saved: {Path(p).name}")
        self._log(f"JSON exported: {Path(p).name}", "success")

    def _render(self):
        if self._satellite is None: return
        img = self._satellite.copy()
        h, w = img.shape[:2]
        if self._polygon:
            b = {"north":self._polygon.north,"south":self._polygon.south,
                 "east":self._polygon.east,"west":self._polygon.west}
        elif self._start and self._goal:
            pad = 0.005
            b = {"north":max(self._start[0],self._goal[0])+pad,
                 "south":min(self._start[0],self._goal[0])-pad,
                 "east":max(self._start[1],self._goal[1])+pad,
                 "west":min(self._start[1],self._goal[1])-pad}
        else:
            b = {"north":33.760,"south":33.735,"east":73.150,"west":73.120}
        self._img_bounds = b

        def px(lat, lon):
            return self._conv.latlon_to_pixel(lat, lon,
                                              b["north"],b["south"],b["west"],b["east"],w,h)

        if self._polygon and self._polygon.coordinates:
            pts = np.array([list(px(la,lo)) for la,lo in self._polygon.coordinates], np.int32)
            ov  = img.copy()
            cv2.fillPoly(ov, [pts], (48, 96, 160))
            img = cv2.addWeighted(img, 0.82, ov, 0.18, 0)
            cv2.polylines(img, [pts], True, (48, 96, 160), 2, cv2.LINE_AA)

        if self._waypoints and len(self._waypoints) > 1:
            wpts = [list(px(wp.lat,wp.lon)) for wp in self._waypoints]
            for i in range(1, len(wpts)):
                cv2.line(img, tuple(wpts[i-1]), tuple(wpts[i]), (153,102,255), 2, cv2.LINE_AA)
            for i, pt in enumerate(wpts):
                if 0 < i < len(wpts)-1:
                    cv2.circle(img, tuple(pt), 4, (153,102,255), -1)

        if self._start and self._goal:
            for a, bb in self._dashed(px(*self._start), px(*self._goal)):
                cv2.line(img, a, bb, (180,180,180), 1, cv2.LINE_AA)

        if self._start:
            p = list(px(*self._start))
            cv2.circle(img, tuple(p), 18, (68,204,68), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 13, (68,204,68), -1)
            cv2.circle(img, tuple(p), 15, (255,255,255), 2, cv2.LINE_AA)
            self._draw_label(img, "START", p[0]+20, p[1], (68,204,68))

        if self._goal:
            p = list(px(*self._goal))
            cv2.circle(img, tuple(p), 18, (255,68,0), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 13, (255,68,0), -1)
            cv2.circle(img, tuple(p), 15, (255,255,255), 2, cv2.LINE_AA)
            self._draw_label(img, "END", p[0]+20, p[1], (255,68,0))

        self._show(img)

    def _draw_label(self, img, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
        cv2.rectangle(img, (x-4,y-th-6), (x+tw+4,y+4), (18,18,18), -1)
        cv2.putText(img, text, (x,y), font, 0.55, color, 2, cv2.LINE_AA)

    def _dashed(self, p1, p2, dash=14, gap=6):
        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
        L  = math.sqrt(dx*dx+dy*dy)
        if L == 0: return []
        ux, uy = dx/L, dy/L
        segs = []; d = 0.0
        while d < L:
            de = min(d+dash, L)
            segs.append(((int(p1[0]+ux*d),int(p1[1]+uy*d)),
                         (int(p1[0]+ux*de),int(p1[1]+uy*de))))
            d += dash+gap
        return segs

    def _show(self, img: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10: cw = 900
        if ch < 10: ch = 600
        disp  = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        disp  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self._canvas.config(image=photo, text="")
        self._canvas.image = photo

    def _log(self, msg: str, level: str = ""):
        ts = time.strftime("%H:%M:%S")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(tk.END, f"[{ts}]  {msg}\n", level if level else ())
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)
        self._status_l.config(text=msg[:70]+("..." if len(msg)>70 else ""))

    def _clear_log(self):
        self._log_box.config(state=tk.NORMAL)
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state=tk.DISABLED)

    def _get_faiss_db(self) -> Optional[object]:
        if not _FAISS_AVAILABLE:
            messagebox.showerror("Import Error",
                "FAISSDatabase could not be loaded.\n"
                "Install: pip install faiss-cpu torch torchvision")
            return None
        if self._faiss_db is None:
            self._faiss_db = FAISSDatabase()
        return self._faiss_db

    def _on_db_select_image(self):
        p = filedialog.askopenfilename(
            title="Select Satellite Image",
            initialdir="data/satellite/raw",
            filetypes=[("Images","*.jpg *.jpeg *.png *.tif *.tiff"),("All","*.*")])
        if not p: return
        self._db_sat_path = p
        self._db_img_lbl.config(text=f"[+] {Path(p).name}", fg="#44cc44")
        self._log(f"Satellite image selected: {Path(p).name}", "info")

    def _on_build_faiss_db(self):
        if not self._db_sat_path:
            messagebox.showwarning("No Image","Select a satellite image first."); return
        try:
            lat = float(self._db_lat.get().strip())
            lon = float(self._db_lon.get().strip())
            alt = float(self._db_alt.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input","Lat / Lon / Alt must be numeric."); return
        db = self._get_faiss_db()
        if db is None: return
        self._log(f"Building FAISS DB: {Path(self._db_sat_path).name} ({lat:.4f}, {lon:.4f})", "info")
        self._sys_badge.set("BUILDING DB", "working")
        self._db_top_badge.config(text="  DB: BUILDING  ", fg="#f0a030",
                                  highlightbackground="#f0a030")
        self._db_progress.config(text="Building...  0 / ?", fg="#f0a030")
        def _progress(cur, total):
            pct = int(cur/total*100) if total else 0
            self.root.after(0, self._db_progress.config,
                            {"text": f"Building...  {cur}/{total}  ({pct}%)"})
        def _run():
            count = db.build(satellite_image_path=self._db_sat_path,
                             lat=lat, lon=lon, alt=alt,
                             progress_callback=_progress)
            self.root.after(0, self._faiss_build_done, count, db)
        threading.Thread(target=_run, daemon=True).start()

    def _faiss_build_done(self, count: int, db) -> None:
        stats = db.get_stats()
        self._db_progress.config(
            text=f"[+] {count} patches  |  {stats['index_size_mb']} MB",
            fg="#44cc44")
        self._db_status.config(
            text=(f"[+] {stats['total_descriptors']} patches"
                  f"  |  DINOv2 {stats['descriptor_dim']}-dim"
                  f"  |  {stats['unique_images']} image(s)"),
            fg="#44cc44")
        self._log(f"FAISS DB built: {count} patches ({stats['descriptor_dim']}-dim, {stats['index_size_mb']} MB)", "success")
        self._sys_badge.set("DB READY", "ok")
        self._db_top_badge.config(text=f"  DB: {count} patches  ",
                                  fg="#44cc44", highlightbackground="#44cc44")
        self._footer_stats.config(text=f"DB: {count} patches  |  Matches: 0")

    def _on_load_faiss_db(self):
        db = self._get_faiss_db()
        if db is None: return
        ok = db.load()
        if ok:
            stats = db.get_stats()
            self._db_status.config(
                text=(f"[+] {stats['total_descriptors']} patches"
                      f"  |  DINOv2 {stats['descriptor_dim']}-dim"
                      f"  |  {stats['unique_images']} image(s)"),
                fg="#44cc44")
            self._log(f"FAISS DB loaded: {stats['total_descriptors']} descriptors", "success")
            self._sys_badge.set("DB LOADED", "ok")
            self._db_top_badge.config(text=f"  DB: {stats['total_descriptors']} patches  ",
                                      fg="#44cc44", highlightbackground="#44cc44")
            self._footer_stats.config(
                text=f"DB: {stats['total_descriptors']} patches  |  Matches: 0")
        else:
            self._db_status.config(text="Load failed -- build the database first", fg="#f0a030")
            self._log("FAISS DB load failed", "warning")

    def _on_clear_faiss_db(self):
        if not messagebox.askyesno("Clear Database",
                "Delete the FAISS index and metadata from disk?\nThis cannot be undone."):
            return
        db = self._get_faiss_db()
        if db is None: return
        for f in (db.index_path, db.meta_path):
            if f.exists(): f.unlink()
        db.metadata = []; db._init_index()
        self._faiss_db = None
        self._db_status.config(text="Status: cleared", fg="#f0a030")
        self._db_progress.config(text="")
        self._log("FAISS database cleared", "warning")
        self._sys_badge.set("DB CLEARED", "working")
        self._db_top_badge.config(text="  DB: NOT LOADED  ",
                                  fg="#5a4a20", highlightbackground="#2a2200")

    def _on_match_select_uav(self):
        p = filedialog.askopenfilename(
            title="Select UAV Image", initialdir="data/patches/uav",
            filetypes=[("Images","*.jpg *.jpeg *.png *.tif *.tiff"),("All","*.*")])
        if not p: return
        self._uav_match_path = p
        self._match_img_lbl.config(text=f"[+] {Path(p).name}", fg="#44cc44")
        self._log(f"UAV image selected: {Path(p).name}", "info")
        self._vision_badge.config(text="  VISION: IMAGE LOADED  ",
                                  fg="#f0a030", highlightbackground="#f0a030")

    def _on_run_matching(self):
        if not self._uav_match_path:
            messagebox.showwarning("No Image","Select a UAV image first."); return
        db = self._get_faiss_db()
        if db is None: return
        if db.index is None or db.index.ntotal == 0:
            messagebox.showwarning("Empty Database","Build or load the database first."); return
        import cv2 as _cv2
        uav_img = _cv2.imread(self._uav_match_path)
        if uav_img is None:
            messagebox.showerror("Read Error","Cannot read UAV image."); return
        self._match_results.config(text="Running...", fg="#f0a030")
        self._sys_badge.set("MATCHING", "working")
        self._vision_badge.config(text="  VISION: MATCHING  ",
                                  fg="#f0a030", highlightbackground="#f0a030")
        self._log("Running DINOv2 + ORB matching...", "info")
        def _run():
            results = db.query(uav_img, top_k=5, min_confidence=0.0)
            self.root.after(0, self._match_done, results)
        threading.Thread(target=_run, daemon=True).start()

    def _match_done(self, results: list) -> None:
        n_total    = len(results)
        n_verified = sum(1 for r in results if r.get("orb_verified"))
        if not results:
            self._match_results.config(text="No matches found", fg="#ff4400")
            self._sys_badge.set("NO MATCH", "error")
            self._vision_badge.config(text="  VISION: NO LOCK  ",
                                      fg="#ff4400", highlightbackground="#ff4400")
            self._log("Matching: no results", "error"); return
        best  = results[0]
        lines = [
            f"Stage 1 (DINOv2): {n_total} candidates found",
            f"Stage 2 (ORB):    {n_verified} verified",
            f"Best Match: Lat {best['lat']:.6f},  Lon {best['lon']:.6f}",
            f"Confidence: {best['confidence']*100:.1f}%",
        ]
        self._match_results.config(text="\n".join(lines), fg="#44cc44")
        self._sys_badge.set("MATCHED", "ok")
        self._vision_badge.config(text="  VISION: LOCK  ",
                                  fg="#44cc44", highlightbackground="#44cc44")
        self._log(f"Match: lat={best['lat']:.5f} lon={best['lon']:.5f} conf={best['confidence']*100:.1f}%", "success")
        self._footer_stats.config(
            text=f"DB: {self._faiss_db.index.ntotal if self._faiss_db else 0} patches  |  Matches: {n_verified}")

    def _on_play_video(self):
        p = filedialog.askopenfilename(
            title="Select Demo Video",
            filetypes=[("Video","*.mp4 *.avi *.mkv *.mov"),("All","*.*")])
        if not p: return
        self._video_lbl.config(text=Path(p).name, fg="#f0a030")
        self._log(f"Opening video: {Path(p).name}", "info")
        try:
            os.startfile(p)
        except Exception:
            try:
                subprocess.Popen(["vlc", p])
            except Exception:
                subprocess.Popen(["explorer", p])

    def run(self):
        self._log("VNS ready  --  enter coordinates and open Google Earth Pro", "success")
        self.root.mainloop()


# =============================================================================

if __name__ == "__main__":
    app = VNSApp()
    app.run()

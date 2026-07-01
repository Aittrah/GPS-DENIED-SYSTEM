from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("vns.app.google_earth")

DEFAULT_GOOGLE_EARTH_CANDIDATES = (
    "google-earth-pro",
    "googleearth",
    "/usr/bin/google-earth-pro",
    "/usr/bin/googleearth",
    "/opt/google/earth/pro/googleearth",
)
WINDOWS_GOOGLE_EARTH_CANDIDATES = (
    r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
    r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe",
)
DEFAULT_FOCUS_KML_NAME = "google_earth_mission_focus.kml"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_KML_PATH = Path("data") / "kml" / DEFAULT_FOCUS_KML_NAME


@dataclass(frozen=True)
class GoogleEarthLaunchResult:
    opened: bool
    launch_method: str
    kml_path: Path
    attempted_commands: tuple[str, ...]
    error_reason: str
    searched_executables: tuple[str, ...]
    executable_path: Path | None = None


def _google_earth_candidates() -> tuple[str, ...]:
    candidates = list(DEFAULT_GOOGLE_EARTH_CANDIDATES)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidates.append(
                str(
                    Path(local_app_data)
                    / "Google"
                    / "Google Earth Pro"
                    / "client"
                    / "googleearth.exe"
                )
            )
        candidates.extend(WINDOWS_GOOGLE_EARTH_CANDIDATES)
    return tuple(candidates)


def _resolve_output_path(path_like: str | Path | None) -> Path:
    path = _DEFAULT_KML_PATH if path_like is None else Path(path_like)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _format_decimal(value: float) -> str:
    return f"{value:.8f}"


def _format_coordinates(lat: float, lon: float) -> str:
    return f"{_format_decimal(lon)},{_format_decimal(lat)},0"


def _placemark_xml(name: str, lat: float, lon: float) -> str:
    return f"""    <Placemark>
      <name>{name}</name>
      <Point>
        <coordinates>{_format_coordinates(lat, lon)}</coordinates>
      </Point>
    </Placemark>"""


def discover_google_earth_executables(
    candidates: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[str] = set()

    for candidate in tuple(candidates or _google_earth_candidates()):
        resolved_path = shutil.which(candidate)
        logger.info("Checking Google Earth candidate: %s", candidate)
        if not resolved_path:
            continue

        path = Path(resolved_path)
        normalized = str(path)
        if normalized in seen:
            continue

        seen.add(normalized)
        resolved.append(path)
        logger.info("Resolved Google Earth candidate %s -> %s", candidate, path)

    return tuple(resolved)


def resolve_google_earth_executable(
    candidates: Iterable[str] | None = None,
) -> Path | None:
    executables = discover_google_earth_executables(candidates)
    return executables[0] if executables else None


def create_google_earth_focus_kml(
    lat: float,
    lon: float,
    output_path: str | Path | None = None,
    *,
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    lookat_range_m: int | float = 800,
) -> Path:
    path = _resolve_output_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    placemarks = [
        _placemark_xml("Mission Center", lat, lon),
    ]
    if start is not None:
        placemarks.append(_placemark_xml("Start Point", start[0], start[1]))
    if goal is not None:
        placemarks.append(_placemark_xml("End Point", goal[0], goal[1]))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>VNS Mission Area</name>
    <LookAt>
      <longitude>{_format_decimal(lon)}</longitude>
      <latitude>{_format_decimal(lat)}</latitude>
      <altitude>0</altitude>
      <heading>0</heading>
      <tilt>0</tilt>
      <range>{int(lookat_range_m)}</range>
    </LookAt>
{os.linesep.join(placemarks)}
  </Document>
</kml>
"""

    path.write_text(kml, encoding="utf-8")
    logger.info("Wrote Google Earth focus KML to %s", path)
    return path


def _run_launch_command(command: list[str]) -> None:
    logger.info("Launching Google Earth with command: %s", shlex.join(command))
    subprocess.Popen(command)


def open_google_earth_at_location(
    lat: float,
    lon: float,
    *,
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    lookat_range_m: int | float = 800,
) -> GoogleEarthLaunchResult:
    kml_path = create_google_earth_focus_kml(
        lat,
        lon,
        output_path,
        start=start,
        goal=goal,
        lookat_range_m=lookat_range_m,
    )
    searched_executables = _google_earth_candidates()
    attempted_commands: list[str] = []
    last_error = "Google Earth launch did not start."

    for executable in discover_google_earth_executables(searched_executables):
        command = [str(executable), str(kml_path)]
        attempted_commands.append(shlex.join(command))
        try:
            _run_launch_command(command)
            logger.info("Google Earth launch succeeded via %s", executable)
            return GoogleEarthLaunchResult(
                opened=True,
                launch_method="google-earth",
                kml_path=kml_path,
                attempted_commands=tuple(attempted_commands),
                error_reason="",
                searched_executables=searched_executables,
                executable_path=executable,
            )
        except OSError as exc:
            last_error = f"{executable}: {exc}"
            logger.warning("Google Earth launch failed for %s: %s", executable, exc)

    xdg_open = shutil.which("xdg-open") or "xdg-open"
    xdg_command = [xdg_open, str(kml_path)]
    attempted_commands.append(shlex.join(xdg_command))
    try:
        _run_launch_command(xdg_command)
        logger.info("Google Earth launch succeeded via xdg-open")
        return GoogleEarthLaunchResult(
            opened=True,
            launch_method="xdg-open",
            kml_path=kml_path,
            attempted_commands=tuple(attempted_commands),
            error_reason="",
            searched_executables=searched_executables,
        )
    except OSError as exc:
        last_error = f"xdg-open: {exc}"
        logger.warning("xdg-open launch failed: %s", exc)

    if os.name == "nt" and hasattr(os, "startfile"):
        attempted_commands.append(f"os.startfile({kml_path})")
        try:
            logger.info("Launching Google Earth with os.startfile(%s)", kml_path)
            os.startfile(str(kml_path))
            logger.info("Google Earth launch succeeded via os.startfile")
            return GoogleEarthLaunchResult(
                opened=True,
                launch_method="os.startfile",
                kml_path=kml_path,
                attempted_commands=tuple(attempted_commands),
                error_reason="",
                searched_executables=searched_executables,
            )
        except OSError as exc:
            last_error = f"os.startfile: {exc}"
            logger.warning("os.startfile launch failed: %s", exc)

    logger.error("All Google Earth launch methods failed: %s", last_error)
    return GoogleEarthLaunchResult(
        opened=False,
        launch_method="manual",
        kml_path=kml_path,
        attempted_commands=tuple(attempted_commands),
        error_reason=last_error,
        searched_executables=searched_executables,
    )

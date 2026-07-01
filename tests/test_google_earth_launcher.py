from __future__ import annotations

from pathlib import Path

import vns.app.google_earth as google_earth
from vns.app.google_earth import (
    DEFAULT_FOCUS_KML_NAME,
    create_google_earth_focus_kml,
    open_google_earth_at_location,
    resolve_google_earth_executable,
)


def test_resolve_google_earth_executable_prefers_first_candidate(monkeypatch) -> None:
    calls: list[str] = []

    def fake_which(command: str) -> str | None:
        calls.append(command)
        if command == "google-earth-pro":
            return "/usr/bin/google-earth-pro"
        if command == "googleearth":
            return "/usr/bin/googleearth"
        return None

    monkeypatch.setattr(google_earth.shutil, "which", fake_which)

    executable = resolve_google_earth_executable()

    assert executable == Path("/usr/bin/google-earth-pro")
    assert calls[:2] == ["google-earth-pro", "googleearth"]


def test_create_google_earth_focus_kml_writes_lookat_and_lon_lat_order(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / DEFAULT_FOCUS_KML_NAME

    kml_path = create_google_earth_focus_kml(
        33.7510,
        73.1415,
        output_path,
        start=(33.7470, 73.1370),
        goal=(33.7550, 73.1460),
        lookat_range_m=2504,
    )

    content = kml_path.read_text(encoding="utf-8")

    assert kml_path == output_path
    assert "<LookAt>" in content
    assert "<name>Mission Center</name>" in content
    assert "<range>2504</range>" in content
    assert "<coordinates>73.14150000,33.75100000,0</coordinates>" in content
    assert "<coordinates>73.13700000,33.74700000,0</coordinates>" in content
    assert "<coordinates>73.14600000,33.75500000,0</coordinates>" in content
    assert "33.75100000,73.14150000,0" not in content


def test_open_google_earth_at_location_launches_resolved_executable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(google_earth, "_REPO_ROOT", tmp_path)

    def fake_which(command: str) -> str | None:
        if command == "google-earth-pro":
            return "/usr/bin/google-earth-pro"
        if command == "xdg-open":
            return "/usr/bin/xdg-open"
        return None

    def fake_popen(command: list[str]) -> object:
        commands.append(command)
        return object()

    monkeypatch.setattr(google_earth.shutil, "which", fake_which)
    monkeypatch.setattr(google_earth.subprocess, "Popen", fake_popen)

    result = open_google_earth_at_location(
        33.7510,
        73.1415,
        start=(33.7470, 73.1370),
        goal=(33.7550, 73.1460),
        lookat_range_m=2504,
    )

    expected_kml_path = tmp_path / "data" / "kml" / DEFAULT_FOCUS_KML_NAME

    assert result.opened is True
    assert result.launch_method == "google-earth"
    assert result.executable_path == Path("/usr/bin/google-earth-pro")
    assert result.kml_path == expected_kml_path
    assert commands == [["/usr/bin/google-earth-pro", str(expected_kml_path)]]
    assert expected_kml_path.exists()


def test_open_google_earth_at_location_falls_back_to_xdg_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(google_earth, "_REPO_ROOT", tmp_path)

    def fake_which(command: str) -> str | None:
        if command == "google-earth-pro":
            return "/usr/bin/google-earth-pro"
        if command == "xdg-open":
            return "/usr/bin/xdg-open"
        return None

    def fake_popen(command: list[str]) -> object:
        commands.append(command)
        if command[0] == "/usr/bin/google-earth-pro":
            raise OSError("permission denied")
        return object()

    monkeypatch.setattr(google_earth.shutil, "which", fake_which)
    monkeypatch.setattr(google_earth.subprocess, "Popen", fake_popen)

    result = open_google_earth_at_location(
        33.7510,
        73.1415,
        output_path=tmp_path / DEFAULT_FOCUS_KML_NAME,
    )

    assert result.opened is True
    assert result.launch_method == "xdg-open"
    assert commands[0][0] == "/usr/bin/google-earth-pro"
    assert commands[1][0] == "/usr/bin/xdg-open"
    assert any(
        command.startswith("/usr/bin/google-earth-pro ")
        for command in result.attempted_commands
    )
    assert any(
        command.startswith("/usr/bin/xdg-open ")
        for command in result.attempted_commands
    )


def test_open_google_earth_at_location_reports_failure_details(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(google_earth, "_REPO_ROOT", tmp_path)

    def fake_which(command: str) -> str | None:
        if command == "xdg-open":
            return "/usr/bin/xdg-open"
        return None

    def fake_popen(command: list[str]) -> object:
        raise OSError("launch failed")

    monkeypatch.setattr(google_earth.shutil, "which", fake_which)
    monkeypatch.setattr(google_earth.subprocess, "Popen", fake_popen)

    result = open_google_earth_at_location(
        33.7510,
        73.1415,
        output_path=tmp_path / DEFAULT_FOCUS_KML_NAME,
    )

    assert result.opened is False
    assert result.launch_method == "manual"
    assert result.error_reason == "xdg-open: launch failed"
    assert result.searched_executables == google_earth.DEFAULT_GOOGLE_EARTH_CANDIDATES
    assert result.attempted_commands[-1].startswith("/usr/bin/xdg-open ")

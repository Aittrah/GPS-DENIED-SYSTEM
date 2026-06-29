from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

_SIMULATION_CHILDREN = (
    "config",
    "database",
    "launch",
    "models",
    "scripts",
    "worlds",
)


def _is_simulation_root(path: Path) -> bool:
    return path.is_dir() and all((path / child).exists() for child in _SIMULATION_CHILDREN)


def resolve_from_base(path_like: str | Path, base_dir: str | Path | None) -> Path:
    """Resolve a path relative to a known base directory when needed."""
    path = Path(path_like)
    if path.is_absolute() or base_dir is None:
        return path
    return (Path(base_dir) / path).resolve()


def relativize_to_base(path_like: str | Path, base_dir: str | Path | None) -> str:
    """Store a path relative to a base directory when practical."""
    path = Path(path_like)
    if not path.is_absolute() or base_dir is None:
        return path.as_posix()

    try:
        return Path(os.path.relpath(path, Path(base_dir))).as_posix()
    except ValueError:
        # Windows can raise for differing drives; keep the absolute path intact.
        return path.as_posix()


def prepend_search_path(
    path_like: str | Path,
    env_name: str,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Prepend a directory to a PATH-like environment variable."""
    current_env = os.environ if env is None else env
    parts = [str(Path(path_like))]
    existing = current_env.get(env_name, "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def resolve_vns_python(
    simulation_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    fallback_executable: str | None = None,
) -> str:
    """Pick the Python interpreter for ROS launch / helper script execution."""
    current_env = os.environ if env is None else env
    env_python = current_env.get("VNS_PYTHON")
    if env_python:
        return env_python

    venv_python = Path(simulation_root).parent / ".venv-run" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)

    return fallback_executable or sys.executable


def resolve_simulation_root(anchor: str | Path | None = None) -> Path:
    """Resolve the simulation asset root in source-tree or installed layouts."""
    candidates: list[Path] = []

    if anchor is not None:
        start = Path(anchor).resolve()
        current = start if start.is_dir() else start.parent
        for path in (current, *current.parents):
            candidates.append(path)
            candidates.append(path / "simulation")

    prefix_candidate = Path(sys.prefix) / "share" / "vns" / "simulation"
    candidates.append(prefix_candidate)

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        pass
    else:
        candidates.append(Path(get_package_share_directory("vns")) / "simulation")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_simulation_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Unable to locate the VNS simulation asset root from the current environment."
    )

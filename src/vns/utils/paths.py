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


def _gazebo_version_key(path: Path) -> tuple[int, ...]:
    """Sort key so e.g. gazebo-11 ranks above gazebo-9 (and unversioned last)."""
    suffix = path.name.split("-", 1)[1] if "-" in path.name else ""
    try:
        return tuple(int(part) for part in suffix.split("."))
    except ValueError:
        return (-1,)


def detect_gazebo_share_dir(search_root: str | Path = "/usr/share") -> str | None:
    """Locate the Gazebo Classic share directory holding shaders/media.

    This directory (e.g. ``/usr/share/gazebo-11``) must be on
    ``GAZEBO_RESOURCE_PATH`` or the ``type='camera'`` sensor cannot find the
    RTShaderSystem shader libs and renders zero frames. Prefers a versioned
    dir that actually contains the shader programs, then falls back to any
    ``gazebo*`` dir that looks like a share tree, so a future Gazebo version is
    picked up without code changes. Returns ``None`` if nothing suitable exists.
    """
    candidates = sorted(
        (entry for entry in Path(search_root).glob("gazebo*") if entry.is_dir()),
        key=_gazebo_version_key,
        reverse=True,
    )
    # Prefer a dir that actually carries the shader programs the camera needs.
    for path in candidates:
        if (path / "media" / "materials" / "programs").is_dir():
            return str(path)
    # Fall back to anything that still looks like a gazebo share tree.
    for path in candidates:
        if (path / "media").is_dir() or (path / "setup.sh").exists():
            return str(path)
    return None


def compose_gazebo_resource_path(
    worlds_dir: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build ``GAZEBO_RESOURCE_PATH``: worlds dir, gazebo share dir, then existing.

    The worlds dir comes first so the world's custom materials win; the gazebo
    share dir supplies the shaders/media the camera sensor needs to render.
    This removes the need to manually ``source /usr/share/gazebo/setup.sh``
    before ``ros2 launch``.
    """
    current_env = os.environ if env is None else env
    parts = [str(Path(worlds_dir))]
    share_dir = detect_gazebo_share_dir()
    if share_dir:
        parts.append(share_dir)
    existing = current_env.get("GAZEBO_RESOURCE_PATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def resolve_px4_root(px4_root: str | Path | None = None) -> Path:
    """Return the PX4-Autopilot checkout root."""
    return Path(px4_root or Path.home() / "PX4-Autopilot")


def resolve_px4_gazebo_models_dir(px4_root: str | Path | None = None) -> Path | None:
    """Return PX4 Classic SITL models dir (contains ``gps``, ``iris``, etc.)."""
    root = resolve_px4_root(px4_root)
    models = root / "Tools/simulation/gazebo-classic/sitl_gazebo-classic/models"
    return models if models.is_dir() else None


def resolve_px4_gazebo_plugin_dir(px4_root: str | Path | None = None) -> Path | None:
    """Return PX4 Classic Gazebo plugin build dir (``libgazebo_mavlink_interface.so``)."""
    root = resolve_px4_root(px4_root)
    plugins = root / "build/px4_sitl_default/build_gazebo-classic"
    return plugins if plugins.is_dir() else None


def compose_gazebo_plugin_path(
    *,
    px4_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build ``GAZEBO_PLUGIN_PATH`` with PX4 Classic SITL plugins prepended."""
    current_env = os.environ if env is None else env
    parts: list[str] = []
    plugin_dir = resolve_px4_gazebo_plugin_dir(px4_root)
    if plugin_dir is not None:
        parts.append(str(plugin_dir))
    existing = current_env.get("GAZEBO_PLUGIN_PATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def compose_px4_ld_library_path(
    *,
    px4_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Prepend PX4 Classic plugin dir to ``LD_LIBRARY_PATH`` for gzserver."""
    current_env = os.environ if env is None else env
    parts: list[str] = []
    plugin_dir = resolve_px4_gazebo_plugin_dir(px4_root)
    if plugin_dir is not None:
        parts.append(str(plugin_dir))
    existing = current_env.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def compose_gazebo_model_path(
    models_dir: str | Path,
    *,
    extra_model_dirs: tuple[str | Path, ...] = (),
    env: Mapping[str, str] | None = None,
) -> str:
    """Build ``GAZEBO_MODEL_PATH``: project models, PX4/extras, gazebo base, existing.

    Appending ``<share>/models`` lets base models (e.g. ``sun``) resolve offline
    even with the online model database disabled.
    """
    current_env = os.environ if env is None else env
    parts = [str(Path(models_dir))]
    for extra in extra_model_dirs:
        extra_path = Path(extra)
        if extra_path.is_dir():
            parts.append(str(extra_path))
    share_dir = detect_gazebo_share_dir()
    if share_dir:
        base_models = Path(share_dir) / "models"
        if base_models.is_dir():
            parts.append(str(base_models))
    existing = current_env.get("GAZEBO_MODEL_PATH", "")
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
        # Best-effort: 'vns' may not be installed as a ROS package (e.g. running
        # from a source checkout with ROS sourced), in which case this raises
        # PackageNotFoundError — skip the ament candidate rather than abort.
        try:
            candidates.append(Path(get_package_share_directory("vns")) / "simulation")
        except Exception:
            pass

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

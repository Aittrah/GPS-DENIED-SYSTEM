from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_NUMPY_FIX = 'pip3 install "numpy<2"'


@dataclass(frozen=True)
class CvBridgeCheckResult:
    ok: bool
    bridge: Any | None
    message: str
    numpy_version: str


def _numpy_major(version: str) -> int:
    token = version.split(".", 1)[0].strip()
    return int(token) if token.isdigit() else 0


def check_cv_bridge_compatibility(
    *,
    numpy_version: str | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> CvBridgeCheckResult:
    """Validate the ROS 2 Humble Python environment before camera callbacks start."""
    detected_numpy = numpy_version or np.__version__

    if _numpy_major(detected_numpy) >= 2:
        message = (
            "ROS 2 Humble cv_bridge is not compatible with NumPy "
            f"{detected_numpy}. Install {_NUMPY_FIX} in the Python environment "
            "used to run vns_node, then rerun the launch."
        )
        logger.error(message)
        return CvBridgeCheckResult(
            ok=False,
            bridge=None,
            message=message,
            numpy_version=detected_numpy,
        )

    try:
        cv_bridge_module = import_module("cv_bridge")
    except ModuleNotFoundError as exc:
        message = (
            "cv_bridge is not installed in this Python environment. Source ROS 2 "
            "Humble and ensure the ROS Python packages are available before "
            "starting the VNS node."
        )
        logger.error("%s Original error: %s", message, exc)
        return CvBridgeCheckResult(
            ok=False,
            bridge=None,
            message=message,
            numpy_version=detected_numpy,
        )
    except Exception as exc:
        message = (
            f"cv_bridge import failed: {exc}. If the error mentions "
            f"'_ARRAY_API not found', install {_NUMPY_FIX}."
        )
        logger.error(message)
        return CvBridgeCheckResult(
            ok=False,
            bridge=None,
            message=message,
            numpy_version=detected_numpy,
        )

    try:
        bridge = cv_bridge_module.CvBridge()
    except Exception as exc:
        message = (
            f"CvBridge initialization failed: {exc}. If the error mentions "
            f"'_ARRAY_API not found', install {_NUMPY_FIX}."
        )
        logger.error(message)
        return CvBridgeCheckResult(
            ok=False,
            bridge=None,
            message=message,
            numpy_version=detected_numpy,
        )

    message = f"cv_bridge ready for ROS camera conversion (numpy={detected_numpy})."
    logger.info(message)
    return CvBridgeCheckResult(
        ok=True,
        bridge=bridge,
        message=message,
        numpy_version=detected_numpy,
    )

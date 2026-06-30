from __future__ import annotations

from typing import TYPE_CHECKING

from .blender import PositionBlender
from .diagnostics import SubsystemDiagnostic, VnsDiagnostics
from .gnss_monitor import GnssMonitor, GnssState

if TYPE_CHECKING:
    from .runtime import FrameProcessingResult, GroundTruthPose, VnsRuntime

__all__ = [
    "FrameProcessingResult",
    "GroundTruthPose",
    "PositionBlender",
    "SubsystemDiagnostic",
    "VnsDiagnostics",
    "VnsRuntime",
    "GnssMonitor",
    "GnssState",
]


def __getattr__(name: str):
    if name in {"FrameProcessingResult", "GroundTruthPose", "VnsRuntime"}:
        from .runtime import FrameProcessingResult, GroundTruthPose, VnsRuntime

        exported = {
            "FrameProcessingResult": FrameProcessingResult,
            "GroundTruthPose": GroundTruthPose,
            "VnsRuntime": VnsRuntime,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class SubsystemDiagnostic:
    """Health snapshot for one subsystem."""

    name: str
    healthy: bool
    state: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VnsDiagnostics:
    """Top-level runtime diagnostics snapshot."""

    timestamp: float = field(default_factory=_time.time)
    localizer_ready: bool = False
    database_entry_count: int = 0
    gnss_state: str = "DENIED"
    navigation_mode: str = "UNINITIALIZED"
    last_localization_reason: Optional[str] = None
    coverage_gap_detected: bool = False
    failsafe_active: bool = False
    subsystems: list[SubsystemDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/log friendly representation."""
        payload = asdict(self)
        payload["subsystems"] = [asdict(item) for item in self.subsystems]
        return payload

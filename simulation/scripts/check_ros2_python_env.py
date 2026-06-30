#!/usr/bin/env python3
"""Preflight the Python environment used by the ROS 2 VNS node."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from vns.utils.ros_env import check_cv_bridge_compatibility

    result = check_cv_bridge_compatibility()
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

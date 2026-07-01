#!/usr/bin/env python3
"""
Generate a static post-hoc visualization dashboard from VNS validation logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from vns.validation.posthoc_dashboard import build_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML dashboard from VNS logs and saved frames."
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to the JSONL evaluation log.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the image manifest JSONL produced by ImageFrameLogger.",
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Path to the .vnsdb database used during the run.",
    )
    parser.add_argument(
        "--output",
        default="./reports/dashboard",
        help="Output directory for index.html and overlay images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum number of log samples to render into the dashboard.",
    )
    args = parser.parse_args()

    result = build_dashboard(
        log_path=args.log,
        image_manifest_path=args.manifest,
        database_path=args.database,
        output_dir=args.output,
        max_samples=args.max_samples,
    )

    print(f"Dashboard written to: {result.html_path}")
    print(f"Overlay images generated: {result.overlay_count}")
    print(f"Samples skipped: {result.skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

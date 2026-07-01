#!/usr/bin/env python3
"""Validate reference coverage metadata, imagery index, and built database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from vns.validation.reference_coverage import format_report, validate_reference_coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate QAU reference coverage metadata and artifacts."
    )
    parser.add_argument(
        "--metadata",
        default="../database/qau_reference_metadata.yaml",
        help="Path to the reference metadata YAML file.",
    )
    parser.add_argument(
        "--index",
        help="Optional path to a database_index.yaml file to validate against metadata.",
    )
    parser.add_argument(
        "--database",
        help="Optional path to a built .vnsdb file to validate against metadata.",
    )
    parser.add_argument(
        "--simulation-config",
        default="../config/simulation.yaml",
        help="Path to simulation.yaml for camera footprint estimation.",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    metadata_path = (script_dir / args.metadata).resolve()
    index_path = None if args.index is None else (script_dir / args.index).resolve()
    database_path = None if args.database is None else (script_dir / args.database).resolve()
    simulation_config_path = (
        None
        if args.simulation_config is None
        else (script_dir / args.simulation_config).resolve()
    )

    report = validate_reference_coverage(
        metadata_path=metadata_path,
        index_path=index_path,
        database_path=database_path,
        simulation_config_path=simulation_config_path,
    )
    print(format_report(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

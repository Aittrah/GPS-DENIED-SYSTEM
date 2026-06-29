from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from vns.config.config_manager import ConfigError, ConfigManager
from vns.database.reference_db import LEGACY_FORMAT, ReferenceDatabase
from vns.utils.paths import resolve_simulation_root


def _script_path(name: str) -> Path:
    return resolve_simulation_root(Path(__file__).resolve()) / "scripts" / name


def _print_error(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _require_existing_file(path_like: str | Path, *, description: str) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{description} is not a file: {path}")
    return path


def _run_command(command: Sequence[str], *, description: str) -> None:
    try:
        subprocess.run(list(command), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{description} failed with exit code {exc.returncode}."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to start {description}: {exc}") from exc


def _build_database_command(
    args: argparse.Namespace,
    script_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--max-features",
        str(args.max_features),
        "--algorithm",
        str(args.algorithm),
    ]
    command.append("--no-vocab" if args.no_vocab else "--build-vocab")

    if args.vocab_size is not None:
        command.extend(["--vocab-size", str(args.vocab_size)])
    if args.metric is not None:
        command.extend(["--metric", str(args.metric)])
    if args.random_state is not None:
        command.extend(["--random-state", str(args.random_state)])
    if args.stats:
        command.append("--stats")

    return command


def _build_capture_command(
    args: argparse.Namespace,
    script_path: Path,
) -> list[str]:
    command = [sys.executable, str(script_path)]
    if args.synthetic:
        command.append("--synthetic")
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.output:
        command.extend(["--output", str(args.output)])
    return command


def _load_database_for_inspection(
    db_path: Path,
    *,
    trusted_legacy: bool,
) -> tuple[str, ReferenceDatabase]:
    db_format = ReferenceDatabase.detect_format(str(db_path))
    if db_format == LEGACY_FORMAT and not trusted_legacy:
        raise ValueError(
            "Legacy pickle database blocked by default. Re-run with "
            "`--trusted-legacy` to inspect a trusted legacy file or migrate "
            "it with `vns database migrate`."
        )

    loader = (
        ReferenceDatabase.load_legacy_trusted
        if db_format == LEGACY_FORMAT
        else ReferenceDatabase.load
    )
    return db_format, loader(str(db_path))


def _print_database_summary(
    db_path: Path,
    db_format: str,
    db: ReferenceDatabase,
) -> None:
    print(f"Format:     {db_format}")
    print("==================================================")
    print(f"DATABASE INSPECTION: {db_path.name}")
    print("==================================================")
    print(f"Name:       {db.name}")
    print(f"Version:    {db.version}")
    print(f"Created:    {db.created}")
    print(f"Algorithm:  {db.algorithm}")
    print(f"Entries:    {db.entry_count}")
    print(f"BoVW Index: {'yes' if db.vocabulary is not None else 'no'}")
    print("--------------------------------------------------")
    print("Geographic Bounds:")
    print(f"  Min Latitude:  {db.bounds.min_lat:.6f}")
    print(f"  Max Latitude:  {db.bounds.max_lat:.6f}")
    print(f"  Min Longitude: {db.bounds.min_lon:.6f}")
    print(f"  Max Longitude: {db.bounds.max_lon:.6f}")
    print("--------------------------------------------------")
    print("Database Entries:")
    print(
        f"{'ID':<25} | {'Latitude':<10} | {'Longitude':<10} | "
        f"{'Alt':<6} | {'Features':<8}"
    )
    print("-" * 70)

    total_features = 0
    for entry in db.entries.values():
        print(
            f"{entry.id:<25} | {entry.latitude:<10.6f} | "
            f"{entry.longitude:<10.6f} | {entry.altitude:<6.1f} | "
            f"{entry.feature_count:<8}"
        )
        total_features += entry.feature_count

    average_features = total_features / db.entry_count if db.entry_count else 0.0
    print("-" * 70)
    print(f"Total features:   {total_features}")
    print(f"Average features: {average_features:.1f}")
    print("==================================================")


def _validate_config(config_path: str | Path) -> int:
    print(f"Validating configuration: {config_path}")
    try:
        config = ConfigManager.load(config_path)
    except (ConfigError, FileNotFoundError, IsADirectoryError) as exc:
        return _print_error(str(exc))

    db_path = config.resolve_path(
        str(config.get("database.path", "../database/qau_campus.vnsdb"))
    )
    print(f"Loaded config. Database path is set to: {db_path}")
    print("Configuration validation completed successfully.")
    return 0


def db_build(args: argparse.Namespace) -> int:
    """Build a reference database using the canonical offline builder."""
    try:
        index_path = _require_existing_file(
            args.input,
            description="Input index file",
        )
        script_path = _require_existing_file(
            _script_path("build_reference_database.py"),
            description="Database builder script",
        )
        command = _build_database_command(args, script_path)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        return _print_error(str(exc))

    print(f"Building VNS reference database from index: {index_path}")
    try:
        _run_command(command, description="database build")
    except RuntimeError as exc:
        return _print_error(str(exc))
    return 0


def db_inspect(args: argparse.Namespace) -> int:
    """Inspect reference database file contents."""
    try:
        db_path = _require_existing_file(args.database, description="Database file")
        db_format, db = _load_database_for_inspection(
            db_path,
            trusted_legacy=bool(args.trusted_legacy),
        )
    except (
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        return _print_error(str(exc))

    _print_database_summary(db_path, db_format, db)
    return 0


def db_migrate(args: argparse.Namespace) -> int:
    """Migrate a trusted legacy pickle database to the safe archive format."""
    try:
        input_path = _require_existing_file(args.input, description="Database file")
        db_format = ReferenceDatabase.detect_format(str(input_path))
        if db_format != LEGACY_FORMAT:
            raise ValueError(
                f"Expected a legacy pickle database, found format '{db_format}'."
            )
        ReferenceDatabase.migrate_legacy_file(str(input_path), str(args.output))
    except (
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        return _print_error(str(exc))

    print(f"Migrated trusted legacy database to: {args.output}")
    return 0


def db_capture(args: argparse.Namespace) -> int:
    """Trigger image collection from simulation / camera stream."""
    try:
        script_path = _require_existing_file(
            _script_path("capture_reference_images.py"),
            description="Image capture script",
        )
        command = _build_capture_command(args, script_path)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        return _print_error(str(exc))

    print("Starting reference image collection...")
    print(f"Running image capture: {' '.join(command)}")
    try:
        _run_command(command, description="image capture")
    except RuntimeError as exc:
        return _print_error(str(exc))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual Navigation System (VNS) CLI Utility",
        prog="vns",
    )

    subparsers = parser.add_subparsers(dest="command", help="VNS command to run")

    database_parser = subparsers.add_parser("database", help="Database utilities")
    db_subparsers = database_parser.add_subparsers(
        dest="db_command",
        help="Database action",
    )

    parser_db_build = db_subparsers.add_parser(
        "build",
        help="Build reference database",
    )
    parser_db_build.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to image index file (yaml)",
    )
    parser_db_build.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output database file path (.vnsdb)",
    )
    parser_db_build.add_argument(
        "--max-features",
        type=int,
        default=500,
        help="Maximum ORB features per image",
    )
    parser_db_build.add_argument(
        "--algorithm",
        type=str,
        default="ORB",
        choices=["ORB"],
        help="Feature extraction algorithm",
    )
    parser_db_build.add_argument(
        "--no-vocab",
        action="store_true",
        help="Skip BoVW vocabulary build",
    )
    parser_db_build.add_argument(
        "--vocab-size",
        type=int,
        help="BoVW vocabulary size",
    )
    parser_db_build.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "l2"],
        help="BoVW nearest-neighbor metric",
    )
    parser_db_build.add_argument(
        "--random-state",
        type=int,
        help="BoVW MiniBatchKMeans random seed",
    )
    parser_db_build.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed BoVW statistics after build",
    )

    parser_db_inspect = db_subparsers.add_parser(
        "inspect",
        help="Inspect reference database",
    )
    parser_db_inspect.add_argument(
        "database",
        type=str,
        help="Path to reference database file (.vnsdb)",
    )
    parser_db_inspect.add_argument(
        "--trusted-legacy",
        action="store_true",
        help="Explicitly allow inspection of a trusted legacy pickle database",
    )

    parser_db_migrate = db_subparsers.add_parser(
        "migrate",
        help="Migrate a trusted legacy pickle database",
    )
    parser_db_migrate.add_argument(
        "--input",
        type=str,
        required=True,
        help="Legacy pickle database path",
    )
    parser_db_migrate.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output safe database path (.vnsdb)",
    )

    parser_db_capture = db_subparsers.add_parser(
        "capture",
        help="Capture reference images",
    )
    parser_db_capture.add_argument(
        "--synthetic",
        action="store_true",
        help="Capture synthetic images for offline testing",
    )
    parser_db_capture.add_argument(
        "--config",
        type=str,
        help="Path to reference metadata YAML",
    )
    parser_db_capture.add_argument(
        "--output",
        type=str,
        help="Output directory for captured images",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Validate and print configuration metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config and not args.command:
        return _validate_config(args.config)

    if args.command == "database":
        if args.db_command == "build":
            return db_build(args)
        if args.db_command == "inspect":
            return db_inspect(args)
        if args.db_command == "migrate":
            return db_migrate(args)
        if args.db_command == "capture":
            return db_capture(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

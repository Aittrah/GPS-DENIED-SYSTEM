import argparse
import subprocess
import sys
from pathlib import Path

from vns.database.reference_db import ReferenceDatabase


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "simulation" / "scripts" / name


def db_build(args):
    """Build a reference database using the canonical offline builder."""
    index_path = Path(args.input)
    if not index_path.exists():
        print(f"Error: Input index file not found at {index_path}")
        sys.exit(1)

    script_path = _script_path("build_reference_database.py")
    if not script_path.exists():
        print(f"Error: Database builder script not found at {script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(index_path),
        "--output",
        str(args.output),
        "--max-features",
        str(args.max_features),
        "--algorithm",
        args.algorithm,
    ]
    if args.no_vocab:
        cmd.append("--no-vocab")
    else:
        cmd.append("--build-vocab")
    if args.vocab_size is not None:
        cmd.extend(["--vocab-size", str(args.vocab_size)])
    if args.metric is not None:
        cmd.extend(["--metric", args.metric])
    if args.random_state is not None:
        cmd.extend(["--random-state", str(args.random_state)])
    if args.stats:
        cmd.append("--stats")

    print(f"Building VNS reference database from index: {index_path}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error building database: {e}")
        sys.exit(1)

def db_inspect(args):
    """Inspect reference database file contents."""
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)

    try:
        db_format = ReferenceDatabase.detect_format(str(db_path))
        print(f"Format:     {db_format}")
        if db_format == "legacy_pickle" and not args.trusted_legacy:
            print(
                "Legacy pickle database blocked by default. Re-run with "
                "`--trusted-legacy` to inspect a trusted legacy file or migrate "
                "it with `vns database migrate`."
            )
            sys.exit(1)

        if db_format == "legacy_pickle":
            db = ReferenceDatabase.load_legacy_trusted(str(db_path))
        else:
            db = ReferenceDatabase.load(str(db_path))

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
        print(f"{'ID':<25} | {'Latitude':<10} | {'Longitude':<10} | {'Alt':<6} | {'Features':<8}")
        print("-" * 70)
        
        total_feats = 0
        for entry in db.entries.values():
            print(f"{entry.id:<25} | {entry.latitude:<10.6f} | {entry.longitude:<10.6f} | {entry.altitude:<6.1f} | {entry.feature_count:<8}")
            total_feats += entry.feature_count
            
        avg_feats = total_feats / db.entry_count if db.entry_count > 0 else 0
        print("-" * 70)
        print(f"Total features:   {total_feats}")
        print(f"Average features: {avg_feats:.1f}")
        print("==================================================")
        
    except Exception as e:
        print(f"Error loading database: {e}")
        sys.exit(1)


def db_migrate(args):
    """Migrate a trusted legacy pickle database to the safe archive format."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Database file not found at {input_path}")
        sys.exit(1)

    try:
        db_format = ReferenceDatabase.detect_format(str(input_path))
        if db_format != "legacy_pickle":
            print(
                f"Error: Expected a legacy pickle database, found format "
                f"'{db_format}'."
            )
            sys.exit(1)

        ReferenceDatabase.migrate_legacy_file(str(input_path), args.output)
        print(f"Migrated trusted legacy database to: {args.output}")
    except Exception as e:
        print(f"Error migrating database: {e}")
        sys.exit(1)

def db_capture(args):
    """Trigger image collection from simulation / camera stream."""
    print("Starting reference image collection...")
    # Invoke capture_reference_images script logic
    try:
        script_path = _script_path("capture_reference_images.py")
        cmd = [sys.executable, str(script_path)]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.config:
            cmd.extend(["--config", args.config])
        if args.output:
            cmd.extend(["--output", args.output])
            
        print(f"Running image capture: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error capturing images: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Visual Navigation System (VNS) CLI Utility",
        prog="vns"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="VNS command to run")
    
    # database build sub-command
    parser_build = subparsers.add_parser("database", help="Database utilities")
    db_subparsers = parser_build.add_subparsers(dest="db_command", help="Database action")
    
    parser_db_build = db_subparsers.add_parser("build", help="Build reference database")
    parser_db_build.add_argument("--input", type=str, required=True, help="Path to image index file (yaml)")
    parser_db_build.add_argument("--output", type=str, required=True, help="Output database file path (.vnsdb)")
    parser_db_build.add_argument("--max-features", type=int, default=500, help="Maximum ORB features per image")
    parser_db_build.add_argument("--algorithm", type=str, default="ORB", choices=["ORB"], help="Feature extraction algorithm")
    parser_db_build.add_argument("--no-vocab", action="store_true", help="Skip BoVW vocabulary build")
    parser_db_build.add_argument("--vocab-size", type=int, help="BoVW vocabulary size")
    parser_db_build.add_argument("--metric", type=str, choices=["cosine", "l2"], help="BoVW nearest-neighbor metric")
    parser_db_build.add_argument("--random-state", type=int, help="BoVW MiniBatchKMeans random seed")
    parser_db_build.add_argument("--stats", action="store_true", help="Print detailed BoVW statistics after build")
    
    # database inspect sub-command
    parser_db_inspect = db_subparsers.add_parser("inspect", help="Inspect reference database")
    parser_db_inspect.add_argument("database", type=str, help="Path to reference database file (.vnsdb)")
    parser_db_inspect.add_argument(
        "--trusted-legacy",
        action="store_true",
        help="Explicitly allow inspection of a trusted legacy pickle database",
    )

    parser_db_migrate = db_subparsers.add_parser("migrate", help="Migrate a trusted legacy pickle database")
    parser_db_migrate.add_argument("--input", type=str, required=True, help="Legacy pickle database path")
    parser_db_migrate.add_argument("--output", type=str, required=True, help="Output safe database path (.vnsdb)")
    
    # database capture sub-command
    parser_db_capture = db_subparsers.add_parser("capture", help="Capture reference images")
    parser_db_capture.add_argument("--synthetic", action="store_true", help="Capture synthetic images for offline testing")
    parser_db_capture.add_argument("--config", type=str, help="Path to reference metadata YAML")
    parser_db_capture.add_argument("--output", type=str, help="Output directory for captured images")
    
    # configuration validation
    parser.add_argument("--config", type=str, help="Validate and print configuration metadata")
    
    args = parser.parse_args()
    
    if args.config and not args.command:
        print(f"Validating configuration: {args.config}")
        try:
            from vns.config.config_manager import ConfigManager
            config = ConfigManager.load(args.config)
            db_path = config.get("database.path", "../database/qau_campus.vnsdb")
            print(f"Loaded config. Database path is set to: {db_path}")
            print("Configuration validation completed successfully.")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
        sys.exit(0)
        
    if args.command == "database":
        if args.db_command == "build":
            db_build(args)
        elif args.db_command == "inspect":
            db_inspect(args)
        elif args.db_command == "migrate":
            db_migrate(args)
        elif args.db_command == "capture":
            db_capture(args)
        else:
            parser_build.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

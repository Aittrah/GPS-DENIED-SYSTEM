import argparse
import sys
from pathlib import Path
from vns.database.reference_db import ReferenceDatabase

def db_build(args):
    """Build a reference database from image index YAML."""
    index_path = Path(args.input)
    if not index_path.exists():
        print(f"Error: Input index file not found at {index_path}")
        sys.exit(1)
        
    print(f"Building VNS reference database from index: {index_path}")
    
    # Import and call database builder logic
    try:
        from vns.database.reference_db import ReferenceDatabase, DatabaseEntry
        import yaml
        
        with open(index_path, 'r') as f:
            index_data = yaml.safe_load(f) or {}
            
        images = index_data.get('images', [])
        if not images:
            print("Error: No images found in the index.")
            sys.exit(1)
            
        db = ReferenceDatabase(
            name=index_data.get('database', {}).get('name', 'Reference Database'),
            version=index_data.get('database', {}).get('version', '1.0.0'),
            algorithm="ORB"
        )
        
        for i, img in enumerate(images):
            print(f"  [{i+1}/{len(images)}] Indexing {img['id']}...", end="", flush=True)
            db.add_image(
                image_path=img['filepath'],
                latitude=img['latitude'],
                longitude=img['longitude'],
                altitude=img['altitude'],
                heading=img['heading'],
                image_id=img['id']
            )
            print(" Done.")
            
        db.save(args.output)
        print(f"\nSuccessfully built database with {db.entry_count} entries. Saved to: {args.output}")
        
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
        db = ReferenceDatabase.load(str(db_path))
        print("==================================================")
        print(f"DATABASE INSPECTION: {db_path.name}")
        print("==================================================")
        print(f"Name:       {db.name}")
        print(f"Version:    {db.version}")
        print(f"Created:    {db.created}")
        print(f"Algorithm:  {db.algorithm}")
        print(f"Entries:    {db.entry_count}")
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

def db_capture(args):
    """Trigger image collection from simulation / camera stream."""
    print("Starting reference image collection...")
    # Invoke capture_reference_images script logic
    try:
        from subprocess import run
        import os
        
        script_path = Path(__file__).parent.parent.parent / "simulation" / "scripts" / "capture_reference_images.py"
        cmd = [sys.executable, str(script_path)]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.config:
            cmd.extend(["--config", args.config])
        if args.output:
            cmd.extend(["--output", args.output])
            
        print(f"Running image capture: {' '.join(cmd)}")
        run(cmd, check=True)
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
    
    # database inspect sub-command
    parser_db_inspect = db_subparsers.add_parser("inspect", help="Inspect reference database")
    parser_db_inspect.add_argument("database", type=str, help="Path to reference database file (.vnsdb)")
    
    # database capture sub-command
    parser_db_capture = db_subparsers.add_parser("capture", help="Capture reference images")
    parser_db_capture.add_argument("--synthetic", action="store_true", help="Capture synthetic images for offline testing")
    parser_db_capture.add_argument("--config", type=str, help="Path to reference metadata YAML")
    parser_db_capture.add_argument("--output", type=str, help="Output directory for captured images")
    
    # standard vns execution
    parser.add_argument("--config", type=str, help="Run standalone VNS loop with config file")
    
    args = parser.parse_args()
    
    if args.config and not args.command:
        # Run standalone VNS loop (simulated environment)
        print(f"Initializing VNS standalone execution using configuration: {args.config}")
        # Connect MAVLink connectively
        try:
            from vns.config.config_manager import ConfigManager
            config = ConfigManager.load(args.config)
            db_path = config.get("database.path", "../database/qau_campus.vnsdb")
            print(f"Loaded config. Database path is set to: {db_path}")
            print("Successfully loaded standalone VNS pipeline configuration.")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
        sys.exit(0)
        
    if args.command == "database":
        if args.db_command == "build":
            db_build(args)
        elif args.db_command == "inspect":
            db_inspect(args)
        elif args.db_command == "capture":
            db_capture(args)
        else:
            parser_build.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

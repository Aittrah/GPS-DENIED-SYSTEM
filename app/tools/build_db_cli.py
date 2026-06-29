"""
Build the VNS reference database from a satellite image.

Usage:
    python3 tools/build_db_cli.py \\
        --image data/satellite/raw/qau.jpg \\
        --lat 33.7470 --lon 73.1370 --alt 550
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vns.database.reference_database import ReferenceDatabase


def main():
    parser = argparse.ArgumentParser(
        description="Build VNS SQLite reference database from a satellite image"
    )
    parser.add_argument("--image", required=True,
                        help="Path to satellite image (jpg/png/tif)")
    parser.add_argument("--lat", type=float, required=True,
                        help="Center latitude of image (WGS84 degrees)")
    parser.add_argument("--lon", type=float, required=True,
                        help="Center longitude of image (WGS84 degrees)")
    parser.add_argument("--alt", type=float, default=550.0,
                        help="Altitude in metres MSL (default: 550)")
    parser.add_argument("--db", default="data/database/reference.db",
                        help="Output database path")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    db = ReferenceDatabase(db_path=args.db)

    bar_width = 40
    start = time.time()

    def progress(current, total):
        frac = current / total if total else 0
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.time() - start
        print(f"\r  [{bar}] {current}/{total}  ({elapsed:.1f}s)", end="", flush=True)

    print(f"\nBuilding database from: {args.image}")
    print(f"  Center: lat={args.lat}  lon={args.lon}  alt={args.alt}m")
    print(f"  Output: {args.db}\n")

    count = db.build(
        satellite_image_path=args.image,
        lat=args.lat,
        lon=args.lon,
        alt=args.alt,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print(f"\n\nDone — {count} patches stored in {elapsed:.1f}s\n")

    stats = db.get_stats()
    print("Database stats:")
    print(f"  Total patches : {stats['total_patches']}")
    print(f"  Unique images : {stats['unique_images']}")
    print(f"  Lat range     : {stats['lat_range']}")
    print(f"  Lon range     : {stats['lon_range']}")
    print(f"  Last built    : {stats['last_built']}")
    print(f"  DB size       : {stats['db_size_mb']} MB")


if __name__ == "__main__":
    main()

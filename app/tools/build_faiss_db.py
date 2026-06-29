"""
Build the VNS FAISS reference database from a satellite image.

Usage:
    python tools/build_faiss_db.py \\
        --image data/satellite/raw/qau.jpg \\
        --lat 33.7470 --lon 73.1370 --alt 550
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vns.database.faiss_database import FAISSDatabase


def main():
    parser = argparse.ArgumentParser(
        description="Build VNS FAISS reference database (DINOv2 + ORB)"
    )
    parser.add_argument("--image", required=True,
                        help="Path to satellite image (jpg/png/tif)")
    parser.add_argument("--lat", type=float, required=True,
                        help="Centre latitude of image (WGS84 degrees)")
    parser.add_argument("--lon", type=float, required=True,
                        help="Centre longitude of image (WGS84 degrees)")
    parser.add_argument("--alt", type=float, default=550.0,
                        help="Altitude in metres MSL (default: 550)")
    parser.add_argument("--index", default="data/database/faiss.index",
                        help="Output FAISS index path")
    parser.add_argument("--meta", default="data/database/metadata.pkl",
                        help="Output metadata pickle path")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    db = FAISSDatabase(index_path=args.index, meta_path=args.meta)

    if not db.extractor.available:
        print("[ERROR] DINOv2 model could not be loaded.")
        print("  Install: pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    patch_names: list[str] = []
    start = time.time()

    def progress(current: int, total: int) -> None:
        if current == 0:
            return
        patch_id = db.metadata[current - 1]["patch_id"] if db.metadata else f"p{current-1}"
        row = (current - 1) // max(1, (total ** 0.5))
        col = (current - 1) % max(1, (total ** 0.5))
        tag = f"r{int(row)}_c{int(col)}"
        print(f"  [{current}/{total}] Extracting patch {tag}... done")

    print(f"\nBuilding FAISS database from: {args.image}")
    print(f"  Centre: lat={args.lat}  lon={args.lon}  alt={args.alt}m")
    print(f"  Index : {args.index}")
    print(f"  Meta  : {args.meta}\n")

    count = db.build(
        satellite_image_path=args.image,
        lat=args.lat,
        lon=args.lon,
        alt=args.alt,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print()

    stats = db.get_stats()
    print(f"  Built: {count} descriptors (DINOv2 {stats['descriptor_dim']}-dim)")
    print(f"  Index size: {stats['index_size_mb']} MB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved to {Path(args.index).parent}/\n")


if __name__ == "__main__":
    main()

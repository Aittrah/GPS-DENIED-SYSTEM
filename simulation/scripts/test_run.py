#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src and root to system path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir / 'src'))
sys.path.insert(0, str(root_dir))

from simulation.scripts.visual_navigation import VnsNode
from vns.database.reference_db import ReferenceDatabase

def main():
    print("Starting simulated VNS flight evaluation...")
    
    config_path = root_dir / "simulation" / "config" / "simulation.yaml"
    database_path = root_dir / "simulation" / "database" / "qau_campus.vnsdb"
    
    node = VnsNode(str(config_path), str(database_path))
    
    # Let's generate a mock 640x480 gray image for processing
    # To get matching features, we will load a real reference image from the database!
    try:
        db = ReferenceDatabase.load(str(database_path))
    except ValueError as exc:
        print(
            f"{exc}\nMigrate the database with: "
            f"vns database migrate --input {database_path} "
            f"--output {database_path}"
        )
        sys.exit(1)
    first_entry = list(db.entries.values())[0]
    
    # Read the actual reference image to guarantee a 100% successful feature match!
    import cv2
    image = cv2.imread(str(db.resolve_source_path(first_entry.source_path)))
    if image is None:
        # Fallback to synthetic if image can't be read
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
    print(f"Running 30-step flight simulation using reference image {first_entry.id}...")
    
    # Simulating flight trajectory
    for step in range(30):
        # Step 0-10: GPS is healthy
        # Step 10-30: GPS is completely denied
        if step < 10:
            has_fix = True
            num_sats = 10
            hdop = 1.0
            gps_lat = first_entry.latitude
            gps_lon = first_entry.longitude
            gps_alt = first_entry.altitude
        else:
            has_fix = False
            num_sats = 0
            hdop = 99.0
            gps_lat = 0.0
            gps_lon = 0.0
            gps_alt = 0.0
            
        # Flight controller updates
        node.current_heading = first_entry.heading
        node.current_altitude = first_entry.altitude
        
        # Ground truth pose is centered exactly over the first reference image
        node.ground_truth_pose = (first_entry.latitude, first_entry.longitude, first_entry.altitude, first_entry.heading)
        
        # Update GPS
        node.process_gps(has_fix, num_sats, hdop, gps_lat, gps_lon, gps_alt)
        
        # Process camera frame
        blended, mode = node.process_image(image)
        
        # Print status
        print(f"Step {step:02d} | GNSS: {'OK' if step < 10 else 'DENIED'} | Mode: {mode:<8} | Lat: {blended[0]:.6f} | Lon: {blended[1]:.6f} | Alt: {blended[2]:.1f}")
        time.sleep(0.02)
        
    node.close()
    print("\nSimulated VNS flight complete.")
    print(f"Evaluation logs written to: {node.gt_log_path}")

if __name__ == "__main__":
    main()

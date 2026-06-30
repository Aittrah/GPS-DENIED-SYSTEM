"""
Upload planned waypoints to PX4 via MAVSDK as a MAVLink mission.

Usage:
    python3 scripts/upload_mission.py [--csv data/waypoints/qau_path_waypoints.csv]
                                      [--altitude 30.0]
                                      [--speed 5.0]
                                      [--connection udp://:14540]
"""
import asyncio
import csv
import argparse
from pathlib import Path

CSV_DEFAULT = Path(__file__).parent.parent / "data/waypoints/qau_path_waypoints.csv"


def load_waypoints(csv_path: str) -> list[dict]:
    waypoints = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            waypoints.append({
                "index": int(row["index"]),
                "lat":   float(row["latitude"]),
                "lon":   float(row["longitude"]),
            })
    return waypoints


async def upload(csv_path: str, altitude: float, speed: float, connection: str):
    from mavsdk import System
    from mavsdk.mission import MissionItem, MissionPlan

    waypoints = load_waypoints(csv_path)
    print(f"Loaded {len(waypoints)} waypoints from {csv_path}")

    drone = System()
    print(f"Connecting to PX4 at {connection} ...")
    await drone.connect(system_address=connection)

    # Wait for connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to PX4.")
            break

    # Build mission items
    mission_items = []
    for wp in waypoints:
        mission_items.append(
            MissionItem(
                latitude_deg=wp["lat"],
                longitude_deg=wp["lon"],
                relative_altitude_m=altitude,
                speed_m_s=speed,
                is_fly_through=True,
                gimbal_pitch_deg=float("nan"),
                gimbal_yaw_deg=float("nan"),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=float("nan"),
                camera_photo_interval_s=float("nan"),
                acceptance_radius_m=2.0,
                yaw_deg=float("nan"),
                camera_photo_distance_m=float("nan"),
                vehicle_action=MissionItem.VehicleAction.NONE,
            )
        )

    mission_plan = MissionPlan(mission_items)

    print(f"Uploading {len(mission_items)} mission items at {altitude}m / {speed}m/s ...")
    await drone.mission.upload_mission(mission_plan)
    print("Mission uploaded successfully.")

    # Set return-to-launch after mission
    await drone.mission.set_return_to_launch_after_mission(True)
    print("RTL after mission: enabled.")

    # Arm and start mission
    print("Arming drone ...")
    await drone.action.arm()

    print("Starting mission ...")
    await drone.mission.start_mission()
    print("Mission started. Monitoring progress...")

    async for progress in drone.mission.mission_progress():
        print(f"  Waypoint {progress.current}/{progress.total}")
        if progress.current == progress.total:
            print("Mission complete.")
            break


def main():
    parser = argparse.ArgumentParser(description="Upload VNS waypoints to PX4 via MAVSDK")
    parser.add_argument("--csv",        default=str(CSV_DEFAULT), help="Waypoints CSV file")
    parser.add_argument("--altitude",   type=float, default=30.0,            help="Flight altitude in metres (default: 30)")
    parser.add_argument("--speed",      type=float, default=5.0,             help="Flight speed m/s (default: 5)")
    parser.add_argument("--connection", default="udp://:14540",              help="MAVSDK connection string (default: udp://:14540)")
    args = parser.parse_args()

    asyncio.run(upload(args.csv, args.altitude, args.speed, args.connection))


if __name__ == "__main__":
    main()

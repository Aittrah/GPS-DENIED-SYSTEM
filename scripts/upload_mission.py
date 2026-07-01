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


async def _wait_for_health(drone, timeout_s: float = 90.0) -> None:
    """Wait until PX4 reports armable before attempting to arm."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    async for health in drone.telemetry.health():
        if (
            health.is_global_position_ok
            and health.is_home_position_ok
            and health.is_local_position_ok
            and health.is_armable
        ):
            print("PX4 health OK (local/global/home position + armable).")
            return
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                "PX4 did not become armable within "
                f"{timeout_s:.0f}s "
                f"(global={health.is_global_position_ok}, "
                f"local={health.is_local_position_ok}, "
                f"home={health.is_home_position_ok}, "
                f"armable={health.is_armable})."
            )
        await asyncio.sleep(0.5)


async def _arm_with_retries(drone, attempts: int = 8, delay_s: float = 2.0) -> None:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            await drone.action.arm()
            print("Drone armed.")
            return
        except Exception as exc:  # noqa: BLE001 - surface MAVSDK ActionError text
            last_error = exc
            print(f"Arm attempt {attempt}/{attempts} denied: {exc}")
            if attempt < attempts:
                await asyncio.sleep(delay_s)
    raise RuntimeError(f"Failed to arm after {attempts} attempts: {last_error}") from last_error


async def upload(
    csv_path: str,
    altitude: float,
    speed: float,
    connection: str,
    *,
    monitor: bool = True,
):
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

    print("Waiting for PX4 preflight health ...")
    await _wait_for_health(drone)

    # Arm and start mission
    print("Arming drone ...")
    await _arm_with_retries(drone)

    print("Starting mission ...")
    await drone.mission.start_mission()
    print("Mission started.")
    if not monitor:
        print("Monitor disabled; mission running in PX4.")
        return

    print("Monitoring progress...")
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
    parser.add_argument("--no-monitor", action="store_true", help="Upload/arm/start only; do not wait for mission completion")
    args = parser.parse_args()

    asyncio.run(
        upload(
            args.csv,
            args.altitude,
            args.speed,
            args.connection,
            monitor=not args.no_monitor,
        )
    )


if __name__ == "__main__":
    main()

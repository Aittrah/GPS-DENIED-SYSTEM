import pytest

from vns.core.blender import PositionBlender
from vns.core.dead_reckoning import DeadReckoning
from vns.core.gnss_monitor import GnssState
from vns.core.models import IMUSample, NEDPoint, UAVState
from vns.utils.coordinates import enu_to_geodetic


GRAVITY_M_S2 = 9.80665
ORIGIN = {
    "origin_latitude": 33.7470,
    "origin_longitude": 73.1370,
    "origin_altitude": 550.0,
}


def _geodetic_from_ned(pose_ned: tuple[float, float, float]) -> tuple[float, float, float]:
    lat, lon, alt = enu_to_geodetic(
        pose_ned[1],
        pose_ned[0],
        -pose_ned[2],
        ORIGIN["origin_latitude"],
        ORIGIN["origin_longitude"],
        ORIGIN["origin_altitude"],
    )
    return lat, lon, alt


def _sample(
    *,
    accel_x: float = 0.0,
    accel_y: float = 0.0,
    accel_z: float = -GRAVITY_M_S2,
    gyro_x: float = 0.0,
    gyro_y: float = 0.0,
    gyro_z: float = 0.0,
    timestamp: float = 0.0,
) -> IMUSample:
    return IMUSample(
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        timestamp=timestamp,
    )


def test_visual_fix_resets_drift_and_following_dr_estimates_continue_from_reset() -> None:
    drifted_state = UAVState(
        position=NEDPoint(north=55.0, east=10.0, down=-30.0),
        velocity_north=0.0,
        velocity_east=0.0,
        velocity_down=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        confidence=0.4,
        source="fused",
        timestamp=5.0,
    )
    dead_reckoning = DeadReckoning(initial_state=drifted_state)
    blender = PositionBlender(position_continuity_threshold=15.0)

    visual_pose_ned = (45.0, 10.0, -30.0)
    visual_pose_geo = _geodetic_from_ned(visual_pose_ned)

    fused_pose, mode = blender.blend(
        None,
        visual_pose_geo,
        GnssState.DENIED,
        current_time=10.0,
        dead_reckoning=dead_reckoning,
        dead_reckoning_estimate=dead_reckoning.get_estimate(),
        dead_reckoning_ready=False,
        visual_pose_ned=visual_pose_ned,
        visual_confidence=0.92,
        geo_origin=ORIGIN,
    )

    assert mode == "VISION"
    assert fused_pose == pytest.approx(visual_pose_geo)

    corrected = dead_reckoning.get_estimate()
    assert corrected.position.north == pytest.approx(45.0)
    assert corrected.position.east == pytest.approx(10.0)
    assert corrected.position.down == pytest.approx(-30.0)

    dead_reckoning.update(
        _sample(accel_x=1.0, timestamp=11.0),
        dt=1.0,
    )
    continued = dead_reckoning.get_estimate()
    next_pose, next_mode = blender.blend(
        None,
        None,
        GnssState.DENIED,
        current_time=11.0,
        dead_reckoning=dead_reckoning,
        dead_reckoning_estimate=continued,
        dead_reckoning_ready=True,
        geo_origin=ORIGIN,
    )

    assert next_mode == "DR"
    assert continued.position.north == pytest.approx(45.5, abs=1e-6)
    assert next_pose == pytest.approx(_geodetic_from_ned((45.5, 10.0, -30.0)))


def test_dr_only_mode_continues_without_visual_reset() -> None:
    seeded_state = UAVState(
        position=NEDPoint(north=20.0, east=-5.0, down=-30.0),
        velocity_north=0.0,
        velocity_east=0.0,
        velocity_down=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        confidence=0.5,
        source="fused",
        timestamp=0.0,
    )
    dead_reckoning = DeadReckoning(initial_state=seeded_state)
    blender = PositionBlender()

    for step in range(1, 4):
        dead_reckoning.update(
            _sample(accel_x=0.01, timestamp=float(step)),
            dt=1.0,
        )
        estimate = dead_reckoning.get_estimate()
        pose, mode = blender.blend(
            None,
            None,
            GnssState.DENIED,
            current_time=float(step),
            dead_reckoning=dead_reckoning,
            dead_reckoning_estimate=estimate,
            dead_reckoning_ready=True,
            geo_origin=ORIGIN,
        )

        assert mode == "DR"
        assert pose == pytest.approx(
            _geodetic_from_ned(
                (
                    estimate.position.north,
                    estimate.position.east,
                    estimate.position.down,
                )
            )
        )

    final_estimate = dead_reckoning.get_estimate()
    assert final_estimate.position.north > 20.0
    assert blender.failsafe_active is False


def test_gps_healthy_fast_path_preserves_output_contract_and_skips_visual_override() -> None:
    gps_pose = (33.7471, 73.1372, 580.0)
    visual_pose_ned = (100.0, 50.0, -30.0)
    visual_pose_geo = _geodetic_from_ned(visual_pose_ned)
    initial_state = UAVState(
        position=NEDPoint(north=90.0, east=40.0, down=-30.0),
        velocity_north=2.0,
        velocity_east=0.5,
        velocity_down=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.1,
        confidence=0.7,
        source="fused",
        timestamp=2.0,
    )
    dead_reckoning = DeadReckoning(initial_state=initial_state)
    blender = PositionBlender(position_continuity_threshold=250.0)

    before = dead_reckoning.get_estimate()
    fused_pose, mode = blender.blend(
        gps_pose,
        visual_pose_geo,
        GnssState.HEALTHY,
        current_time=20.0,
        dead_reckoning=dead_reckoning,
        dead_reckoning_estimate=before,
        dead_reckoning_ready=True,
        visual_pose_ned=visual_pose_ned,
        visual_confidence=0.95,
        geo_origin=ORIGIN,
    )
    after = dead_reckoning.get_estimate()

    assert mode == "GPS"
    assert fused_pose == gps_pose
    assert isinstance(fused_pose, tuple)
    assert len(fused_pose) == 3
    assert all(isinstance(component, float) for component in fused_pose)
    assert after.position.north == pytest.approx(before.position.north)
    assert after.position.east == pytest.approx(before.position.east)
    assert after.position.down == pytest.approx(before.position.down)


def test_denied_mode_does_not_use_dr_when_bridge_is_not_ready() -> None:
    seeded_state = UAVState(
        position=NEDPoint(north=5.0, east=0.0, down=-30.0),
        velocity_north=0.0,
        velocity_east=0.0,
        velocity_down=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        confidence=1.0,
        source="fused",
        timestamp=0.0,
    )
    dead_reckoning = DeadReckoning(
        initial_state=seeded_state,
        max_bridge_duration_seconds=2.0,
        confidence_decay_per_second=0.0,
    )
    blender = PositionBlender()

    blender.blend(
        (33.7470, 73.1370, 580.0),
        None,
        GnssState.HEALTHY,
        current_time=0.0,
        dead_reckoning=dead_reckoning,
        dead_reckoning_estimate=dead_reckoning.get_estimate(current_time=0.0),
        dead_reckoning_ready=True,
        geo_origin=ORIGIN,
    )

    estimate = dead_reckoning.get_estimate(current_time=3.0)
    pose, mode = blender.blend(
        None,
        None,
        GnssState.DENIED,
        current_time=3.0,
        dead_reckoning=dead_reckoning,
        dead_reckoning_estimate=estimate,
        dead_reckoning_ready=False,
        geo_origin=ORIGIN,
    )

    assert mode == "FAILSAFE"
    assert pose == pytest.approx((33.7470, 73.1370, 580.0))

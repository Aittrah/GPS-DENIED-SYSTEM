import numpy as np

from vns.core.runtime import VnsRuntime


def test_runtime_reports_database_unavailable(sample_config, textured_image) -> None:
    runtime = VnsRuntime(sample_config, database=None)

    result = runtime.process_frame(textured_image, timestamp=100.0)
    diagnostics = runtime.get_diagnostics()

    assert not result.success
    assert result.reason == "database_unavailable"
    assert diagnostics.localizer_ready is False
    assert diagnostics.navigation_mode == "FAILSAFE"
    assert diagnostics.subsystems[0].state == "MISSING"


def test_runtime_detects_coverage_gap(sample_config, sample_database) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=34.5,
        longitude=74.5,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )

    result = runtime.process_frame(np.zeros((480, 640), dtype=np.uint8), timestamp=101.0)
    diagnostics = runtime.get_diagnostics()

    assert not result.success
    assert result.reason == "coverage_gap"
    assert diagnostics.coverage_gap_detected is True
    assert diagnostics.last_localization_reason == "coverage_gap"
    assert diagnostics.subsystems[2].details["coverage_gap_detected"] is True


def test_runtime_successful_localization_updates_diagnostics(
    sample_config,
    sample_database,
    textured_image,
) -> None:
    runtime = VnsRuntime(sample_config, database=sample_database)
    runtime.update_gps(
        latitude=33.7470,
        longitude=73.1370,
        altitude=580.0,
        has_fix=True,
        num_satellites=10,
        hdop=1.0,
        timestamp=100.0,
    )
    runtime.update_heading_from_quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    result = runtime.process_frame(textured_image, timestamp=101.0)
    diagnostics = runtime.get_diagnostics()

    assert result.success
    assert result.localization.reason == "ok"
    assert result.blended_pose is not None
    assert diagnostics.localizer_ready is True
    assert diagnostics.database_entry_count == sample_database.entry_count
    assert diagnostics.coverage_gap_detected is False
    assert diagnostics.last_localization_reason == "ok"

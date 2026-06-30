from types import SimpleNamespace

from vns.utils.ros_env import check_cv_bridge_compatibility


def test_cv_bridge_check_rejects_numpy_2() -> None:
    result = check_cv_bridge_compatibility(
        numpy_version="2.1.0",
        import_module=lambda _: SimpleNamespace(CvBridge=object),
    )

    assert result.ok is False
    assert 'numpy<2' in result.message


def test_cv_bridge_check_accepts_compatible_environment() -> None:
    class FakeBridge:
        pass

    result = check_cv_bridge_compatibility(
        numpy_version="1.26.4",
        import_module=lambda _: SimpleNamespace(CvBridge=FakeBridge),
    )

    assert result.ok is True
    assert isinstance(result.bridge, FakeBridge)


def test_cv_bridge_check_reports_missing_module() -> None:
    def _missing(_: str):
        raise ModuleNotFoundError("No module named 'cv_bridge'")

    result = check_cv_bridge_compatibility(
        numpy_version="1.26.4",
        import_module=_missing,
    )

    assert result.ok is False
    assert "cv_bridge is not installed" in result.message


def test_cv_bridge_check_reports_import_failure() -> None:
    def _broken(_: str):
        raise ImportError("_ARRAY_API not found")

    result = check_cv_bridge_compatibility(
        numpy_version="1.26.4",
        import_module=_broken,
    )

    assert result.ok is False
    assert "_ARRAY_API not found" in result.message

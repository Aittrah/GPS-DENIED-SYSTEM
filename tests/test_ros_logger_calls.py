"""Guards against RcutilsLogger.info/warning/error/debug() being called with
extra positional (printf-style) arguments, which raises:

    TypeError: RcutilsLogger.info() takes 2 positional arguments but 3 were given

ROS 2 loggers only accept a single preformatted message string.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [REPO_ROOT / "src" / "vns", REPO_ROOT / "simulation" / "scripts"]
LOGGER_METHODS = {"info", "warning", "warn", "error", "debug", "fatal"}


def _is_get_logger_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_logger"
    )


def _find_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LOGGER_METHODS:
            continue
        if not _is_get_logger_call(func.value):
            continue
        if len(node.args) > 1:
            violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return violations


def _python_files() -> list[Path]:
    files = []
    for directory in SCAN_DIRS:
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    return files


def test_no_multi_arg_ros_logger_calls() -> None:
    violations: list[str] = []
    for path in _python_files():
        violations.extend(_find_violations(path))

    assert not violations, (
        "RcutilsLogger calls must use a single preformatted message string, "
        f"not printf-style args: {violations}"
    )

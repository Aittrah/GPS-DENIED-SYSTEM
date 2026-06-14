from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "vns"
ROOT = Path(__file__).parent


def package_data_files(*directories: str):
    data_files = []
    for directory in directories:
        source_dir = ROOT / directory
        if not source_dir.exists():
            continue
        for path in source_dir.rglob("*"):
            if path.is_file():
                target = Path("share") / PACKAGE_NAME / path.parent.relative_to(ROOT)
                data_files.append((str(target), [str(path)]))
    return data_files

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    description="GNSS-Denied Visual Navigation System (VNS) for autonomous drones",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [str(ROOT / "resource" / PACKAGE_NAME)],
        ),
        (str(Path("share") / PACKAGE_NAME), [str(ROOT / "package.xml")]),
    ] + package_data_files(
        "simulation/launch",
        "simulation/config",
        "simulation/worlds",
        "simulation/models",
        "simulation/scripts",
    ),
    install_requires=[
        "numpy",
        "opencv-python",
        "PyYAML",
        "requests",
        "scikit-learn",
        "mavsdk",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ]
    },
    entry_points={
        "console_scripts": [
            "vns=vns.cli:main",
            "vns_node=vns.core.vns_node:main",
        ]
    },
)

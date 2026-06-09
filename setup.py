from setuptools import setup, find_packages

setup(
    name="vns",
    version="1.0.0",
    description="GNSS-Denied Visual Navigation System (VNS) for autonomous drones",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-python",
        "PyYAML",
        "requests",
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

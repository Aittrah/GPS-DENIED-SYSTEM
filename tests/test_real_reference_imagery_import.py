from __future__ import annotations

import runpy
from pathlib import Path

import cv2
import numpy as np
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "simulation"
    / "scripts"
    / "import_real_reference_imagery.py"
)


def _script_functions() -> dict:
    return runpy.run_path(str(SCRIPT_PATH))


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_import_manifest_to_index_computes_metadata_and_dimensions(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_image(image_root / "real_01.jpg", 1)
    _write_image(image_root / "real_02.jpg", 2)

    manifest_path = tmp_path / "real_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "database:",
                "  name: Imported QAU Set",
                "  version: 2.0.0",
                "images:",
                "  - id: real_01",
                "    filename: real_01.jpg",
                "    latitude: 33.7470",
                "    longitude: 73.1370",
                "    altitude: 550.0",
                "    heading: 0.0",
                "  - id: real_02",
                "    filename: real_02.jpg",
                "    latitude: 33.7480",
                "    longitude: 73.1380",
                "    altitude: 550.0",
                "    heading: 90.0",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "database_index.yaml"
    payload = _script_functions()["import_manifest_to_index"](
        manifest_path=manifest_path,
        output_path=output_path,
        image_root=image_root,
    )

    assert output_path.exists()
    assert payload["database"]["image_count"] == 2
    assert payload["database"]["bounds"]["min_latitude"] == 33.747
    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert loaded["images"][0]["width"] == 160
    assert loaded["images"][0]["height"] == 120
    assert loaded["images"][0]["filepath"].endswith("real_01.jpg")


def test_import_manifest_can_stage_images_into_destination(tmp_path: Path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    _write_image(image_root / "campus_a.png", 3)

    manifest_path = tmp_path / "real_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "images:",
                "  - id: campus_a",
                "    filename: campus_a.png",
                "    latitude: 33.7470",
                "    longitude: 73.1370",
                "    altitude: 550.0",
                "    heading: 180.0",
            ]
        ),
        encoding="utf-8",
    )

    staged_dir = tmp_path / "staged"
    output_path = tmp_path / "database_index.yaml"
    _script_functions()["import_manifest_to_index"](
        manifest_path=manifest_path,
        output_path=output_path,
        image_root=image_root,
        copy_images_to=staged_dir,
    )

    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert (staged_dir / "campus_a.png").exists()
    assert loaded["images"][0]["filepath"] == "staged/campus_a.png"

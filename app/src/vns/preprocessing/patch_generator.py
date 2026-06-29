from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from ..core.models import ImagePatch
from ..core.geo_utils import NEDPoint

PATCH_SIZE = 256
STRIDE = 128


class PatchGenerator:

    def generate_patches(
        self,
        image: np.ndarray,
        image_center: NEDPoint,
        meters_per_pixel: float,
        source: str,
        base_id: str,
        save_dir: Optional[str] = None,
    ) -> List[ImagePatch]:
        """
        Crop PATCH_SIZE×PATCH_SIZE patches with STRIDE overlap.

        Parameters
        ----------
        image       : BGR uint8 numpy array
        save_dir    : if given, each patch is written to disk as a PNG
                      so you can visually verify content is correct
        """
        assert image.dtype == np.uint8, (
            f"PatchGenerator expects uint8 BGR input, got {image.dtype}"
        )
        assert len(image.shape) == 3 and image.shape[2] == 3, (
            f"PatchGenerator expects 3-channel image, got shape {image.shape}"
        )

        out_dir: Optional[Path] = None
        if save_dir is not None:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        patches: List[ImagePatch] = []
        h, w = image.shape[:2]
        patch_idx = 0

        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch_data = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()

                assert patch_data.dtype == np.uint8, (
                    f"Cropped patch dtype became {patch_data.dtype} — "
                    "input must stay uint8 throughout preprocessing"
                )
                assert patch_data.shape == (PATCH_SIZE, PATCH_SIZE, 3), (
                    f"Unexpected patch shape: {patch_data.shape}"
                )

                # Save to disk for visual inspection (BGR → correct colours in imwrite)
                if out_dir is not None:
                    patch_name = f"{base_id}_p{patch_idx:04d}.png"
                    cv2.imwrite(str(out_dir / patch_name), patch_data)

                pixel_offset_north = (y + PATCH_SIZE // 2) - h // 2
                pixel_offset_east  = (x + PATCH_SIZE // 2) - w // 2
                patch_center = NEDPoint(
                    north=image_center.north + pixel_offset_north * meters_per_pixel,
                    east=image_center.east   + pixel_offset_east  * meters_per_pixel,
                    down=image_center.down,
                )
                patches.append(ImagePatch(
                    data=patch_data,
                    center_position=patch_center,
                    patch_id=f'{base_id}_p{patch_idx}',
                    source=source,
                ))
                patch_idx += 1

        return patches

    def generate_single_patch(
        self,
        image: np.ndarray,
        center: NEDPoint,
        source: str,
        patch_id: str,
    ) -> ImagePatch:
        if image.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
        return ImagePatch(
            data=image.copy(),
            center_position=center,
            patch_id=patch_id,
            source=source,
        )

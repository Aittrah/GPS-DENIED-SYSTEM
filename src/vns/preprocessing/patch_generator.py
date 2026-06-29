from typing import List
import numpy as np
import cv2
from ..core.models import ImagePatch
from ..core.geo_utils import NEDPoint

PATCH_SIZE = 256
STRIDE = 128

class PatchGenerator:

    def generate_patches(self, image: np.ndarray, image_center: NEDPoint,
                         meters_per_pixel: float, source: str,
                         base_id: str) -> List[ImagePatch]:
        patches = []
        h, w = image.shape[:2]
        patch_idx = 0
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch_data = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE].copy()
                pixel_offset_north = (y + PATCH_SIZE // 2) - h // 2
                pixel_offset_east  = (x + PATCH_SIZE // 2) - w // 2
                patch_center = NEDPoint(
                    north=image_center.north + pixel_offset_north * meters_per_pixel,
                    east=image_center.east   + pixel_offset_east  * meters_per_pixel,
                    down=image_center.down
                )
                patches.append(ImagePatch(
                    data=patch_data,
                    center_position=patch_center,
                    patch_id=f'{base_id}_p{patch_idx}',
                    source=source
                ))
                patch_idx += 1
        return patches

    def generate_single_patch(self, image: np.ndarray, center: NEDPoint,
                               source: str, patch_id: str) -> ImagePatch:
        if image.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            image = cv2.resize(image, (PATCH_SIZE, PATCH_SIZE))
        return ImagePatch(
            data=image.copy(),
            center_position=center,
            patch_id=patch_id,
            source=source
        )

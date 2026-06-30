import cv2
import numpy as np

from ..core.geo_utils import NEDPoint
from ..core.models import ImagePatch

PATCH_SIZE = 256
STRIDE = 128


class PatchGenerator:
    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        stride: int | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        effective_stride = patch_size // 2 if stride is None else stride
        if effective_stride <= 0:
            raise ValueError("stride must be positive.")
        self._patch_size = patch_size
        self._stride = effective_stride

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def stride(self) -> int:
        return self._stride

    def generate_patches(
        self,
        image: np.ndarray,
        image_center: NEDPoint,
        meters_per_pixel: float,
        source: str,
        base_id: str,
    ) -> list[ImagePatch]:
        patches: list[ImagePatch] = []
        h, w = image.shape[:2]
        patch_idx = 0
        for y in range(0, h - self._patch_size + 1, self._stride):
            for x in range(0, w - self._patch_size + 1, self._stride):
                patch_data = image[y:y + self._patch_size, x:x + self._patch_size].copy()
                pixel_offset_north = (y + self._patch_size // 2) - h // 2
                pixel_offset_east = (x + self._patch_size // 2) - w // 2
                patch_center = NEDPoint(
                    north=image_center.north + pixel_offset_north * meters_per_pixel,
                    east=image_center.east + pixel_offset_east * meters_per_pixel,
                    down=image_center.down,
                )
                patches.append(ImagePatch(
                    data=patch_data,
                    center_position=patch_center,
                    patch_id=f"{base_id}_p{patch_idx}",
                    source=source,
                    offset_x=x,
                    offset_y=y,
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
        if image.shape[:2] != (self._patch_size, self._patch_size):
            image = cv2.resize(image, (self._patch_size, self._patch_size))
        return ImagePatch(
            data=image.copy(),
            center_position=center,
            patch_id=patch_id,
            source=source,
        )

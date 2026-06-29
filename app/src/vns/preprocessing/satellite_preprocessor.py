from typing import Optional
import cv2
import numpy as np

# Resize so the shorter side becomes this many pixels — aspect ratio preserved.
# Larger = sharper patches but more DINOv2 calls (slower build).
# 1024 gives ~77 patches from an 8K image in ~10 min on CPU.
MAX_SHORT_SIDE = 1024


class SatellitePreprocessor:

    def load_and_preprocess(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            print(f'[WARNING] Could not load: {image_path}')
            return None
        print(f'[Preprocessor] Loaded {image_path}: {img.shape[1]}×{img.shape[0]} px')
        return self.load_and_preprocess_array(img)

    def load_and_preprocess_array(self, img: np.ndarray) -> np.ndarray:
        """
        Resize preserving aspect ratio (shorter side → MAX_SHORT_SIDE),
        then apply mild CLAHE for contrast normalisation.
        Returns BGR uint8 — never float, never normalised.
        """
        img = self._resize_preserve_ratio(img)
        img = self._apply_clahe(img)
        assert img.dtype == np.uint8, f"Must return uint8, got {img.dtype}"
        h, w = img.shape[:2]
        from .patch_generator import PATCH_SIZE, STRIDE
        n_cols = (w - PATCH_SIZE) // STRIDE + 1
        n_rows = (h - PATCH_SIZE) // STRIDE + 1
        print(f'[Preprocessor] After resize: {w}x{h} px -> '
              f'{n_cols}x{n_rows} = {n_cols * n_rows} patches')
        return img

    def _resize_preserve_ratio(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        short = min(h, w)
        if short <= MAX_SHORT_SIDE:
            return img          # already small enough — don't upscale
        scale  = MAX_SHORT_SIDE / short
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def debug_pipeline(image_path: str) -> None:
    """Print shape/dtype/range at every pipeline stage."""
    img = cv2.imread(image_path)
    if img is None:
        print(f'[DEBUG] Cannot load: {image_path}')
        return
    print(f'[DEBUG] Loaded   : {img.shape[1]}×{img.shape[0]}  '
          f'dtype={img.dtype}  min={img.min()}  max={img.max()}')

    preprocessor = SatellitePreprocessor()
    processed = preprocessor.load_and_preprocess_array(img)
    print(f'[DEBUG] Processed: {processed.shape[1]}×{processed.shape[0]}  '
          f'dtype={processed.dtype}  min={processed.min()}  max={processed.max()}')

    from .patch_generator import PatchGenerator
    from ..core.geo_utils import NEDPoint
    pg = PatchGenerator()
    patches = pg.generate_patches(
        image=processed,
        image_center=NEDPoint(0.0, 0.0, 0.0),
        meters_per_pixel=0.5,
        source='satellite',
        base_id='debug',
    )
    if patches:
        p0 = patches[0].data
        print(f'[DEBUG] Patches  : count={len(patches)}  '
              f'dtype={p0.dtype}  min={p0.min()}  max={p0.max()}  '
              f'std={p0.std():.1f}')
    else:
        print('[DEBUG] Patches  : NONE generated')

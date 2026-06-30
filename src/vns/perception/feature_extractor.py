from __future__ import annotations

from typing import Optional
import numpy as np

import cv2

# ── DINOv2 transform (MUST be identical for build and query — never change) ──
try:
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage

    DINO_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    DINO_TRANSFORM = None


class DINOv2Extractor:
    """
    Wraps facebook/dinov2 loaded via torch.hub.
    Produces L2-normalised (384,) float32 descriptors from BGR patches.
    Falls back gracefully when torch is not installed.
    """

    def __init__(self, model_name: str = "dinov2_vits14"):
        """
        model_name options:
          dinov2_vits14 — 384-dim, ~300 MB, recommended for CPU
          dinov2_vitb14 — 768-dim, ~330 MB, GPU preferred
        """
        self.model_name = model_name
        self.device = None
        self.model = None
        self.available = False

        if _TORCH_AVAILABLE:
            self._load_model()

    # ── private ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        import torch
        self.device = torch.device("cpu")
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                verbose=False,
            )
            self.model.eval()
            self.model.to(self.device)
            self.available = True
            print(f"[DINOv2] {self.model_name} loaded on CPU")
        except Exception as exc:
            print(f"[DINOv2] Could not load model: {exc}")
            self.available = False

    def _bgr_to_pil(self, image: np.ndarray):
        """Convert BGR numpy array to PIL RGB image."""
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    # ── public ───────────────────────────────────────────────────────────────

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Input : BGR numpy array (any size)
        Output: (384,) float32 descriptor, or None if unavailable
        """
        if not self.available:
            return None

        import torch
        pil = self._bgr_to_pil(image)
        tensor = DINO_TRANSFORM(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy().astype(np.float32)

    def extract_batch(self,
                      images: list[np.ndarray],
                      batch_size: int = 4) -> Optional[np.ndarray]:
        """
        Process a list of BGR images in batches.
        Returns: np.ndarray shape (N, 384), or None if unavailable.
        """
        if not self.available or not images:
            return None

        import torch
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors = []
            for img in batch_imgs:
                pil = self._bgr_to_pil(img)
                tensors.append(DINO_TRANSFORM(pil))
            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model(batch_tensor)
            results.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(results, axis=0)


class ORBVerifier:
    """
    Geometric verification step used after DINOv2 retrieval.
    Confirms a candidate match using ORB keypoint matching.
    """

    ORB_PARAMS = {
        "nfeatures": 500,
        "scaleFactor": 1.2,
        "nlevels": 8,
    }

    def __init__(self):
        self.orb = cv2.ORB_create(**self.ORB_PARAMS)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def verify(self,
               uav_patch: np.ndarray,
               sat_patch: np.ndarray) -> dict:
        """
        Returns:
          {inlier_count, match_count, verified (bool), confidence (0-1)}
        """
        kp1, desc1 = self.orb.detectAndCompute(uav_patch, None)
        kp2, desc2 = self.orb.detectAndCompute(sat_patch, None)

        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return {"inlier_count": 0, "match_count": 0,
                    "verified": False, "confidence": 0.0}

        try:
            matches = self.matcher.match(desc1, desc2)
        except cv2.error:
            return {"inlier_count": 0, "match_count": 0,
                    "verified": False, "confidence": 0.0}

        matches = sorted(matches, key=lambda m: m.distance)
        good = [m for m in matches if m.distance < 192]
        inlier_count = len(good)
        avg_dist = (sum(m.distance for m in good) / inlier_count
                    if inlier_count else 256.0)
        confidence = max(0.0, 1.0 - avg_dist / 256.0)

        return {
            "inlier_count": inlier_count,
            "match_count":  len(matches),
            "verified":     inlier_count > 10,
            "confidence":   confidence,
        }

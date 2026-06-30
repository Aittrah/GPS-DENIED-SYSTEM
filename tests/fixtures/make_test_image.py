"""Run once to create tests/fixtures/test_sat.jpg used by unit tests."""
import cv2
import numpy as np
from pathlib import Path

def create():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    # Quadrants with distinct colours so patches have real variation
    img[  0:300,   0:300] = (120, 80,  40)   # dark brown
    img[  0:300, 300:600] = (60,  110, 200)   # blue
    img[300:600,   0:300] = (30,  150, 60)    # green
    img[300:600, 300:600] = (200, 200, 100)   # yellow
    # Add a grid pattern so ORB has features to detect
    for i in range(0, 600, 40):
        cv2.line(img, (i, 0), (i, 600), (255, 255, 255), 1)
        cv2.line(img, (0, i), (600, i), (255, 255, 255), 1)
    out = Path(__file__).parent / "test_sat.jpg"
    cv2.imwrite(str(out), img)
    print(f"Created {out}")

if __name__ == "__main__":
    create()

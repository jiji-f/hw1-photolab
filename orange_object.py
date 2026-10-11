"""Reproduce the orange box example: python orange_object.py."""
import json
from pathlib import Path

import cv2
import numpy as np

import photolab as pl


def main():
    root = Path(__file__).resolve().parent
    img = pl.imread(str(root / "data/object.jpeg"))
    bg = pl.imread(str(root / "data/bg.jpeg"))
    lo, hi = (5, 170, 150), (25, 255, 255)
    raw = pl.color_mask(img, lo, hi, 0, 0)
    cleaned = pl.color_mask(img, lo, hi, 3, 9)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select the box using a point on its face, rather than a larger red object.
    candidates = [c for c in contours if cv2.pointPolygonTest(c, (470, 770), False) >= 0]
    if not candidates:
        raise ValueError("orange box not found")
    contour = max(candidates, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], np.uint8)
    # Fill only this outer contour: preserve white artwork and dark lettering.
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1)[..., None]
    composite = pl.as_uint8(img.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha))
    pl.imwrite(str(root / "out/object_on_bg.jpg"), composite)
    pl.imwrite(str(root / "out/object_mask.png"), mask)
    print(json.dumps({"command": "orange_object", "lo": lo, "hi": hi,
                      "open": 3, "close": 9, "feather": 1, "seed": [470, 770],
                      "components_before": int(cv2.connectedComponents(raw)[0] - 1),
                      "components_after": int(cv2.connectedComponents(cleaned)[0] - 1),
                      "components_final": int(cv2.connectedComponents(mask)[0] - 1),
                      "area": int(np.count_nonzero(mask)),
                      "mask_pct": round(100 * np.count_nonzero(mask) / mask.size, 2),
                      "holes": 0, "bbox": cv2.boundingRect(contour)}))


if __name__ == "__main__":
    main()

"""Print measurements for the course images without changing out/."""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

import photolab as pl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lectures", type=Path, help="Path to the course lectures folder")
    args = parser.parse_args()
    lectures = args.lectures or next((parent / "lectures" for parent in Path(__file__).resolve().parents
                                     if (parent / "lectures").is_dir()), None)
    if lectures is None:
        parser.error("course images are required: use --lectures PATH_TO_COURSE/lectures")
    building = pl.imread(str(lectures / "L05-filtering/demo/iu-building.jpg"))
    reference = pl.lanczos_reference(building, 0.25)
    psnr = {m: round(pl.psnr(reference, pl.resize_aa(building, 0.25, m)), 1)
            for m in ("naive", "blur", "area")}
    ball = pl.imread(str(lectures / "L03-color-histograms/demo/ball-lamp.jpg"))
    _, mask, stats = pl.chroma_key(ball, ball, (140, 121, 40), (168, 230, 255))
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    truth = (xx - 190) ** 2 + (yy - 560) ** 2 <= 58 ** 2
    found = mask > 0
    tp = np.count_nonzero(found & truth)
    poster = pl.imread(str(lectures / "L04-geometry/demo/poster-angle.jpg"))
    corners = np.float32([[560, 330], [1250, 220], [1290, 1010], [600, 840]])
    front, _ = pl.rectify(poster, corners, size=(640, 905))
    expected = pl.imread(str(lectures / "L04-geometry/demo/poster-gt.png"))
    ink = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128
    print(json.dumps({"iu_psnr_db": psnr, "ball_stats": stats,
                      "recall_pct": round(100 * tp / truth.sum(), 2),
                      "precision_pct": round(100 * tp / found.sum(), 2),
                      "poster_match_pct": round(100 * np.mean(ink(front) == ink(expected)), 2)}))


if __name__ == "__main__":
    main()

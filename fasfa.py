import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from mtcnn import MTCNN

# ── Settings ──────────────────────────────────────────────────────────────────
RECURSIVE   = True       # recurse into subdirectories
OUTPUT_SIZE = 160        # output image size in pixels (square)
MARGIN      = -0.05       # padding around face; negative = zoom in tighter
NUM_WORKERS = cpu_count()  # number of parallel workers
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Module-level detector, initialized once per worker process
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector


def crop_face(img_rgb):
    detector = _get_detector()
    results = detector.detect_faces(img_rgb)
    if not results:
        return None

    x, y, w, h = max(results, key=lambda r: r["box"][2] * r["box"][3])["box"]
    x, y = max(0, x), max(0, y)
    ih, iw = img_rgb.shape[:2]

    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) / 2 * (1 + MARGIN)

    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(iw, x2), min(ih, y2)

    crop = img_rgb[y1c:y2c, x1c:x2c]

    pad = (x1c - x1, y1c - y1, x2 - x2c, y2 - y2c)  # left, top, right, bottom
    if any(pad):
        crop = cv2.copyMakeBorder(crop, pad[1], pad[3], pad[0], pad[2],
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)


def process_image(args) -> tuple[str, str]:
    """Process a single image. Returns (status, filename)."""
    path, output_dir = args
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return "SKIP", path.name

        face = crop_face(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if face is None:
            return "NO FACE", path.name

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        return "OK", path.name

    except Exception as e:
        # Log the exception if you want
        print(f"[ERROR] {path.name} -> {e}")
        return "ERROR", path.name


def process_dataset(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if RECURSIVE else "*"
    images = [p for p in input_dir.glob(pattern) if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(images)} images — using {NUM_WORKERS} workers\n")

    ok = fail = 0
    with Pool(processes=NUM_WORKERS) as pool:
        args = [(p, output_dir) for p in sorted(images)]

        for status, name in pool.imap_unordered(process_image, args):
            if status == "OK":
                print(f"[OK]       {name}")
                ok += 1
            else:
                print(f"[{status:<8}] {name}")
                fail += 1

    print(f"\nDone — saved: {ok}  |  skipped: {fail}")


if __name__ == "__main__":
    process_dataset("personB_images", "personB")
    process_dataset("personA_images", "personA")
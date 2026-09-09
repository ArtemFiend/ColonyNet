from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_petri_circle(
    img_rgb: np.ndarray,
    min_r_frac: float,
    max_r_frac: float,
    center_tol: float,
) -> Optional[Tuple[int, int, int]]:
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    min_r = int(min(h, w) * min_r_frac)
    max_r = int(min(h, w) * max_r_frac)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        cx, cy, r = circles[np.argmax(circles[:, 2])]
    else:
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        (cx_f, cy_f), r_f = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx_f), int(cy_f), int(r_f)

    # Validate circle
    if r < min_r or r > max_r:
        return None
    cx0, cy0 = w // 2, h // 2
    max_off = center_tol * min(h, w)
    if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 > max_off:
        return None

    return cx, cy, r


def crop_with_circle(
    img_rgb: np.ndarray,
    inst: np.ndarray,
    pad: float,
    mask_outside: bool,
    min_r_frac: float,
    max_r_frac: float,
    center_tol: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    h, w = img_rgb.shape[:2]
    circ = detect_petri_circle(img_rgb, min_r_frac, max_r_frac, center_tol)
    if circ is None:
        meta = {
            "status": "copy",
            "orig_h": h,
            "orig_w": w,
        }
        return img_rgb, inst, meta

    cx, cy, r = circ
    r = int(r * (1.0 + pad))

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    img_crop = img_rgb[y1:y2, x1:x2].copy()
    inst_crop = inst[y1:y2, x1:x2].copy()

    if mask_outside:
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)
        inv = ~mask
        img_crop[inv] = 0
        inst_crop[inv] = 0

    meta = {
        "status": "cropped",
        "orig_h": h,
        "orig_w": w,
        "cx": cx,
        "cy": cy,
        "r": r,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
    }
    return img_crop, inst_crop, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--instances_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--min_r_frac", type=float, default=0.35)
    ap.add_argument("--max_r_frac", type=float, default=0.55)
    ap.add_argument("--center_tol", type=float, default=0.25)
    ap.add_argument("--mask_outside", action="store_true", help="Zero-out pixels outside dish")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    inst_dir = Path(args.instances_dir)
    out_root = Path(args.out_root)
    out_images = out_root / "images"
    out_instances = out_root / "instances"
    out_images.mkdir(parents=True, exist_ok=True)
    out_instances.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "crop_meta.jsonl"
    fail_path = out_root / "failed.txt"

    allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    total = 0
    cropped = 0
    copied = 0
    missing = 0

    def to_py(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (list, tuple)):
            return [to_py(x) for x in v]
        return v

    with open(meta_path, "w", encoding="utf-8") as mf, open(fail_path, "w", encoding="utf-8") as ff:
        for img_path in tqdm(sorted(img_dir.iterdir()), desc="crop petri"):
            if not img_path.is_file() or img_path.suffix.lower() not in allowed_ext:
                continue
            stem = img_path.stem
            inst_path = inst_dir / f"{stem}.png"
            if not inst_path.exists():
                missing += 1
                ff.write(str(img_path) + " | no instance\n")
                continue

            img_rgb = read_rgb(str(img_path))
            inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
            if inst is None:
                missing += 1
                ff.write(str(img_path) + " | failed instance read\n")
                continue

            img_crop, inst_crop, meta = crop_with_circle(
                img_rgb,
                inst,
                pad=args.pad,
                mask_outside=args.mask_outside,
                min_r_frac=args.min_r_frac,
                max_r_frac=args.max_r_frac,
                center_tol=args.center_tol,
            )

            if meta.get("status") == "cropped":
                cropped += 1
            else:
                copied += 1

            out_img = out_images / f"{stem}.png"
            out_inst = out_instances / f"{stem}.png"
            cv2.imwrite(str(out_img), cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_inst), inst_crop)

            meta["id"] = stem
            meta = {k: to_py(v) for k, v in meta.items()}
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            total += 1

    print("Done.")
    print(f"Total processed: {total}")
    print(f"Cropped: {cropped}")
    print(f"Copied (no reliable circle): {copied}")
    print(f"Missing instances: {missing}")
    print("Output:", out_root)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import json
import argparse
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

# Optional: for COCO RLE
try:
    from pycocotools import mask as maskUtils  # type: ignore
except Exception:
    maskUtils = None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_image_any(path: str) -> np.ndarray:
    """Reads image with OpenCV; returns RGB uint8 HxWx3."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    # Handle grayscale
    if img.ndim == 2:
        # uint16 -> uint8
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    # Handle color
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # BGR -> RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # unexpected channels, fallback: take first 3 if possible
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def ann_to_mask(segmentation: Any, h: int, w: int) -> np.ndarray:
    """Convert COCO segmentation (polygon or RLE) to a binary mask uint8 (0/1)."""
    if segmentation is None:
        return np.zeros((h, w), dtype=np.uint8)

    def align_mask_shape(m: np.ndarray, h: int, w: int) -> np.ndarray:
        """Ensure decoded mask matches expected (h, w); fix common swapped dims."""
        if m.shape[0] == h and m.shape[1] == w:
            return m
        if m.shape[0] == w and m.shape[1] == h:
            if m.ndim == 2:
                return m.T
            return np.transpose(m, (1, 0, 2))
        raise ValueError(f"Decoded mask shape {m.shape[:2]} does not match expected {(h, w)}")

    # Polygon(s): list of lists
    if isinstance(segmentation, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segmentation:
            if not poly:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    # RLE dict: can be compressed (counts=str/bytes) or uncompressed (counts=list)
    if isinstance(segmentation, dict):
        if maskUtils is None:
            raise RuntimeError("pycocotools is required to decode RLE. Install: pip install pycocotools")

        counts = segmentation.get("counts", None)

        if isinstance(counts, list):
            # uncompressed RLE -> convert to compressed RLE first
            rle = maskUtils.frPyObjects(segmentation, h, w)  # returns list or dict
            m = maskUtils.decode(rle)
        else:
            # already compressed RLE
            m = maskUtils.decode(segmentation)

        # m can be HxW or HxWxN
        if m.ndim == 3:
            m = align_mask_shape(m, h, w)
            m = np.any(m, axis=2).astype(np.uint8)
        else:
            m = align_mask_shape(m, h, w)
            m = (m > 0).astype(np.uint8)
        return m

    # Fallback (rare)
    if maskUtils is None:
        raise RuntimeError("Unknown COCO segmentation type and pycocotools is not available.")
    rle = maskUtils.frPyObjects(segmentation, h, w)
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = align_mask_shape(m, h, w)
        m = np.any(m, axis=2).astype(np.uint8)
    else:
        m = align_mask_shape(m, h, w)
        m = (m > 0).astype(np.uint8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder with LiveCell images (tif)")
    ap.add_argument("--ann_json", required=True, help="COCO annotation json (e.g., 3_train25percent.json)")
    ap.add_argument("--out_root", required=True, help="Output folder (will create images/ and instances/)")
    ap.add_argument("--export_png", action="store_true", help="Export images as 8-bit RGB PNG (recommended)")
    ap.add_argument("--copy_images", action="store_true", help="If not export_png: copy original image files to out/images/")
    args = ap.parse_args()

    out_images = os.path.join(args.out_root, "images")
    out_instances = os.path.join(args.out_root, "instances")
    ensure_dir(out_images)
    ensure_dir(out_instances)

    with open(args.ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # image_id -> list[ann]
    ann_map: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        img_id = int(a["image_id"])
        ann_map.setdefault(img_id, []).append(a)

    # Process each image
    for im in tqdm(images, desc="convert LIVECell"):
        img_id = int(im["id"])
        file_name = im["file_name"]
        h = int(im.get("height", 0))
        w = int(im.get("width", 0))

        src_path = os.path.join(args.images_dir, file_name)
        if not os.path.exists(src_path):
            # Sometimes file_name contains subfolders; try basename fallback
            src_path2 = os.path.join(args.images_dir, os.path.basename(file_name))
            if os.path.exists(src_path2):
                src_path = src_path2
            else:
                raise FileNotFoundError(f"Image not found: {src_path}")

        stem = os.path.splitext(os.path.basename(file_name))[0]

        # Read image (for size sanity + optional export)
        img_rgb = read_image_any(src_path)
        H, W = img_rgb.shape[:2]
        if h != H or w != W:
            h, w = H, W

        # Build instance-id mask
        inst = np.zeros((h, w), dtype=np.uint16)
        obj_id = 1
        for a in ann_map.get(img_id, []):
            seg = a.get("segmentation", None)
            m = ann_to_mask(seg, h, w)
            if m.sum() == 0:
                continue
            # assign id; if overlaps occur, later instances overwrite (acceptable for pretrain)
            inst[m > 0] = obj_id
            obj_id += 1
            if obj_id >= 65535:
                break

        # Save instances mask
        inst_path = os.path.join(out_instances, f"{stem}.png")
        cv2.imwrite(inst_path, inst)

        # Save/copy image
        if args.export_png:
            out_img_path = os.path.join(out_images, f"{stem}.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        elif args.copy_images:
            out_img_path = os.path.join(out_images, os.path.basename(file_name))
            ensure_dir(os.path.dirname(out_img_path))
            shutil.copy2(src_path, out_img_path)
        else:
            # Default: still export PNG if neither option specified
            out_img_path = os.path.join(out_images, f"{stem}.png")
            cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    print("Done.")
    print("Images:", out_images)
    print("Instances:", out_instances)


if __name__ == "__main__":
    main()

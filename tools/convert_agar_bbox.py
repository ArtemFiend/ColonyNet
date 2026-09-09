from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_image_json_pairs(root: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Finds (jpg_path, json_path, background_label) pairs.
    background_label is inferred from folder name: bright/dark/vague if present.
    """
    pairs: List[Tuple[str, str, Optional[str]]] = []

    # search jpg recursively
    jpgs = glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    jpgs += glob(os.path.join(root, "**", "*.jpeg"), recursive=True)
    jpgs = sorted(set(jpgs))

    for jpg_path in jpgs:
        stem, _ = os.path.splitext(jpg_path)
        json_path = stem + ".json"
        if not os.path.exists(json_path):
            # allow mismatch in extension case
            continue

        # infer background by path component
        bg = None
        parts = [p.lower() for p in os.path.normpath(jpg_path).split(os.sep)]
        for cand in ("bright", "dark", "vague"):
            if cand in parts:
                bg = cand
                break

        pairs.append((jpg_path, json_path, bg))

    return pairs


def clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Clip bbox to image boundaries and return possibly reduced (x,y,w,h)."""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    nw = max(0, x2 - x1)
    nh = max(0, y2 - y1)
    return x1, y1, nw, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agar_root", required=True, help="Root folder: higher-resolution or lower-resolution")
    ap.add_argument("--out_root", required=True, help="Output folder: data/agar_hr or data/agar_lr")
    ap.add_argument("--shape", default="ellipse", choices=["ellipse", "rect"],
                    help="Pseudo-instance shape drawn inside bbox (ellipse recommended)")
    ap.add_argument("--copy_images", action="store_true", help="Copy jpg as-is to out/images (recommended)")
    args = ap.parse_args()

    out_images = os.path.join(args.out_root, "images")
    out_instances = os.path.join(args.out_root, "instances")
    meta_path = os.path.join(args.out_root, "meta.jsonl")
    ensure_dir(out_images)
    ensure_dir(out_instances)

    pairs = list_image_json_pairs(args.agar_root)
    if not pairs:
        raise RuntimeError(f"No (jpg,json) pairs found under: {args.agar_root}")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for jpg_path, json_path, bg in tqdm(pairs, desc="convert AGAR"):
            img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(jpg_path)
            H, W = img.shape[:2]

            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            labels = ann.get("labels", [])
            # instance-id mask
            inst = np.zeros((H, W), dtype=np.uint16)

            obj_id = 1
            for obj in labels:
                x = int(obj.get("x", 0))
                y = int(obj.get("y", 0))
                w = int(obj.get("width", 0))
                h = int(obj.get("height", 0))
                if w <= 0 or h <= 0:
                    continue

                x, y, w, h = clip_box(x, y, w, h, W, H)
                if w <= 0 or h <= 0:
                    continue

                if args.shape == "rect":
                    inst[y:y+h, x:x+w] = obj_id
                else:
                    cx = x + w // 2
                    cy = y + h // 2
                    ax1 = max(1, w // 2)
                    ax2 = max(1, h // 2)
                    cv2.ellipse(inst, (cx, cy), (ax1, ax2), 0, 0, 360, int(obj_id), thickness=-1)

                obj_id += 1
                if obj_id >= 65535:
                    break

            # file naming: keep original stem
            stem = os.path.splitext(os.path.basename(jpg_path))[0]

            # save instances
            inst_out = os.path.join(out_instances, f"{stem}.png")
            cv2.imwrite(inst_out, inst)

            # save/copy image
            if args.copy_images:
                img_out = os.path.join(out_images, os.path.basename(jpg_path))
                shutil.copy2(jpg_path, img_out)
            else:
                # export as png
                img_out = os.path.join(out_images, f"{stem}.png")
                cv2.imwrite(img_out, img)

            # write metadata line
            meta = {
                "id": stem,
                "image": os.path.basename(img_out),
                "instances": os.path.basename(inst_out),
                "background": bg or ann.get("background", None),
                "classes": ann.get("classes", None),
                "colonies_number": ann.get("colonies_number", len(labels)),
                "source_json": os.path.basename(json_path),
                "H": H,
                "W": W,
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print("Done.")
    print("Images:", out_images)
    print("Instances:", out_instances)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()

from __future__ import annotations
import os
import argparse
import glob
import numpy as np
import cv2
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsb_root", required=True, help="Path to stage1_train folder")
    ap.add_argument("--out_root", required=True, help="Output folder with images/ and instances/")
    args = ap.parse_args()

    img_out = os.path.join(args.out_root, "images")
    inst_out = os.path.join(args.out_root, "instances")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(inst_out, exist_ok=True)

    ids = [d for d in os.listdir(args.dsb_root) if os.path.isdir(os.path.join(args.dsb_root, d))]
    ids = sorted(ids)

    for _id in tqdm(ids, desc="convert"):
        img_dir = os.path.join(args.dsb_root, _id, "images")
        msk_dir = os.path.join(args.dsb_root, _id, "masks")

        img_files = glob.glob(os.path.join(img_dir, "*"))
        if len(img_files) != 1:
            continue
        img_path = img_files[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        mask_files = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
        if len(mask_files) == 0:
            continue

        H, W = img.shape[:2]
        inst = np.zeros((H, W), dtype=np.uint16)
        cur = 1
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[:, :, 0]
            m = (m > 0).astype(np.uint8)
            if m.sum() == 0:
                continue
            inst[m > 0] = cur
            cur += 1

        cv2.imwrite(os.path.join(img_out, f"{_id}.png"), img)
        cv2.imwrite(os.path.join(inst_out, f"{_id}.png"), inst)

if __name__ == "__main__":
    main()

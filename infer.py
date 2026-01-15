from __future__ import annotations
import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from skimage.segmentation import find_boundaries

from colonyseg.models.colonymet import ColonyNet
from colonyseg.post.watershed import postprocess_watershed
from colonyseg.utils import ensure_dir

def overlay_boundaries(img_rgb: np.ndarray, labels: np.ndarray):
    out = img_rgb.copy()
    b = find_boundaries(labels, mode="outer")
    out[b] = (0, 255, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--t_sem", type=float, default=0.5)
    ap.add_argument("--t_center", type=float, default=0.35)
    ap.add_argument("--min_distance", type=int, default=6)
    ap.add_argument("--lambda_boundary", type=float, default=3.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    backbone_id = cfg.get("model", {}).get("backbone_id", "nvidia/mit-b2")
    fpn_dim = int(cfg.get("model", {}).get("fpn_dim", 256))

    model = ColonyNet(backbone_id=backbone_id, fpn_dim=fpn_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    files = [p for p in os.listdir(args.input_dir) if p.lower().endswith((".png",".jpg",".jpeg"))]
    files = sorted(files)

    for fn in tqdm(files, desc="infer"):
        p = os.path.join(args.input_dir, fn)
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)

        x = torch.from_numpy(img_rs).float().permute(2,0,1) / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            sem_p = torch.sigmoid(pred["sem"]).cpu().numpy()[0,0]
            cen_p = torch.sigmoid(pred["center"]).cpu().numpy()[0,0]
            bnd_p = torch.sigmoid(pred["boundary"]).cpu().numpy()[0,0]

        labels = postprocess_watershed(
            sem_p, cen_p, bnd_p,
            t_sem=args.t_sem, t_center=args.t_center,
            min_distance=args.min_distance, lambda_boundary=args.lambda_boundary
        )
        # Upscale labels to img_size for overlay
        labels_up = cv2.resize(labels.astype(np.int32), (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        over = overlay_boundaries(img_rs, labels_up)

        out_path = os.path.join(args.out_dir, os.path.splitext(fn)[0] + "_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()

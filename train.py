from __future__ import annotations
import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from colonyseg.utils import set_seed, ensure_dir
from colonyseg.data.datasets import ImageInstancesDataset, split_ids
from colonyseg.data.transforms import build_train_tf, build_val_tf
from colonyseg.models.colonymet import ColonyNet, set_backbone_trainable
from colonyseg.losses import loss_total
from colonyseg.post.watershed import postprocess_watershed
from colonyseg.metrics.instance_metrics import instance_scores

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs_dir", default="runs")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    run_dir = os.path.join(args.runs_dir, cfg["run_name"])
    ensure_dir(run_dir)

    # Build datasets
    img_size = int(cfg["data"]["img_size"])
    out_stride = int(cfg["data"]["out_stride"])
    train_tf = build_train_tf(img_size)
    val_tf = build_val_tf(img_size)

    # IDs list
    all_imgs = sorted([p for p in os.listdir(cfg["data"]["train_images"]) if p.lower().endswith((".png",".jpg",".jpeg"))])
    all_ids = [os.path.splitext(p)[0] for p in all_imgs]
    train_ids, val_ids = split_ids(all_ids, float(cfg["data"]["val_split"]), seed=int(cfg.get("seed", 42)))

    train_ds = ImageInstancesDataset(
        cfg["data"]["train_images"], cfg["data"]["train_instances"],
        transform=train_tf, img_size=img_size, out_stride=out_stride, ids=train_ids
    )
    val_ds = ImageInstancesDataset(
        cfg["data"]["val_images"], cfg["data"]["val_instances"],
        transform=val_tf, img_size=img_size, out_stride=out_stride, ids=val_ids
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                              num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, int(cfg["train"]["num_workers"])//2), pin_memory=True)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColonyNet(backbone_id=cfg["model"]["backbone_id"], fpn_dim=int(cfg["model"]["fpn_dim"])).to(device)

    # Optimizer with differential LR
    lr_head = float(cfg["train"]["lr_head"])
    lr_backbone = float(cfg["train"]["lr_backbone"])
    wd = float(cfg["train"]["weight_decay"])

    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optim = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=wd
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    best_f1 = -1.0
    freeze_epochs = int(cfg["train"]["freeze_backbone_epochs"])
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        # Freeze/unfreeze schedule
        if epoch <= freeze_epochs:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        loss_avg = 0.0
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y_sem = batch["y_sem"].to(device, non_blocking=True)
            y_center = batch["y_center"].to(device, non_blocking=True)
            y_boundary = batch["y_boundary"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                pred = model(x)
                loss, parts = loss_total(pred, y_sem, y_center, y_boundary)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            loss_avg = 0.98*loss_avg + 0.02*loss.item() if loss_avg > 0 else loss.item()
            pbar.set_postfix(loss=f"{loss_avg:.4f}", **{k: f"{v:.3f}" for k,v in parts.items()})

        # Validation: postprocess + instance metrics
        model.eval()
        metrics = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                x = batch["image"].to(device, non_blocking=True)
                gt_inst = batch["instances"].cpu().numpy()[0]  # H,W
                pred = model(x)

                # probs at out-res (stride 4)
                sem_p = torch.sigmoid(pred["sem"]).cpu().numpy()[0,0]
                cen_p = torch.sigmoid(pred["center"]).cpu().numpy()[0,0]
                bnd_p = torch.sigmoid(pred["boundary"]).cpu().numpy()[0,0]

                pr_labels = postprocess_watershed(
                    sem_p, cen_p, bnd_p,
                    t_sem=float(cfg["post"]["t_sem"]),
                    t_center=float(cfg["post"]["t_center"]),
                    min_distance=int(cfg["post"]["min_distance"]),
                    lambda_boundary=float(cfg["post"]["lambda_boundary"]),
                    area_min=int(cfg["post"]["area_min"]),
                    area_max=int(cfg["post"]["area_max"]),
                )

                # Downscale GT instances to out-res for fair comparison
                H, W = gt_inst.shape
                out_h, out_w = sem_p.shape
                import cv2
                gt_small = cv2.resize(gt_inst.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)

                m = instance_scores(gt_small, pr_labels, iou_thr=float(cfg["train"]["iou_thr"]))
                metrics.append(m)

        # Aggregate
        mean_f1 = float(np.mean([m["f1"] for m in metrics])) if metrics else 0.0
        mean_mer = float(np.mean([m["merge"] for m in metrics])) if metrics else 0.0
        mean_spl = float(np.mean([m["split"] for m in metrics])) if metrics else 0.0
        mean_cnt = float(np.mean([m["count_err"] for m in metrics])) if metrics else 0.0

        print(f"[val] f1={mean_f1:.4f} merge={mean_mer:.3f} split={mean_spl:.3f} count_err={mean_cnt:.3f}")

        # Save
        last_path = os.path.join(run_dir, "last.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, last_path)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_path = os.path.join(run_dir, "best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, best_path)
            print(f"Saved best: {best_path}")

if __name__ == "__main__":
    main()

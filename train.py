from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from colonyseg.data.datasets import ImageInstancesDataset, split_ids
from colonyseg.data.splits import validate_split
from colonyseg.data.transforms import build_train_tf, build_val_tf
from colonyseg.eval import evaluate_checkpoint_on_test_image
from colonyseg.losses import loss_total
from colonyseg.metrics.instance_metrics import instance_scores
from colonyseg.models.colonymet import ColonyNet, set_backbone_trainable
from colonyseg.post.watershed import postprocess_watershed
from colonyseg.utils import ensure_dir, set_seed

try:
    import mlflow
except Exception:
    mlflow = None


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MODEL_VARIANTS = {
    "mit-b2": "nvidia/mit-b2",
    "mit-b3": "nvidia/mit-b3",
}


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ids_file(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def collect_pair_ids(images_dir: str, instances_dir: str) -> list[str]:
    image_ids = {
        os.path.splitext(p)[0] for p in os.listdir(images_dir) if p.lower().endswith(IMG_EXTS)
    }
    instance_ids = {
        os.path.splitext(p)[0] for p in os.listdir(instances_dir) if p.lower().endswith(".png")
    }
    return sorted(image_ids & instance_ids)


def flatten_cfg(cfg: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_cfg(v, key))
        elif isinstance(v, (list, tuple)):
            sval = ",".join(str(x) for x in v)
            out[key] = sval[:250]
        elif v is not None:
            out[key] = str(v)[:250]
    return out


def infer_model_variant(backbone_id: str) -> str:
    reverse = {v: k for k, v in MODEL_VARIANTS.items()}
    return reverse.get(backbone_id, backbone_id)


def resolve_model_config(model_cfg: dict) -> tuple[str, int, str]:
    raw_variant = model_cfg.get("model_variant", model_cfg.get("variant", ""))
    model_variant = str(raw_variant).strip() if raw_variant is not None else ""
    raw_backbone = model_cfg.get("backbone_id", "")
    backbone_id = str(raw_backbone).strip() if raw_backbone is not None else ""
    fpn_dim = int(model_cfg.get("fpn_dim", 256))

    backbone_from_variant = None
    if model_variant:
        key = model_variant.lower()
        if key in MODEL_VARIANTS:
            backbone_from_variant = MODEL_VARIANTS[key]
        elif "/" in model_variant:
            # Allow direct HF model id in model_variant
            backbone_from_variant = model_variant

    if backbone_id:
        if backbone_from_variant is not None and backbone_id != backbone_from_variant:
            print(
                f"[warn] model_variant='{model_variant}' resolves to '{backbone_from_variant}', "
                f"but model.backbone_id='{backbone_id}'. Using explicit backbone_id."
            )
    else:
        if backbone_from_variant is not None:
            backbone_id = backbone_from_variant
        elif model_variant:
            known = ", ".join(sorted(MODEL_VARIANTS.keys()))
            raise ValueError(
                "Unknown model.model_variant without model.backbone_id. "
                f"Got '{model_variant}'. Known variants: {known}. "
                "Or set model.backbone_id to full HF id."
            )
        else:
            raise ValueError("Set either model.backbone_id or model.model_variant in config.")

    if not model_variant:
        model_variant = infer_model_variant(backbone_id)

    return backbone_id, fpn_dim, model_variant


def save_history(run_dir: str, history: dict[str, list[float]]) -> tuple[str, str | None]:
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_path: str | None = os.path.join(run_dir, "history.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))

        ax[0, 0].plot(epochs, history["train_loss"], label="train_loss")
        ax[0, 0].set_title("Train Loss")
        ax[0, 0].grid(True, alpha=0.3)

        ax[0, 1].plot(epochs, history["val_f1"], label="val_f1")
        ax[0, 1].plot(epochs, history["val_precision"], label="val_precision")
        ax[0, 1].plot(epochs, history["val_recall"], label="val_recall")
        ax[0, 1].set_title("Validation Quality")
        ax[0, 1].legend()
        ax[0, 1].grid(True, alpha=0.3)

        ax[1, 0].plot(epochs, history["val_merge"], label="merge")
        ax[1, 0].plot(epochs, history["val_split"], label="split")
        ax[1, 0].plot(epochs, history["val_count_err"], label="count_err")
        ax[1, 0].set_title("Validation Errors")
        ax[1, 0].legend()
        ax[1, 0].grid(True, alpha=0.3)

        ax[1, 1].plot(epochs, history["epoch_time"], label="epoch_time_sec")
        ax[1, 1].set_title("Epoch Time (s)")
        ax[1, 1].grid(True, alpha=0.3)

        for a in ax.ravel():
            a.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        plot_path = None
        print(f"[warn] history plot skipped: {exc}")

    return history_path, plot_path


def run_test_segmentation(cfg: dict, run_dir: str, ckpt_path: str, device: str) -> dict | None:
    test_cfg = cfg.get("test_seg", {})
    if not bool(test_cfg.get("enabled", False)):
        return None

    image_path = test_cfg.get("image_path")
    if not image_path:
        print("[warn] test_seg.enabled=true but image_path is missing")
        return None

    reference_path = (
        test_cfg.get("reference_path")
        or test_cfg.get("reference_masks")
        or test_cfg.get("reference_masks_dir")
    )
    out_dir = os.path.join(run_dir, test_cfg.get("out_subdir", "test_seg"))
    iou_thr = float(test_cfg.get("iou_thr", cfg.get("train", {}).get("iou_thr", 0.5)))

    return evaluate_checkpoint_on_test_image(
        ckpt_path=ckpt_path,
        image_path=image_path,
        out_dir=out_dir,
        reference_path=reference_path,
        iou_thr=iou_thr,
        petri_pad=float(test_cfg.get("pad", 0.02)),
        petri_mask_outside=bool(test_cfg.get("mask_outside", True)),
        petri_min_r_frac=float(test_cfg.get("min_r_frac", 0.35)),
        petri_max_r_frac=float(test_cfg.get("max_r_frac", 0.55)),
        petri_center_tol=float(test_cfg.get("center_tol", 0.25)),
        device=device,
    )


def log_mlflow_artifacts_if_enabled(
    use_mlflow: bool,
    args: argparse.Namespace,
    run_dir: str,
    history_path: str,
    history_plot_path: str | None,
    best_path: str,
    last_path: str,
    test_eval: dict | None,
    epochs: int,
) -> None:
    if not use_mlflow:
        return

    mlflow.log_artifact(args.config, artifact_path="config")
    mlflow.log_artifact(history_path, artifact_path="history")
    if history_plot_path is not None:
        mlflow.log_artifact(history_plot_path, artifact_path="history")
    if os.path.exists(best_path):
        mlflow.log_artifact(best_path, artifact_path="checkpoints")
    if os.path.exists(last_path):
        mlflow.log_artifact(last_path, artifact_path="checkpoints")

    if test_eval is not None:
        for k, v in test_eval.get("metrics", {}).items():
            if isinstance(v, (float, int)) and np.isfinite(float(v)):
                mlflow.log_metric(f"test_seg_{k}", float(v), step=epochs)
        for apath in test_eval.get("artifacts", {}).values():
            if os.path.exists(apath):
                mlflow.log_artifact(apath, artifact_path="test_seg")

    mlflow.log_param("run_dir", run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    ap.add_argument("--no_mlflow", action="store_true", help="Disable MLflow logging")
    ap.add_argument("--mlflow_tracking_uri", default=None)
    ap.add_argument("--mlflow_experiment", default=None)
    ap.add_argument("--mlflow_run_name", default=None)
    ap.add_argument(
        "--model_variant",
        default=None,
        help="Model variant shortcut (e.g. mit-b2, mit-b3) or full HF model id.",
    )
    ap.add_argument("--backbone_id", default=None, help="Explicit HF backbone id override.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("model", {})
    if args.model_variant is not None:
        cfg["model"]["model_variant"] = str(args.model_variant)
    if args.backbone_id is not None:
        cfg["model"]["backbone_id"] = str(args.backbone_id)

    set_seed(int(cfg.get("seed", 42)))

    run_dir = os.path.join(args.runs_dir, cfg["run_name"])
    ensure_dir(run_dir)

    mlflow_cfg = cfg.get("mlflow", {})
    use_mlflow = bool(mlflow_cfg.get("enabled", False))
    if args.mlflow:
        use_mlflow = True
    if args.no_mlflow:
        use_mlflow = False

    if use_mlflow and mlflow is None:
        raise RuntimeError("MLflow is enabled but package `mlflow` is not installed.")

    if use_mlflow:
        tracking_uri = args.mlflow_tracking_uri or mlflow_cfg.get("tracking_uri")
        experiment = args.mlflow_experiment or mlflow_cfg.get("experiment", "colony_segmentation")
        run_name = args.mlflow_run_name or mlflow_cfg.get("run_name", cfg.get("run_name"))
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        run_ctx = mlflow.start_run(run_name=run_name)
    else:
        run_ctx = contextlib.nullcontext()

    with run_ctx:
        # Build datasets
        img_size = int(cfg["data"]["img_size"])
        out_stride = int(cfg["data"]["out_stride"])
        train_tf = build_train_tf(img_size)
        val_tf = build_val_tf(img_size)

        # IDs list
        train_images_dir = cfg["data"]["train_images"]
        train_instances_dir = cfg["data"]["train_instances"]
        val_images_dir = cfg["data"]["val_images"]
        val_instances_dir = cfg["data"]["val_instances"]
        train_ids_file = cfg["data"].get("train_ids_file", None)
        val_ids_file = cfg["data"].get("val_ids_file", None)
        same_train_val = (
            os.path.abspath(train_images_dir) == os.path.abspath(val_images_dir)
            and os.path.abspath(train_instances_dir) == os.path.abspath(val_instances_dir)
        )

        if train_ids_file is not None and val_ids_file is not None:
            train_ids = load_ids_file(train_ids_file)
            val_ids = load_ids_file(val_ids_file)
        elif same_train_val:
            all_ids = collect_pair_ids(train_images_dir, train_instances_dir)
            train_ids, val_ids = split_ids(
                all_ids, float(cfg["data"]["val_split"]), seed=int(cfg.get("seed", 42))
            )
        else:
            train_ids = collect_pair_ids(train_images_dir, train_instances_dir)
            val_ids = collect_pair_ids(val_images_dir, val_instances_dir)

        validate_split(train_ids, val_ids)
        target_cfg = cfg.get("targets", None)
        train_repeat = int(cfg["data"].get("train_repeat", 1))
        train_ds = ImageInstancesDataset(
            cfg["data"]["train_images"],
            cfg["data"]["train_instances"],
            transform=train_tf,
            img_size=img_size,
            out_stride=out_stride,
            ids=train_ids,
            target_cfg=target_cfg,
            repeat=train_repeat,
        )
        val_ds = ImageInstancesDataset(
            cfg["data"]["val_images"],
            cfg["data"]["val_instances"],
            transform=val_tf,
            img_size=img_size,
            out_stride=out_stride,
            ids=val_ids,
            target_cfg=target_cfg,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, int(cfg["train"]["num_workers"]) // 2),
            pin_memory=True,
        )

        # Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone_id, fpn_dim, model_variant = resolve_model_config(cfg["model"])
        # Persist resolved model setup into cfg for checkpoints and downstream inference.
        cfg["model"]["backbone_id"] = backbone_id
        cfg["model"]["fpn_dim"] = fpn_dim
        cfg["model"]["model_variant"] = model_variant

        model = ColonyNet(
            backbone_id=backbone_id,
            fpn_dim=fpn_dim,
        ).to(device)
        amp_enabled = bool(cfg["train"]["amp"]) and device == "cuda"
        init_ckpt = cfg["train"].get("init_ckpt", None)
        if init_ckpt:
            ckpt = torch.load(init_ckpt, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
            print(f"Loaded init checkpoint: {init_ckpt}")

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
            weight_decay=wd,
        )

        scaler = torch.amp.GradScaler(device, enabled=amp_enabled)

        best_f1 = -1.0
        best_epoch = 0
        freeze_epochs = int(cfg["train"]["freeze_backbone_epochs"])
        epochs = int(cfg["train"]["epochs"])
        last_path = os.path.join(run_dir, "last.pt")
        best_path = os.path.join(run_dir, "best.pt")
        history = {
            "train_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_merge": [],
            "val_split": [],
            "val_count_err": [],
            "epoch_time": [],
        }

        if use_mlflow:
            mlflow.set_tags(
                {
                    "model_name": "ColonyNet",
                    "model_variant": model_variant,
                    "backbone_id": backbone_id,
                }
            )
            mlflow.log_params(flatten_cfg(cfg))
            mlflow.log_param("model_variant", model_variant)
            mlflow.log_param("backbone_id", backbone_id)
            mlflow.log_metric(
                "model_params_total", float(sum(p.numel() for p in model.parameters())), step=0
            )
            mlflow.log_metric(
                "model_params_trainable",
                float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                step=0,
            )

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            # Freeze/unfreeze schedule
            if epoch <= freeze_epochs:
                set_backbone_trainable(model, False)
            else:
                set_backbone_trainable(model, True)

            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
            loss_sum = 0.0
            n_batches = 0
            for batch in pbar:
                x = batch["image"].to(device, non_blocking=True)
                y_sem = batch["y_sem"].to(device, non_blocking=True)
                y_center = batch["y_center"].to(device, non_blocking=True)
                y_boundary = batch["y_boundary"].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device, enabled=amp_enabled):
                    pred = model(x)
                    loss_weights = cfg["train"].get("loss_weights", None)
                    boundary_dice = float(cfg["train"].get("boundary_dice", 0.0))
                    loss, parts = loss_total(
                        pred,
                        y_sem,
                        y_center,
                        y_boundary,
                        weights=loss_weights,
                        boundary_dice=boundary_dice,
                    )

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                loss_sum += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", **{k: f"{v:.3f}" for k, v in parts.items()})

            train_loss = loss_sum / max(1, n_batches)

            # Validation: postprocess + instance metrics
            model.eval()
            metrics = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                    x = batch["image"].to(device, non_blocking=True)
                    gt_inst = batch["instances"].cpu().numpy()[0]  # H,W
                    pred = model(x)

                    # probs at out-res (stride 4)
                    sem_p = torch.sigmoid(pred["sem"]).cpu().numpy()[0, 0]
                    cen_p = torch.sigmoid(pred["center"]).cpu().numpy()[0, 0]
                    bnd_p = torch.sigmoid(pred["boundary"]).cpu().numpy()[0, 0]

                    pr_labels = postprocess_watershed(
                        sem_p,
                        cen_p,
                        bnd_p,
                        t_sem=float(cfg["post"]["t_sem"]),
                        t_center=float(cfg["post"]["t_center"]),
                        min_distance=int(cfg["post"]["min_distance"]),
                        lambda_boundary=float(cfg["post"]["lambda_boundary"]),
                        area_min=int(cfg["post"]["area_min"]),
                        area_max=int(cfg["post"]["area_max"]),
                    )

                    # Downscale GT instances to out-res for fair comparison
                    out_h, out_w = sem_p.shape
                    gt_small = cv2.resize(
                        gt_inst.astype(np.int32), (out_w, out_h), interpolation=cv2.INTER_NEAREST
                    )
                    m = instance_scores(gt_small, pr_labels, iou_thr=float(cfg["train"]["iou_thr"]))
                    metrics.append(m)

            # Aggregate
            mean_prec = float(np.mean([m["precision"] for m in metrics])) if metrics else 0.0
            mean_rec = float(np.mean([m["recall"] for m in metrics])) if metrics else 0.0
            mean_f1 = float(np.mean([m["f1"] for m in metrics])) if metrics else 0.0
            mean_mer = float(np.mean([m["merge"] for m in metrics])) if metrics else 0.0
            mean_spl = float(np.mean([m["split"] for m in metrics])) if metrics else 0.0
            mean_cnt = float(np.mean([m["count_err"] for m in metrics])) if metrics else 0.0

            # Save
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, last_path)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_epoch = epoch
                torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, best_path)
                print(f"Saved best: {best_path}")

            epoch_time = time.time() - epoch_start
            history["train_loss"].append(train_loss)
            history["val_precision"].append(mean_prec)
            history["val_recall"].append(mean_rec)
            history["val_f1"].append(mean_f1)
            history["val_merge"].append(mean_mer)
            history["val_split"].append(mean_spl)
            history["val_count_err"].append(mean_cnt)
            history["epoch_time"].append(epoch_time)

            avg_epoch = float(np.mean(history["epoch_time"][-5:]))
            eta_sec = avg_epoch * (epochs - epoch)

            print(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_f1={mean_f1:.4f} | merge={mean_mer:.3f} | "
                f"split={mean_spl:.3f} | count_err={mean_cnt:.3f}"
            )
            print(
                f"Epoch time: {epoch_time:.1f}s | Avg (last 5): {avg_epoch:.1f}s | "
                f"ETA: {eta_sec/60.0:.1f} min"
            )

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_precision": mean_prec,
                        "val_recall": mean_rec,
                        "val_f1": mean_f1,
                        "val_merge": mean_mer,
                        "val_split": mean_spl,
                        "val_count_err": mean_cnt,
                        "epoch_time": epoch_time,
                        "best_val_f1": best_f1,
                    },
                    step=epoch,
                )

        history_path, history_plot_path = save_history(run_dir, history)
        final_ckpt_for_test = best_path if os.path.exists(best_path) else last_path

        test_eval = None
        try:
            test_eval = run_test_segmentation(
                cfg=cfg, run_dir=run_dir, ckpt_path=final_ckpt_for_test, device=device
            )
            if test_eval is not None:
                print("Test segmentation metrics:")
                for k, v in sorted(test_eval.get("metrics", {}).items()):
                    print(f"  {k}: {v}")
        except Exception as exc:
            print(f"[warn] test segmentation failed: {exc}")
            if use_mlflow:
                mlflow.log_text(str(exc), "test_seg/error.txt")

        if use_mlflow:
            mlflow.log_metric("best_epoch", float(best_epoch), step=epochs)
            mlflow.log_metric("best_val_f1", float(best_f1), step=epochs)
            log_mlflow_artifacts_if_enabled(
                use_mlflow=use_mlflow,
                args=args,
                run_dir=run_dir,
                history_path=history_path,
                history_plot_path=history_plot_path,
                best_path=best_path,
                last_path=last_path,
                test_eval=test_eval,
                epochs=epochs,
            )


if __name__ == "__main__":
    main()

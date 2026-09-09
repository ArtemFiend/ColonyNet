from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from skimage.segmentation import find_boundaries

from colonyseg.metrics.instance_metrics import instance_scores
from colonyseg.models.colonymet import ColonyNet
from colonyseg.post.watershed import postprocess_watershed
from colonyseg.utils import ensure_dir


def detect_petri_circle(
    img_rgb: np.ndarray,
    min_r_frac: float = 0.35,
    max_r_frac: float = 0.55,
    center_tol: float = 0.25,
) -> Optional[tuple[int, int, int]]:
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

    if r < min_r or r > max_r:
        return None
    cx0, cy0 = w // 2, h // 2
    max_off = center_tol * min(h, w)
    if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 > max_off:
        return None
    return cx, cy, r


def crop_petri(
    img_rgb: np.ndarray,
    pad: float = 0.02,
    mask_outside: bool = True,
    min_r_frac: float = 0.35,
    max_r_frac: float = 0.55,
    center_tol: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = img_rgb.shape[:2]
    circ = detect_petri_circle(
        img_rgb, min_r_frac=min_r_frac, max_r_frac=max_r_frac, center_tol=center_tol
    )
    if circ is None:
        return img_rgb.copy(), {
            "status": "no_circle",
            "bbox": [0, 0, w, h],
            "orig_hw": [h, w],
            "mask_outside": bool(mask_outside),
        }

    cx, cy, r = circ
    r = int(r * (1.0 + pad))
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)

    crop = img_rgb[y1:y2, x1:x2].copy()
    if mask_outside:
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)
        crop[~mask] = 0

    meta = {
        "status": "cropped",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "circle": [int(cx), int(cy), int(r)],
        "orig_hw": [h, w],
        "mask_outside": bool(mask_outside),
    }
    return crop, meta


def overlay_boundaries(img_rgb: np.ndarray, labels: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = img_rgb.copy()
    b = find_boundaries(labels, mode="outer")
    out[b] = color
    return out


def _natural_key(name: str) -> list[Any]:
    parts = re.split(r"(\d+)", name.lower())
    key: list[Any] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return key


def _read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_mask_any(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(str(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m


def load_reference_instances(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.is_file():
        labels = _read_mask_any(p).astype(np.int32)
        if labels.max() <= 1:
            labels = (labels > 0).astype(np.int32)
        return labels, {"source": str(p), "type": "file", "instances": int(labels.max())}

    files = [q for q in p.iterdir() if q.is_file() and q.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    files = sorted(files, key=lambda q: _natural_key(q.name))
    if not files:
        raise RuntimeError(f"No masks found in: {p}")

    first = _read_mask_any(files[0])
    h, w = first.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    overlap_px = 0

    for idx, mp in enumerate(files, start=1):
        m = _read_mask_any(mp)
        if m.shape[:2] != (h, w):
            raise RuntimeError(f"Mask shape mismatch: {mp}")
        fg = m > 0
        overlap_px += int(np.count_nonzero((labels > 0) & fg))
        labels[fg] = idx

    return labels, {
        "source": str(p),
        "type": "dir",
        "mask_files": len(files),
        "instances": int(labels.max()),
        "overlap_px": int(overlap_px),
    }


def _apply_crop_to_instances(inst: np.ndarray, crop_meta: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = crop_meta["bbox"]
    crop = inst[y1:y2, x1:x2].copy()

    if crop_meta.get("status") == "cropped" and crop_meta.get("mask_outside", False):
        cx, cy, r = crop_meta["circle"]
        yy, xx = np.ogrid[y1:y2, x1:x2]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * r)
        crop[~mask] = 0
    return crop


def _binary_scores(gt_fg: np.ndarray, pr_fg: np.ndarray) -> dict[str, float]:
    gt = gt_fg.astype(bool)
    pr = pr_fg.astype(bool)
    tp = int(np.count_nonzero(gt & pr))
    fp = int(np.count_nonzero((~gt) & pr))
    fn = int(np.count_nonzero(gt & (~pr)))
    tn = int(np.count_nonzero((~gt) & (~pr)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    dice = 2.0 * tp / max(1, 2 * tp + fp + fn)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "semantic_precision": float(precision),
        "semantic_recall": float(recall),
        "semantic_f1": float(f1),
        "semantic_iou": float(iou),
        "semantic_dice": float(dice),
        "semantic_acc": float(acc),
    }


def _save_panel(
    out_path: Path,
    img_rgb: np.ndarray,
    img_petri: np.ndarray,
    pred_overlay: np.ndarray,
    pred_labels: np.ndarray,
    gt_overlay: Optional[np.ndarray] = None,
    agreement_rgb: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if gt_overlay is None:
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].set_title("Original")
        ax[0].imshow(img_rgb)
        ax[1].set_title("Petri crop")
        ax[1].imshow(img_petri)
        ax[2].set_title("Pred boundaries")
        ax[2].imshow(pred_overlay)
        ax[3].set_title("Pred instances")
        ax[3].imshow(pred_labels, cmap="nipy_spectral")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].set_title("Original")
    ax[0, 0].imshow(img_rgb)
    ax[0, 1].set_title("Petri crop")
    ax[0, 1].imshow(img_petri)
    ax[0, 2].set_title("Pred boundaries")
    ax[0, 2].imshow(pred_overlay)
    ax[1, 0].set_title("GT boundaries")
    ax[1, 0].imshow(gt_overlay)
    ax[1, 1].set_title("Agreement (TP/FP/FN)")
    ax[1, 1].imshow(agreement_rgb)
    ax[1, 2].set_title("Pred instances")
    ax[1, 2].imshow(pred_labels, cmap="nipy_spectral")
    for row in ax:
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_checkpoint_on_test_image(
    ckpt_path: str,
    image_path: str,
    out_dir: str,
    reference_path: Optional[str] = None,
    iou_thr: float = 0.5,
    petri_pad: float = 0.02,
    petri_mask_outside: bool = True,
    petri_min_r_frac: float = 0.35,
    petri_max_r_frac: float = 0.55,
    petri_center_tol: float = 0.25,
    device: Optional[str] = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    out_root = Path(out_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    backbone_id = cfg.get("model", {}).get("backbone_id", "nvidia/mit-b2")
    fpn_dim = int(cfg.get("model", {}).get("fpn_dim", 256))
    img_size = int(cfg.get("data", {}).get("img_size", 512))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = ColonyNet(backbone_id=backbone_id, fpn_dim=fpn_dim).to(dev)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    img_rgb = _read_image_rgb(Path(image_path))
    img_petri, crop_meta = crop_petri(
        img_rgb,
        pad=petri_pad,
        mask_outside=petri_mask_outside,
        min_r_frac=petri_min_r_frac,
        max_r_frac=petri_max_r_frac,
        center_tol=petri_center_tol,
    )
    img_rs = cv2.resize(img_petri, (img_size, img_size), interpolation=cv2.INTER_AREA)

    x = torch.from_numpy(img_rs).float().permute(2, 0, 1) / 255.0
    x = x.unsqueeze(0).to(dev)

    with torch.no_grad():
        pred = model(x)
        sem_p = torch.sigmoid(pred["sem"]).cpu().numpy()[0, 0]
        cen_p = torch.sigmoid(pred["center"]).cpu().numpy()[0, 0]
        bnd_p = torch.sigmoid(pred["boundary"]).cpu().numpy()[0, 0]

    h_in, w_in = img_rs.shape[:2]
    h_out, w_out = sem_p.shape
    scale = h_in / float(h_out)

    sem_hi = cv2.resize(sem_p.astype(np.float32), (w_in, h_in), interpolation=cv2.INTER_LINEAR)
    cen_hi = cv2.resize(cen_p.astype(np.float32), (w_in, h_in), interpolation=cv2.INTER_LINEAR)
    bnd_hi = cv2.resize(bnd_p.astype(np.float32), (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    post_cfg = cfg.get("post", {})
    min_distance_hi = max(1, int(round(float(post_cfg.get("min_distance", 6)) * scale)))
    area_min_hi = max(1, int(round(float(post_cfg.get("area_min", 20)) * (scale**2))))
    area_max_hi = int(round(float(post_cfg.get("area_max", 200000)) * (scale**2)))

    labels_up = postprocess_watershed(
        sem_hi,
        cen_hi,
        bnd_hi,
        t_sem=float(post_cfg.get("t_sem", 0.5)),
        t_center=float(post_cfg.get("t_center", 0.35)),
        min_distance=min_distance_hi,
        lambda_boundary=float(post_cfg.get("lambda_boundary", 3.0)),
        area_min=area_min_hi,
        area_max=area_max_hi,
    )

    pred_overlay = overlay_boundaries(img_rs, labels_up, color=(0, 255, 0))

    pred_labels_path = out_root / "pred_labels.png"
    pred_overlay_path = out_root / "pred_overlay.png"
    cv2.imwrite(str(pred_labels_path), labels_up.astype(np.uint16))
    cv2.imwrite(str(pred_overlay_path), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

    metrics: dict[str, float] = {
        "pred_instances": float(labels_up.max()),
    }
    artifacts: dict[str, str] = {
        "pred_labels": str(pred_labels_path),
        "pred_overlay": str(pred_overlay_path),
    }
    reference_meta: dict[str, Any] | None = None
    gt_overlay: Optional[np.ndarray] = None
    agreement_rgb: Optional[np.ndarray] = None

    if reference_path is not None:
        gt_full, reference_meta = load_reference_instances(reference_path)
        if gt_full.shape[:2] != img_rgb.shape[:2]:
            gt_full = cv2.resize(
                gt_full.astype(np.int32),
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_crop = _apply_crop_to_instances(gt_full, crop_meta)
        gt_rs = cv2.resize(gt_crop.astype(np.int32), (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        gt_labels_path = out_root / "gt_labels.png"
        cv2.imwrite(str(gt_labels_path), gt_rs.astype(np.uint16))
        artifacts["gt_labels"] = str(gt_labels_path)

        m_inst = instance_scores(gt_rs, labels_up, iou_thr=float(iou_thr))
        m_sem = _binary_scores(gt_rs > 0, labels_up > 0)
        for k, v in m_inst.items():
            metrics[f"inst_{k}"] = float(v)
        metrics.update(m_sem)

        gt_overlay = overlay_boundaries(img_rs, gt_rs, color=(255, 0, 0))
        gt_overlay_path = out_root / "gt_overlay.png"
        cv2.imwrite(str(gt_overlay_path), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
        artifacts["gt_overlay"] = str(gt_overlay_path)

        gt_fg = gt_rs > 0
        pr_fg = labels_up > 0
        agreement_rgb = np.zeros((*gt_fg.shape, 3), dtype=np.uint8)
        agreement_rgb[gt_fg & pr_fg] = (0, 255, 0)  # TP
        agreement_rgb[(~gt_fg) & pr_fg] = (255, 0, 0)  # FP
        agreement_rgb[gt_fg & (~pr_fg)] = (0, 0, 255)  # FN
        agreement_path = out_root / "agreement_map.png"
        cv2.imwrite(str(agreement_path), cv2.cvtColor(agreement_rgb, cv2.COLOR_RGB2BGR))
        artifacts["agreement_map"] = str(agreement_path)

    panel_path = out_root / "comparison_panel.png"
    _save_panel(
        panel_path,
        img_rgb=img_rgb,
        img_petri=img_petri,
        pred_overlay=pred_overlay,
        pred_labels=labels_up,
        gt_overlay=gt_overlay,
        agreement_rgb=agreement_rgb,
    )
    artifacts["comparison_panel"] = str(panel_path)

    report = {
        "ckpt_path": str(ckpt_path),
        "image_path": str(image_path),
        "reference_path": str(reference_path) if reference_path is not None else None,
        "crop": crop_meta,
        "metrics": metrics,
        "reference_meta": reference_meta,
        "img_size": img_size,
        "device": dev,
    }
    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    artifacts["report"] = str(report_path)

    return {
        "metrics": metrics,
        "artifacts": artifacts,
        "report_path": str(report_path),
    }

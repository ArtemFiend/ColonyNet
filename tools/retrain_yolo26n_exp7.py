"""Retrain YOLO26n-seg for MLflow experiment 7 and log test metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from ultralytics import YOLO, settings as yolo_settings


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "colony_yolo_seg_736_4"
DATA_YAML = Path("cropped_736_aug_leaky/dataset/data.yaml")
CKPT = "yolo26n-seg.pt"
RUN_NAME = "yolo26n-seg_cropped736_offline_aug"
MLFLOW_PROJECT = "runs/segment/runs/colony_seg_mlflow_736"
RESULTS_CSV_NAME = "results_736_4.csv"

TRAIN_EPOCHS = 50
TRAIN_BATCH = 8
TRAIN_IMGSZ = 736
TRAIN_PATIENCE = 80
TRAIN_DEVICE = None

PRED_CONF = 0.25
PRED_IOU = 0.7
PRED_MAX_DET = 300
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        value_float = to_float(value)
        if math.isfinite(value_float):
            out[key] = value_float
    return out


def summarize_section(section: Any, prefix: str) -> dict[str, float]:
    if section is None:
        return {
            f"precision_{prefix}": float("nan"),
            f"recall_{prefix}": float("nan"),
            f"mAP50_{prefix}": float("nan"),
            f"mAP50_95_{prefix}": float("nan"),
        }
    return {
        f"precision_{prefix}": to_float(getattr(section, "mp", float("nan"))),
        f"recall_{prefix}": to_float(getattr(section, "mr", float("nan"))),
        f"mAP50_{prefix}": to_float(getattr(section, "map50", float("nan"))),
        f"mAP50_95_{prefix}": to_float(getattr(section, "map", float("nan"))),
    }


def safe_metric_name(name: str) -> str:
    safe = str(name).strip().replace(" ", "_")
    for bad in ["(", ")", "/", "\\", ":", ",", "|", "-", "."]:
        safe = safe.replace(bad, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def metric_aliases(raw_name: str) -> list[str]:
    aliases = {
        "metrics/precision(B)": "metrics/precisionB",
        "metrics/recall(B)": "metrics/recallB",
        "metrics/mAP50(B)": "metrics/mAP50B",
        "metrics/mAP50-95(B)": "metrics/mAP50-95B",
        "metrics/precision(M)": "metrics/precisionM",
        "metrics/recall(M)": "metrics/recallM",
        "metrics/mAP50(M)": "metrics/mAP50M",
        "metrics/mAP50-95(M)": "metrics/mAP50-95M",
    }
    return [aliases[raw_name]] if raw_name in aliases else []


def log_training_csv_metrics(results_csv: Path) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    for i, row in df.iterrows():
        step_raw = row.get("epoch", i)
        step_float = to_float(step_raw)
        step = int(step_float) if math.isfinite(step_float) else int(i)
        for key, value in row.items():
            value_float = to_float(value)
            if not math.isfinite(value_float):
                continue
            mlflow.log_metric(f"train_{safe_metric_name(key)}", value_float, step=step)
            for alias in metric_aliases(str(key).strip()):
                mlflow.log_metric(alias, value_float, step=step)


def resolve_test_split_dirs(data_yaml_path: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    test_raw = data.get("test")
    if not test_raw:
        raise KeyError("`test` path is missing in data.yaml")

    test_img_dir = Path(str(test_raw).strip().strip('"').strip("'"))
    if not test_img_dir.is_absolute():
        test_img_dir = (data_yaml_path.parent / test_img_dir).resolve()

    split_name = test_img_dir.name
    label_candidates = [data_yaml_path.parent / "labels" / split_name]
    if test_img_dir.parent.name == "images":
        label_candidates.append(test_img_dir.parent.parent / "labels" / split_name)

    replaced = Path(
        str(test_img_dir)
        .replace("\\images\\", "\\labels\\")
        .replace("/images/", "/labels/")
    )
    label_candidates.append(replaced)

    for candidate in label_candidates:
        if candidate.exists():
            return test_img_dir, candidate
    return test_img_dir, label_candidates[0]


def read_yolo_seg_polygons(label_path: Path) -> list[np.ndarray]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    polygons: list[np.ndarray] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = []
        values = parts[1:]
        if len(values) % 2:
            values = values[:-1]
        for i in range(0, len(values), 2):
            try:
                coords.append((float(values[i]), float(values[i + 1])))
            except Exception:
                coords = []
                break
        if len(coords) >= 3:
            polygons.append(np.asarray(coords, dtype=np.float32))
    return polygons


def polygons_norm_to_mask(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = poly.copy()
        pts[:, 0] = np.clip(pts[:, 0] * width, 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1] * height, 0, height - 1)
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def result_to_pred_mask(result: Any, height: int, width: int) -> tuple[np.ndarray, int]:
    mask = np.zeros((height, width), dtype=bool)
    if result.masks is None or result.masks.data is None:
        return mask, 0

    data = result.masks.data
    try:
        masks_np = data.detach().cpu().numpy()
    except Exception:
        masks_np = np.asarray(data)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    pred_count = int(masks_np.shape[0])
    for m in masks_np:
        m_bool = m > 0.5
        if m_bool.shape != (height, width):
            m_bool = cv2.resize(
                m_bool.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask |= m_bool
    return mask, pred_count


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    pred_sum = float(pred.sum(dtype=np.float64))
    gt_sum = float(gt.sum(dtype=np.float64))
    if pred_sum == 0.0 and gt_sum == 0.0:
        return 1.0
    inter = float(np.logical_and(pred, gt).sum(dtype=np.float64))
    return float((2.0 * inter + eps) / (pred_sum + gt_sum + eps))


def compute_test_custom_metrics(best_model: YOLO, data_yaml: Path, imgsz: int) -> tuple[dict[str, float], pd.DataFrame]:
    test_img_dir, test_lbl_dir = resolve_test_split_dirs(data_yaml)
    test_images = [p for p in sorted(test_img_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not test_images:
        raise RuntimeError(f"No test images in: {test_img_dir}")

    rows = []
    pred_iter = best_model.predict(
        source=str(test_img_dir),
        imgsz=imgsz,
        conf=PRED_CONF,
        iou=PRED_IOU,
        max_det=PRED_MAX_DET,
        stream=True,
        verbose=False,
        save=False,
    )
    for result in pred_iter:
        image_path = Path(result.path)
        height, width = map(int, result.orig_shape)
        label_path = test_lbl_dir / f"{image_path.stem}.txt"
        gt_polys = read_yolo_seg_polygons(label_path)
        gt_mask = polygons_norm_to_mask(gt_polys, height, width)
        pred_mask, pred_count = result_to_pred_mask(result, height, width)
        gt_count = int(len(gt_polys))
        count_error = int(pred_count - gt_count)
        abs_error = abs(count_error)
        ape = (abs_error / gt_count) if gt_count > 0 else np.nan
        rows.append(
            {
                "image": image_path.name,
                "label_exists": int(label_path.exists()),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "count_error": count_error,
                "count_abs_error": abs_error,
                "count_ape": float(ape),
                "dice": dice_score(pred_mask, gt_mask),
            }
        )

    df = pd.DataFrame(rows)
    sq = np.square(df["count_error"].to_numpy(dtype=np.float64))
    ape_valid = df["count_ape"].dropna()
    metrics = {
        "dice_M_mean": float(df["dice"].mean()),
        "dice_M_median": float(df["dice"].median()),
        "mae_count": float(df["count_abs_error"].mean()),
        "rmse_count": float(np.sqrt(sq.mean())),
        "mape_count_nonzero": float(ape_valid.mean() * 100.0) if len(ape_valid) else np.nan,
        "test_images_eval": int(len(df)),
        "test_images_missing_labels": int((df["label_exists"] == 0).sum()),
    }
    return metrics, df


def ensure_experiment(name: str):
    client = MlflowClient()
    for exp in client.search_experiments(view_type=ViewType.ALL):
        if exp.name == name:
            if getattr(exp, "lifecycle_stage", "active") != "active":
                client.restore_experiment(exp.experiment_id)
            return mlflow.set_experiment(name)
    return mlflow.set_experiment(name)


def log_artifact_if_exists(path: Path, artifact_path: str) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--device", default=TRAIN_DEVICE)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--weights", default=CKPT)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    data_yaml = args.data
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    previous_mlflow_setting = yolo_settings.get("mlflow")
    yolo_settings.update({"mlflow": False})

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = ensure_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}", flush=True)
    print(f"Experiment: {experiment.name} ({experiment.experiment_id})", flush=True)
    print(f"Run name: {RUN_NAME}", flush=True)

    train_run_id = None
    train_dir = None
    best_w = None
    test_dir = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        with mlflow.start_run(run_name=RUN_NAME) as active_run:
            train_run_id = active_run.info.run_id
            mlflow.set_tags(
                {
                    "task": "segment_train_retrain_n",
                    "model_family": "yolo26",
                    "model_size": "n",
                    "retrain": "true",
                    "source": "tools/retrain_yolo26n_exp7.py",
                }
            )
            mlflow.log_params(
                {
                    "model": args.weights,
                    "data_yaml": str(data_yaml),
                    "imgsz": TRAIN_IMGSZ,
                    "epochs": int(args.epochs),
                    "batch": int(args.batch),
                    "patience": TRAIN_PATIENCE,
                    "project": MLFLOW_PROJECT,
                    "run_name": RUN_NAME,
                    "tracking_uri": mlflow.get_tracking_uri(),
                    "pred_conf": PRED_CONF,
                    "pred_iou": PRED_IOU,
                    "pred_max_det": PRED_MAX_DET,
                }
            )

            model = YOLO(args.weights)
            train_results = model.train(
                data=str(data_yaml),
                task="segment",
                imgsz=TRAIN_IMGSZ,
                epochs=int(args.epochs),
                batch=int(args.batch),
                patience=TRAIN_PATIENCE,
                device=args.device,
                project=MLFLOW_PROJECT,
                name=RUN_NAME,
                exist_ok=False,
                plots=True,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                fliplr=0.0,
                flipud=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                erasing=0.0,
            )

            train_dir = Path(train_results.save_dir)
            raw_results_csv = train_dir / "results.csv"
            results_csv = train_dir / RESULTS_CSV_NAME
            if raw_results_csv.exists():
                shutil.copy2(raw_results_csv, results_csv)
            elif not results_csv.exists():
                results_csv = raw_results_csv

            best_w = train_dir / "weights" / "best.pt"
            last_w = train_dir / "weights" / "last.pt"
            if not best_w.exists():
                raise FileNotFoundError(best_w)

            log_training_csv_metrics(results_csv)
            mlflow.log_params(
                {
                    "ultralytics_train_dir": str(train_dir.resolve()),
                    "best_weights": str(best_w.resolve()),
                    "last_weights": str(last_w.resolve()) if last_w.exists() else "",
                }
            )

            for artifact in [
                train_dir / "args.yaml",
                train_dir / "results.png",
                train_dir / "confusion_matrix.png",
                train_dir / "confusion_matrix_normalized.png",
                train_dir / "BoxF1_curve.png",
                train_dir / "BoxP_curve.png",
                train_dir / "BoxPR_curve.png",
                train_dir / "BoxR_curve.png",
                train_dir / "MaskF1_curve.png",
                train_dir / "MaskP_curve.png",
                train_dir / "MaskPR_curve.png",
                train_dir / "MaskR_curve.png",
                results_csv,
            ]:
                log_artifact_if_exists(artifact, "train")
            log_artifact_if_exists(best_w, "weights")
            log_artifact_if_exists(last_w, "weights")

        if best_w is None or train_dir is None:
            raise RuntimeError("Training did not produce weights")

        best_model = YOLO(str(best_w))
        test_results = best_model.val(
            data=str(data_yaml),
            split="test",
            imgsz=TRAIN_IMGSZ,
            project=MLFLOW_PROJECT,
            name=f"{RUN_NAME}_test_{timestamp}",
            plots=True,
        )
        test_dir = Path(test_results.save_dir)
        metrics_summary: dict[str, Any] = {}
        metrics_summary.update(summarize_section(getattr(test_results, "box", None), "B"))
        metrics_summary.update(summarize_section(getattr(test_results, "seg", None), "M"))
        metrics_summary["fitness"] = to_float(getattr(test_results, "fitness", float("nan")))

        custom_metrics, per_image_df = compute_test_custom_metrics(best_model, data_yaml, TRAIN_IMGSZ)
        metrics_summary.update(custom_metrics)

        test_json = test_dir / "test_metrics_summary.json"
        test_json.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
        per_image_csv = test_dir / "test_per_image_metrics.csv"
        per_image_df.to_csv(per_image_csv, index=False, encoding="utf-8")

        with mlflow.start_run(run_name=f"{RUN_NAME}_test_eval_{timestamp}") as eval_run:
            mlflow.set_tags(
                {
                    "task": "segment_test_eval",
                    "model_family": "yolo26",
                    "model_size": "n",
                    "retrain": "true",
                    "source_train_run_id": train_run_id or "",
                    "source": "tools/retrain_yolo26n_exp7.py",
                }
            )
            mlflow.log_params(
                {
                    "model": args.weights,
                    "data_yaml": str(data_yaml),
                    "best_weights": str(best_w.resolve()),
                    "ultralytics_train_dir": str(train_dir.resolve()),
                    "ultralytics_test_dir": str(test_dir.resolve()),
                    "source_train_run_id": train_run_id or "",
                    "imgsz": TRAIN_IMGSZ,
                    "pred_conf": PRED_CONF,
                    "pred_iou": PRED_IOU,
                    "pred_max_det": PRED_MAX_DET,
                }
            )
            mlflow.log_metrics(finite_metrics(metrics_summary))
            if test_dir.exists():
                mlflow.log_artifacts(str(test_dir), artifact_path="test")

            print(f"Train run id: {train_run_id}", flush=True)
            print(f"Eval run id: {eval_run.info.run_id}", flush=True)
            print(f"Best weights: {best_w}", flush=True)
            print(f"Test metrics: {json.dumps(metrics_summary, ensure_ascii=False, indent=2)}", flush=True)
    finally:
        if previous_mlflow_setting is not None:
            yolo_settings.update({"mlflow": previous_mlflow_setting})


if __name__ == "__main__":
    main()

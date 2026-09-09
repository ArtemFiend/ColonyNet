# Auto-generated standalone full pipeline from notebooks/colony_anomaly_detection_yolo_x_improved.ipynb.
# Runs: Petri detector -> crop/resize 736x736 -> YOLO26x-seg -> colony anomaly analysis.

import os
import math
import json
import colorsys
import warnings
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy import stats
from scipy.ndimage import binary_fill_holes

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from skimage.measure import regionprops, label, find_contours
from skimage.morphology import binary_dilation, disk
from skimage.feature import local_binary_pattern
from skimage import exposure

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

try:
    from .runtime import configure_offline
except ImportError:
    from runtime import configure_offline

configure_offline()
from ultralytics import YOLO, settings
# Change only this process; do not overwrite the user's training settings file.
dict.update(settings, {key: False for key in (
    "sync", "mlflow", "wandb", "clearml", "comet", "dvc", "raytune", "tensorboard"
) if key in settings})

warnings.filterwarnings("ignore")

PIPELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PIPELINE_DIR / "models"
DEFAULT_PETRI_DETECTOR_WEIGHTS = MODELS_DIR / "petri_detector_yolo26s_best.pt"
DEFAULT_COLONY_SEG_WEIGHTS = MODELS_DIR / "colony_yolo26x_seg_best.pt"

@dataclass
class AnomalyDetectionConfig:
    # Preprocessing mode: already_cropped_736 / detect_petri / resize_only
    preprocess_mode: str = "already_cropped_736"
    use_petri_detector: bool = False

    # Optional Petri dish detector for raw photos
    petri_detector_weights_path: str = str(DEFAULT_PETRI_DETECTOR_WEIGHTS)
    petri_det_conf: float = 0.25
    petri_det_iou: float = 0.45
    petri_det_max_det: int = 10
    petri_bbox_margin: float = 0.01

    # YOLO26x-seg colony segmentation model
    model_dir: str = str(MODELS_DIR)
    model_weights_path: str | None = str(DEFAULT_COLONY_SEG_WEIGHTS)

    input_path: str = r"data\colonies_736"
    output_dir: str = r"outputs\colony_anomaly_detection"

    # YOLO inference
    imgsz: int = 736
    conf: float = 0.10
    iou: float = 0.65
    max_det: int = 1000
    retina_masks: bool = True
    classes: tuple[int, ...] | None = None
    device: str | None = None

    # Mask filtering
    min_area: int = 25
    min_area_relative_to_median: float = 0.03
    max_area_fraction: float = 0.10
    max_area_relative_to_median: float = 10.0
    min_yolo_conf_for_analysis: float = 0.05
    edge_margin_px: int = 2
    exclude_border_touching_from_anomaly: bool = True
    exclude_technical_from_anomaly: bool = True
    enforce_non_overlap: bool = True
    suspicious_aspect_ratio_threshold: float = 4.0

    # Mask cleanup
    clean_masks: bool = True
    keep_largest_component: bool = True
    fill_mask_holes: bool = True
    min_component_area: int = 10

    # Mask smoothing before anomaly analysis
    smooth_masks_before_anomaly: bool = True
    mask_smoothing_sigma: float = 1.6
    mask_smoothing_threshold: float = 0.50

    # Local background and neighbors
    local_background_radius: int = 12
    local_density_radius: float = 35.0
    neighbor_k: int = 5

    # Texture
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2)
    lbp_p: int = 8
    lbp_r: int = 1

    # Histograms
    use_histogram_features: bool = True
    hist_bins: int = 32
    hist_use_lab: bool = True
    hist_use_intensity: bool = True
    hist_use_contrast: bool = True

    # Morphotypes
    use_morphotype_clustering: bool = True
    morphotype_method: str = "kmeans"
    max_morphotypes: int = 4
    min_morphotype_size: int = 5
    rare_morphotype_fraction: float = 0.08
    min_morphotype_silhouette: float = 0.20

    # Anomaly scoring
    lof_neighbors: int = 20
    contamination: float = 0.05
    dbscan_eps: float = 1.8
    dbscan_min_samples: int | None = None
    use_group_scores: bool = True
    use_weighted_final_score: bool = True
    random_state: int = 42

    # Objective selection controls
    min_independent_evidence_count: int = 2
    min_reliability_score: float = 0.50
    min_final_score_raw: float = 0.65
    min_absolute_evidence_strength: float = 0.35
    min_consensus_score: float = 0.30
    clip_relative_features: float = 5.0
    neighbor_difference_weight: float = 0.05
    spatial_context_weight: float = 0.03
    morphotype_weight: float = 0.08

    # Stability / ablation
    use_ablation_stability: bool = True
    use_perturbation_stability: bool = True
    perturbation_stability_variants: tuple[str, ...] = (
        "brightness_plus_5pct",
        "contrast_minus_5pct",
        "slight_blur",
    )
    min_perturbation_stability_score: float = 0.66
    min_perturbation_stability_trials: int = 2
    perturbation_stability_match_distance: float = 20.0

    # Selection
    top_k: int | None = None
    top_percent: float | None = 0.02
    min_score: float | None = None
    visual_highlight_percent: float | None = 0.20
    visual_highlight_max_count: int | None = 20
    visual_highlight_min_count: int = 0
    visual_highlight_max_per_neighborhood: int = 2
    visual_highlight_neighborhood_radius_factor: float = 2.5
    visual_highlight_min_distance_px: float | None = None
    visual_highlight_max_per_morphotype: int = 2
    visual_highlight_edge_band_diameters: float = 2.0
    visual_highlight_max_edge_fraction: float | None = 0.25
    visual_highlight_edge_score_penalty: float = 0.20
    visual_highlight_max_review_segmentation_fraction: float | None = 0.20
    visual_highlight_max_review_segmentation_count: int | None = 4

    # Saving
    save_visualizations: bool = True
    save_csv: bool = True
    save_xlsx: bool = True
    save_colony_crops: bool = True
    save_feature_space_plot: bool = True
    save_feature_correlation_report: bool = True


config = AnomalyDetectionConfig()

def find_best_yolo_weights(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates = [
        model_dir / "weights" / "best.pt",
        model_dir / "best.pt",
        model_dir / "weights" / "last.pt",
        model_dir / "last.pt",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    pt_files = [p for p in model_dir.rglob("*.pt") if p.is_file()]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in model directory: {model_dir}")

    pt_files = sorted(
        pt_files,
        key=lambda p: ("yolo26x" not in str(p).lower(), str(p).lower()),
    )

    best_named = [p for p in pt_files if "best" in p.name.lower()]
    if best_named:
        return best_named[0]

    return max(pt_files, key=lambda p: p.stat().st_mtime)

def collect_image_paths(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in image_exts:
            raise ValueError(f"File is not a supported image: {input_path}")
        return [input_path]

    image_paths = [
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts
    ]
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in directory {input_path} with extensions {sorted(image_exts)}"
        )

    return image_paths


def read_image_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # np.fromfile + imdecode works with Cyrillic paths on Windows, unlike cv2.imread.
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_to_config(image_rgb: np.ndarray, config: AnomalyDetectionConfig) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if (h, w) == (config.imgsz, config.imgsz):
        return image_rgb.copy()
    interpolation = cv2.INTER_AREA if max(h, w) > int(config.imgsz) else cv2.INTER_LINEAR
    return cv2.resize(image_rgb, (int(config.imgsz), int(config.imgsz)), interpolation=interpolation)


def detect_petri_and_resize(
    image_rgb: np.ndarray,
    detector_model,
    config: AnomalyDetectionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    meta: dict[str, Any] = {
        "original_shape": [int(h), int(w)],
        "preprocess_mode": config.preprocess_mode,
        "petri_detected": False,
        "detector_conf": np.nan,
        "crop_xyxy": [0, 0, int(w), int(h)],
        "detection_error": None,
    }

    crop = image_rgb

    if detector_model is not None:
        try:
            det_results = detector_model.predict(
                source=image_rgb,
                conf=float(config.petri_det_conf),
                iou=float(config.petri_det_iou),
                max_det=int(config.petri_det_max_det),
                device=config.device,
                verbose=False,
            )

            if det_results is not None and len(det_results) > 0:
                det_res = det_results[0]
                if det_res.boxes is not None and det_res.boxes.xyxy is not None and len(det_res.boxes.xyxy) > 0:
                    boxes = det_res.boxes.xyxy.cpu().numpy().astype(float)
                    confs = (
                        det_res.boxes.conf.cpu().numpy().astype(float)
                        if det_res.boxes.conf is not None
                        else np.full((boxes.shape[0],), np.nan, dtype=float)
                    )

                    best_idx = int(np.nanargmax(confs)) if confs.size and np.isfinite(confs).any() else 0
                    x1, y1, x2, y2 = boxes[best_idx]
                    conf_best = float(confs[best_idx]) if confs.size > best_idx else np.nan

                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    mx = int(round(bw * float(config.petri_bbox_margin)))
                    my = int(round(bh * float(config.petri_bbox_margin)))

                    x1i = max(0, int(np.floor(x1)) - mx)
                    y1i = max(0, int(np.floor(y1)) - my)
                    x2i = min(w, int(np.ceil(x2)) + mx)
                    y2i = min(h, int(np.ceil(y2)) + my)

                    if x2i > x1i and y2i > y1i:
                        crop_candidate = image_rgb[y1i:y2i, x1i:x2i]
                        if crop_candidate.size > 0:
                            crop = crop_candidate
                            meta["petri_detected"] = True
                            meta["detector_conf"] = conf_best
                            meta["crop_xyxy"] = [int(x1i), int(y1i), int(x2i), int(y2i)]
        except Exception as exc:
            meta["detection_error"] = str(exc)

    ch, cw = crop.shape[:2]
    prepared = resize_to_config(crop, config)
    meta["crop_shape"] = [int(ch), int(cw)]
    meta["prepared_shape"] = [int(prepared.shape[0]), int(prepared.shape[1])]
    return prepared, meta


def prepare_image_for_segmentation(
    image_rgb: np.ndarray,
    detector_model,
    config: AnomalyDetectionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(config.preprocess_mode).lower()
    h, w = image_rgb.shape[:2]

    if mode == "already_cropped_736":
        prepared = resize_to_config(image_rgb, config)
        return prepared, {
            "original_shape": [int(h), int(w)],
            "preprocess_mode": mode,
            "petri_detected": False,
            "crop_xyxy": [0, 0, int(w), int(h)],
            "crop_shape": [int(h), int(w)],
            "prepared_shape": [int(prepared.shape[0]), int(prepared.shape[1])],
        }

    if mode == "resize_only":
        prepared = resize_to_config(image_rgb, config)
        return prepared, {
            "original_shape": [int(h), int(w)],
            "preprocess_mode": mode,
            "petri_detected": False,
            "crop_xyxy": [0, 0, int(w), int(h)],
            "crop_shape": [int(h), int(w)],
            "prepared_shape": [int(prepared.shape[0]), int(prepared.shape[1])],
        }

    if mode == "detect_petri":
        if not config.use_petri_detector:
            raise ValueError("preprocess_mode='detect_petri' requires config.use_petri_detector=True")
        return detect_petri_and_resize(image_rgb, detector_model, config)

    raise ValueError(
        "Unsupported preprocess_mode. Use 'already_cropped_736', 'resize_only', or 'detect_petri'."
    )

def _component_count(mask_bool: np.ndarray) -> int:
    mask_bool = np.asarray(mask_bool).astype(bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return 0
    return int(len(regionprops(label(mask_bool.astype(np.uint8)))))


def clean_instance_mask(mask: np.ndarray, config: AnomalyDetectionConfig) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.asarray(mask).astype(bool)
    diagnostics: dict[str, Any] = {
        "original_area": float(mask.sum()) if mask.ndim == 2 else 0.0,
        "cleaned_area": 0.0,
        "area_loss_fraction": 1.0,
        "n_components_before": 0,
        "n_components_after": 0,
        "mask_fragment_after_overlap": False,
    }

    if mask.ndim != 2:
        return np.zeros_like(mask, dtype=bool), diagnostics

    diagnostics["n_components_before"] = _component_count(mask)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool), diagnostics

    if not bool(getattr(config, "clean_masks", True)):
        cleaned = mask.astype(bool)
        cleaned_area = float(cleaned.sum())
        diagnostics.update(
            cleaned_area=cleaned_area,
            area_loss_fraction=0.0,
            n_components_after=_component_count(cleaned),
            mask_fragment_after_overlap=bool(cleaned_area < float(config.min_component_area)),
        )
        return cleaned, diagnostics

    cleaned = mask.copy()

    if config.min_component_area and int(config.min_component_area) > 1:
        labeled = label(cleaned.astype(np.uint8))
        keep = np.zeros_like(cleaned, dtype=bool)
        for region in regionprops(labeled):
            if region.area >= int(config.min_component_area):
                keep[labeled == region.label] = True
        cleaned = keep

    if config.keep_largest_component and cleaned.any():
        labeled = label(cleaned.astype(np.uint8))
        regions = regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            cleaned = labeled == largest.label

    if config.fill_mask_holes and cleaned.any():
        try:
            cleaned = binary_fill_holes(cleaned).astype(bool)
        except Exception:
            pass

    original_area = max(float(diagnostics["original_area"]), 1.0)
    cleaned_area = float(cleaned.sum())
    area_loss_fraction = float(np.clip(1.0 - cleaned_area / original_area, 0.0, 1.0))
    diagnostics.update(
        cleaned_area=cleaned_area,
        area_loss_fraction=area_loss_fraction,
        n_components_after=_component_count(cleaned),
        mask_fragment_after_overlap=bool(
            cleaned_area < float(config.min_component_area)
            or (diagnostics["original_area"] >= float(config.min_area) and cleaned_area < 0.5 * diagnostics["original_area"])
        ),
    )
    return cleaned.astype(bool), diagnostics


MASK_SMOOTHING_DIAGNOSTIC_COLUMNS = [
    "colony_id",
    "mask_area_before_smoothing",
    "mask_area_after_smoothing",
    "mask_area_smoothing_delta_fraction",
    "mask_smoothing_sigma",
    "mask_smoothing_threshold",
    "mask_smoothing_used_fallback",
    "n_components_after_smoothing",
]


def smooth_single_mask_for_anomaly(
    mask: np.ndarray,
    sigma: float = 1.6,
    threshold: float = 0.50,
) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return mask_bool

    sigma = float(sigma)
    if sigma <= 0:
        return mask_bool.copy()

    mask_prob = cv2.GaussianBlur(
        mask_bool.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    smoothed = mask_prob >= float(threshold)
    if not smoothed.any():
        return mask_bool.copy()
    return smoothed.astype(bool)


def build_mask_smoothing_diagnostics(
    masks,
    sigma: float,
    threshold: float,
    applied: bool,
) -> pd.DataFrame:
    masks_list = normalize_masks_input(masks)
    rows = []
    for colony_id, mask in enumerate(masks_list, start=1):
        area = float(np.asarray(mask).astype(bool).sum())
        rows.append(
            {
                "colony_id": int(colony_id),
                "mask_area_before_smoothing": area,
                "mask_area_after_smoothing": area,
                "mask_area_smoothing_delta_fraction": 0.0,
                "mask_smoothing_sigma": float(sigma) if applied else 0.0,
                "mask_smoothing_threshold": float(threshold) if applied else 1.0,
                "mask_smoothing_used_fallback": False,
                "n_components_after_smoothing": _component_count(mask),
            }
        )
    return pd.DataFrame(rows, columns=MASK_SMOOTHING_DIAGNOSTIC_COLUMNS)


def smooth_masks_before_anomaly(
    masks,
    config: AnomalyDetectionConfig,
    sigma: float = 1.6,
    threshold: float = 0.50,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    smoothed_masks: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []

    for colony_id, mask in enumerate(normalize_masks_input(masks), start=1):
        mask_bool = np.asarray(mask).astype(bool)
        before_area = float(mask_bool.sum())

        smoothed = smooth_single_mask_for_anomaly(mask_bool, sigma=sigma, threshold=threshold)
        cleaned, diag = clean_instance_mask(smoothed, config)
        used_fallback = False
        if not cleaned.any() and mask_bool.any():
            cleaned = mask_bool.copy()
            used_fallback = True

        after_area = float(cleaned.sum())
        delta_fraction = (after_area - before_area) / max(before_area, 1.0)

        smoothed_masks.append(cleaned.astype(bool))
        diagnostics.append(
            {
                "colony_id": int(colony_id),
                "mask_area_before_smoothing": before_area,
                "mask_area_after_smoothing": after_area,
                "mask_area_smoothing_delta_fraction": float(delta_fraction),
                "mask_smoothing_sigma": float(sigma),
                "mask_smoothing_threshold": float(threshold),
                "mask_smoothing_used_fallback": bool(used_fallback),
                "n_components_after_smoothing": int(diag.get("n_components_after", 0)),
            }
        )

    return smoothed_masks, pd.DataFrame(diagnostics, columns=MASK_SMOOTHING_DIAGNOSTIC_COLUMNS)


def make_masks_non_overlapping(
    masks: list[np.ndarray],
    confs: np.ndarray,
    boxes: np.ndarray,
    config: AnomalyDetectionConfig,
    mask_meta: list[dict[str, Any]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if mask_meta is None:
        mask_meta = []
        for m in masks:
            area = float(np.asarray(m).sum())
            mask_meta.append(
                {
                    "mask_area_raw": area,
                    "mask_area_cleaned": area,
                    "mask_area_after_overlap": area,
                    "mask_area_loss_fraction": 0.0,
                    "original_area": area,
                    "cleaned_area": area,
                    "area_loss_fraction": 0.0,
                    "n_components_before": _component_count(m),
                    "n_components_after": _component_count(m),
                    "mask_fragment_after_overlap": False,
                }
            )

    if len(masks) <= 1 or not config.enforce_non_overlap:
        for i, m in enumerate(masks):
            area = float(np.asarray(m).sum())
            before = float(mask_meta[i].get("mask_area_cleaned", area))
            loss = 1.0 - area / max(before, 1.0)
            mask_meta[i]["mask_area_after_overlap"] = area
            mask_meta[i]["mask_area_loss_fraction"] = float(np.clip(loss, 0.0, 1.0))
            mask_meta[i]["mask_fragment_after_overlap"] = bool(
                mask_meta[i].get("mask_fragment_after_overlap", False)
                or area < float(config.min_component_area)
                or (before >= float(config.min_area) and area < 0.5 * before)
            )
        keep_indices = [i for i, m in enumerate(masks) if np.asarray(m).any()]
        return (
            [masks[i] for i in keep_indices],
            confs[keep_indices] if len(keep_indices) else np.array([], dtype=float),
            boxes[keep_indices] if len(keep_indices) else np.empty((0, 4), dtype=float),
            [mask_meta[i] for i in keep_indices],
        )

    h, w = masks[0].shape
    conf_priority = np.nan_to_num(confs, nan=-np.inf)
    priority_indices = np.argsort(conf_priority)[::-1]
    occupied = np.zeros((h, w), dtype=bool)
    exclusive_masks: list[np.ndarray] = [np.zeros((h, w), dtype=bool) for _ in range(len(masks))]

    for idx in priority_indices:
        before_area = float(np.asarray(masks[idx]).sum())
        unique_pixels = masks[idx] & (~occupied)
        if config.clean_masks:
            unique_pixels, diag = clean_instance_mask(unique_pixels, config)
            mask_meta[idx]["n_components_after"] = diag.get("n_components_after", mask_meta[idx].get("n_components_after", np.nan))
        after_area = float(np.asarray(unique_pixels).sum())
        loss_fraction = 1.0 - after_area / max(before_area, 1.0)
        mask_meta[idx]["mask_area_after_overlap"] = after_area
        mask_meta[idx]["mask_area_loss_fraction"] = float(np.clip(loss_fraction, 0.0, 1.0))
        mask_meta[idx]["mask_fragment_after_overlap"] = bool(
            mask_meta[idx].get("mask_fragment_after_overlap", False)
            or after_area < float(config.min_component_area)
            or (before_area >= float(config.min_area) and after_area < 0.5 * before_area)
        )
        exclusive_masks[idx] = unique_pixels
        occupied |= unique_pixels

    keep_indices = [i for i, m in enumerate(exclusive_masks) if np.asarray(m).any()]
    masks = [exclusive_masks[i] for i in keep_indices]
    confs = confs[keep_indices] if len(keep_indices) else np.array([], dtype=float)
    boxes = boxes[keep_indices] if len(keep_indices) else np.empty((0, 4), dtype=float)
    mask_meta = [mask_meta[i] for i in keep_indices]
    return masks, confs, boxes, mask_meta


def predict_colony_masks(
    model,
    image_rgb: np.ndarray,
    config: AnomalyDetectionConfig,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    h, w = image_rgb.shape[:2]

    detections_columns = [
        "colony_id",
        "yolo_conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_area_yolo",
        "mask_area_raw",
        "mask_area_cleaned",
        "mask_area_after_overlap",
        "mask_area_loss_fraction",
        "original_area",
        "cleaned_area",
        "area_loss_fraction",
        "n_components_before",
        "n_components_after",
        "mask_fragment_after_overlap",
    ]

    predict_kwargs = dict(
        source=image_rgb,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        max_det=config.max_det,
        device=config.device,
        verbose=False,
        retina_masks=config.retina_masks,
    )
    if config.classes is not None:
        predict_kwargs["classes"] = list(config.classes)

    results = model.predict(**predict_kwargs)

    if results is None or len(results) == 0:
        return [], pd.DataFrame(columns=detections_columns)

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return [], pd.DataFrame(columns=detections_columns)

    masks_np = result.masks.data.cpu().numpy()
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    masks: list[np.ndarray] = []
    raw_keep_indices: list[int] = []
    mask_meta: list[dict[str, Any]] = []
    for raw_idx, m in enumerate(masks_np):
        m_bool = m.astype(np.float32) > 0.5
        if m_bool.shape != (h, w):
            m_bool = cv2.resize(m_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        raw_area = float(m_bool.sum())
        cleaned, diag = clean_instance_mask(m_bool, config)
        cleaned_area = float(cleaned.sum())
        if cleaned.any():
            masks.append(cleaned)
            raw_keep_indices.append(raw_idx)
            mask_meta.append(
                {
                    "mask_area_raw": raw_area,
                    "mask_area_cleaned": cleaned_area,
                    "mask_area_after_overlap": cleaned_area,
                    "mask_area_loss_fraction": float(diag.get("area_loss_fraction", 0.0)),
                    "original_area": float(diag.get("original_area", raw_area)),
                    "cleaned_area": float(diag.get("cleaned_area", cleaned_area)),
                    "area_loss_fraction": float(diag.get("area_loss_fraction", 0.0)),
                    "n_components_before": int(diag.get("n_components_before", 0)),
                    "n_components_after": int(diag.get("n_components_after", 0)),
                    "mask_fragment_after_overlap": bool(diag.get("mask_fragment_after_overlap", False)),
                }
            )

    n_masks = len(masks)
    if n_masks == 0:
        return [], pd.DataFrame(columns=detections_columns)

    confs = result.boxes.conf.cpu().numpy().astype(float) if result.boxes is not None and result.boxes.conf is not None else np.array([], dtype=float)
    boxes = result.boxes.xyxy.cpu().numpy().astype(float) if result.boxes is not None and result.boxes.xyxy is not None else np.empty((0, 4), dtype=float)

    if raw_keep_indices and confs.shape[0] > max(raw_keep_indices):
        confs = confs[raw_keep_indices]
    if raw_keep_indices and boxes.shape[0] > max(raw_keep_indices):
        boxes = boxes[raw_keep_indices]

    if confs.shape[0] < n_masks:
        confs = np.pad(confs, (0, n_masks - confs.shape[0]), constant_values=np.nan)
    elif confs.shape[0] > n_masks:
        confs = confs[:n_masks]

    if boxes.shape[0] < n_masks:
        pad_rows = n_masks - boxes.shape[0]
        boxes = np.vstack([boxes, np.full((pad_rows, 4), np.nan)]) if boxes.size else np.full((n_masks, 4), np.nan)
    elif boxes.shape[0] > n_masks:
        boxes = boxes[:n_masks]

    masks, confs, boxes, mask_meta = make_masks_non_overlapping(masks, confs, boxes, config, mask_meta)
    n_masks = len(masks)
    if n_masks == 0:
        return [], pd.DataFrame(columns=detections_columns)

    meta_df = pd.DataFrame(mask_meta)
    if len(meta_df) != n_masks:
        meta_df = pd.DataFrame(index=range(n_masks))
    defaults = {
        "mask_area_raw": np.nan,
        "mask_area_cleaned": np.nan,
        "mask_area_after_overlap": np.nan,
        "mask_area_loss_fraction": np.nan,
        "original_area": np.nan,
        "cleaned_area": np.nan,
        "area_loss_fraction": np.nan,
        "n_components_before": np.nan,
        "n_components_after": np.nan,
        "mask_fragment_after_overlap": False,
    }
    for col, default in defaults.items():
        if col not in meta_df.columns:
            meta_df[col] = default

    detections_df = pd.DataFrame(
        {
            "colony_id": np.arange(1, n_masks + 1, dtype=int),
            "yolo_conf": confs,
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "mask_area_raw": meta_df["mask_area_raw"].to_numpy(dtype=float),
            "mask_area_cleaned": meta_df["mask_area_cleaned"].to_numpy(dtype=float),
            "mask_area_after_overlap": meta_df["mask_area_after_overlap"].to_numpy(dtype=float),
            "mask_area_loss_fraction": meta_df["mask_area_loss_fraction"].to_numpy(dtype=float),
            "original_area": meta_df["original_area"].to_numpy(dtype=float),
            "cleaned_area": meta_df["cleaned_area"].to_numpy(dtype=float),
            "area_loss_fraction": meta_df["area_loss_fraction"].to_numpy(dtype=float),
            "n_components_before": meta_df["n_components_before"].to_numpy(dtype=float),
            "n_components_after": meta_df["n_components_after"].to_numpy(dtype=float),
            "mask_fragment_after_overlap": meta_df["mask_fragment_after_overlap"].fillna(False).astype(bool).to_numpy(),
        }
    )

    bbox_w = np.maximum(0.0, detections_df["x2"] - detections_df["x1"])
    bbox_h = np.maximum(0.0, detections_df["y2"] - detections_df["y1"])
    detections_df["bbox_area_yolo"] = bbox_w * bbox_h

    return masks, detections_df[detections_columns]

def normalize_masks_input(masks) -> list[np.ndarray]:
    if masks is None:
        return []

    if isinstance(masks, list):
        normalized = []
        for m in masks:
            arr = np.asarray(m)
            if arr.ndim != 2:
                continue
            arr_bool = arr.astype(bool)
            if arr_bool.any():
                normalized.append(arr_bool)
        return normalized

    arr = np.asarray(masks)

    if arr.ndim == 3:
        normalized = []
        for i in range(arr.shape[0]):
            mask_i = arr[i].astype(bool)
            if mask_i.any():
                normalized.append(mask_i)
        return normalized

    if arr.ndim == 2:
        if arr.dtype == bool:
            return [arr] if arr.any() else []

        unique_values = np.unique(arr)
        non_zero_ids = [int(v) for v in unique_values if v > 0]

        if len(non_zero_ids) == 0:
            return []

        if set(unique_values.tolist()).issubset({0, 1}):
            arr_bool = arr.astype(bool)
            return [arr_bool] if arr_bool.any() else []

        return [(arr == idx) for idx in sorted(non_zero_ids)]

    raise ValueError(
        "Неподдерживаемый формат masks. Ожидалось: list[np.ndarray], (N,H,W) или labeled mask (H,W)."
    )

def compute_local_background_ring(
    mask: np.ndarray,
    all_masks: list[np.ndarray],
    radius: int = 12,
) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("mask должен иметь форму (H, W)")

    radius = max(1, int(radius))

    dilated = binary_dilation(mask, disk(radius))
    ring = dilated & ~mask

    other_objects = np.zeros_like(mask, dtype=bool)
    for other in normalize_masks_input(all_masks):
        if other.shape == mask.shape:
            other_objects |= other

    ring = ring & ~other_objects
    return ring.astype(bool)

def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= eps:
        return np.nan
    return float(numerator / (denominator + eps))


def _hist_entropy_from_values(values: np.ndarray, bins: int = 32, value_range: tuple[float, float] = (0, 256)) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts.astype(float) / float(total)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _finite_mad(values: np.ndarray, eps: float = 1e-8) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return float(max(mad, eps))


def _robust_neighbor_delta(value: float, neighbor_values: np.ndarray, eps: float = 1e-8) -> float:
    if not np.isfinite(value):
        return np.nan
    neighbor_values = np.asarray(neighbor_values, dtype=float)
    neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
    if neighbor_values.size == 0:
        return np.nan
    med = np.nanmedian(neighbor_values)
    mad = _finite_mad(neighbor_values, eps=eps)
    if not np.isfinite(med) or not np.isfinite(mad):
        return np.nan
    return float((value - med) / (1.4826 * mad + eps))


def _mask_contour_perimeter(mask_bool: np.ndarray) -> float:
    if mask_bool is None:
        return np.nan
    mask_bool = np.asarray(mask_bool).astype(bool)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return np.nan
    if mask_bool.shape[0] < 2 or mask_bool.shape[1] < 2:
        return np.nan
    try:
        contours = find_contours(mask_bool.astype(float), level=0.5)
    except ValueError:
        return np.nan
    lengths: list[float] = []
    for contour in contours:
        if contour.shape[0] < 2:
            continue
        diffs = np.diff(contour, axis=0)
        lengths.append(float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1)))))
    return float(np.sum(lengths)) if lengths else np.nan


def _radial_contour_features(mask_bool: np.ndarray, centroid_x: float, centroid_y: float) -> tuple[float, float, float]:
    mask_arr = np.asarray(mask_bool).astype(float)
    if mask_arr.ndim != 2 or mask_arr.shape[0] < 2 or mask_arr.shape[1] < 2:
        return np.nan, np.nan, np.nan
    try:
        contours = find_contours(mask_arr, level=0.5)
    except ValueError:
        return np.nan, np.nan, np.nan
    if not contours:
        return np.nan, np.nan, np.nan
    points = np.vstack([c for c in contours if c.shape[0] >= 3]) if any(c.shape[0] >= 3 for c in contours) else np.empty((0, 2))
    if points.shape[0] < 3:
        return np.nan, np.nan, np.nan
    distances = np.sqrt((points[:, 1] - centroid_x) ** 2 + (points[:, 0] - centroid_y) ** 2)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return np.nan, np.nan, np.nan
    mean_v = float(np.mean(distances))
    std_v = float(np.std(distances))
    cv_v = std_v / (mean_v + 1e-8) if mean_v > 0 else np.nan
    return mean_v, std_v, float(cv_v)


def _polygon_area(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return np.nan
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _compute_voronoi_areas(coords: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    areas = np.full(n, np.nan, dtype=float)
    if n < 4:
        return areas
    try:
        from scipy.spatial import Voronoi

        vor = Voronoi(coords)
        h, w = image_shape
        max_reasonable_area = float(h * w)
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or any(v < 0 for v in region):
                continue
            vertices = vor.vertices[region]
            if vertices.size == 0:
                continue
            if np.any(vertices[:, 0] < 0) or np.any(vertices[:, 0] > w) or np.any(vertices[:, 1] < 0) or np.any(vertices[:, 1] > h):
                continue
            area = _polygon_area(vertices)
            if np.isfinite(area) and 0 <= area <= max_reasonable_area:
                areas[i] = area
    except Exception:
        pass
    return areas


def _add_feature_knn_distance(features_df: pd.DataFrame, config: AnomalyDetectionConfig) -> pd.DataFrame:
    """Средняя дистанция до ближайших соседей в пространстве признаков.

    Важно: расстояние считается только среди валидных объектов. Технические маски
    не должны становиться соседями и искажать локальную норму.
    """
    base_cols = [
        "log_area",
        "circularity",
        "eccentricity",
        "solidity",
        "intensity_mean",
        "intensity_iqr",
        "intensity_entropy",
        "glcm_contrast",
        "lbp_std",
        "local_color_delta_lab",
        "intensity_hist_js_to_plate_median",
    ]
    cols = [c for c in base_cols if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])]
    features_df["feature_knn_distance"] = np.nan
    if len(features_df) < 2 or not cols:
        return features_df

    valid_mask = pd.Series(True, index=features_df.index)
    if "valid_for_anomaly" in features_df.columns:
        valid_mask &= features_df["valid_for_anomaly"].fillna(False).astype(bool)
    if "technical_warning" in features_df.columns:
        valid_mask &= ~features_df["technical_warning"].fillna(False).astype(bool)
    valid_idx = np.where(valid_mask.to_numpy())[0]
    if valid_idx.size < 2:
        return features_df

    try:
        valid_df = features_df.iloc[valid_idx]
        X = valid_df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = RobustScaler().fit_transform(X)
        tree = cKDTree(X)
        k = min(int(config.neighbor_k) + 1, len(valid_idx))
        dists, _ = tree.query(X, k=k)
        dists = np.atleast_2d(dists)
        distances = []
        for i in range(len(valid_idx)):
            row_dists = np.asarray(dists[i], dtype=float)
            row_dists = row_dists[np.isfinite(row_dists)]
            row_dists = row_dists[row_dists > 1e-12]
            distances.append(float(np.mean(row_dists[: max(1, k - 1)])) if row_dists.size else np.nan)
        features_df.loc[valid_idx, "feature_knn_distance"] = distances
    except Exception:
        features_df["feature_knn_distance"] = np.nan
    return features_df

def _build_technical_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    checks = [
        ("is_too_small", "too_small"),
        ("is_too_large", "too_large"),
        ("touches_image_border", "touches_border"),
        ("suspicious_aspect_ratio", "suspicious_aspect_ratio"),
        ("low_yolo_conf", "low_yolo_conf"),
        ("critical_poor_local_background", "critical_poor_local_background"),
        ("mask_fragment_after_overlap", "mask_fragment_after_overlap"),
        ("invalid_geometry", "invalid_geometry"),
        ("possible_segmentation_artifact", "possible_segmentation_artifact"),
    ]
    for col, reason in checks:
        try:
            if bool(row.get(col, False)):
                reasons.append(reason)
        except Exception:
            pass
    return ";".join(reasons)


def _is_segmentation_review_candidate(row: pd.Series, config: AnomalyDetectionConfig | None = None) -> bool:
    checks = [
        "possible_merged_colony",
        "possible_segmentation_artifact",
        "mask_fragment_after_overlap",
        "invalid_geometry",
        "low_yolo_conf",
        "suspicious_aspect_ratio",
    ]
    if config is None or bool(getattr(config, "exclude_border_touching_from_anomaly", True)):
        checks.append("touches_image_border")

    for col in checks:
        try:
            if bool(row.get(col, False)):
                return True
        except Exception:
            continue
    return False



def _normalize_histogram(hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    hist = np.asarray(hist, dtype=float)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.maximum(hist, 0.0)
    total = float(hist.sum())
    if total <= eps:
        if hist.size == 0:
            return hist
        return np.full(hist.shape, 1.0 / hist.size, dtype=float)
    return hist / total


def _safe_jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    try:
        p = _normalize_histogram(p)
        q = _normalize_histogram(q)
        if p.size == 0 or q.size == 0 or p.size != q.size:
            return np.nan
        value = distance.jensenshannon(p, q, base=2.0)
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _safe_wasserstein_hist(p: np.ndarray, q: np.ndarray) -> float:
    try:
        p = _normalize_histogram(p)
        q = _normalize_histogram(q)
        if p.size == 0 or q.size == 0 or p.size != q.size:
            return np.nan
        x = np.arange(p.size, dtype=float)
        return float(stats.wasserstein_distance(x, x, u_weights=p, v_weights=q))
    except Exception:
        return np.nan


def _hist_width_fraction(hist: np.ndarray) -> float:
    hist = _normalize_histogram(hist)
    if hist.size == 0:
        return np.nan
    nz = np.where(hist > 0.01 * max(float(hist.max()), 1e-12))[0]
    if nz.size == 0:
        return 0.0
    return float((nz.max() - nz.min() + 1) / max(1, hist.size))


def _robust_z_value(value: float, values: np.ndarray, eps: float = 1e-8) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or not np.isfinite(value):
        return np.nan
    med = np.nanmedian(values)
    mad = _finite_mad(values, eps=eps)
    if not np.isfinite(mad) or mad <= 0:
        return np.nan
    return float((value - med) / (1.4826 * mad + eps))


def _compute_histograms_for_features(
    features_df: pd.DataFrame,
    image_rgb: np.ndarray,
    masks_by_colony_id: dict[int, np.ndarray],
    gray: np.ndarray,
    lab: np.ndarray,
    config: AnomalyDetectionConfig,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    bins = int(max(4, getattr(config, "hist_bins", 32)))
    eps = 1e-8
    hist_store: dict[int, dict[str, np.ndarray]] = {}

    hist_cols = [
        "hist_intensity_entropy",
        "hist_intensity_peak_bin",
        "hist_intensity_peak_value",
        "hist_intensity_width",
        "hist_intensity_skewness",
        "hist_intensity_kurtosis",
        "dark_fraction",
        "bright_fraction",
        "contrast_hist_entropy",
        "contrast_dark_fraction",
        "contrast_bright_fraction",
        "intensity_hist_js_to_plate_median",
        "intensity_hist_wasserstein_to_plate_median",
        "intensity_hist_js_to_neighbors",
        "L_hist_js_to_plate_median",
        "L_hist_js_to_neighbors",
        "contrast_hist_js_to_plate_median",
        "relative_histogram_vs_neighbors",
    ]
    for c in hist_cols:
        if c not in features_df.columns:
            features_df[c] = np.nan

    if not bool(getattr(config, "use_histogram_features", True)) or features_df.empty:
        return features_df, hist_store

    all_valid_pixels: list[np.ndarray] = []
    for _, row in features_df.iterrows():
        if not bool(row.get("valid_for_anomaly", True)):
            continue
        mask = masks_by_colony_id.get(int(row["colony_id"]))
        if mask is None or not np.asarray(mask).any():
            continue
        vals = gray[mask].astype(float)
        if vals.size:
            all_valid_pixels.append(vals)
    if all_valid_pixels:
        valid_pixels = np.concatenate(all_valid_pixels)
        plate_p25 = float(np.nanpercentile(valid_pixels, 25))
        plate_p75 = float(np.nanpercentile(valid_pixels, 75))
    else:
        plate_p25, plate_p75 = 64.0, 192.0

    for idx, row in features_df.iterrows():
        cid = int(row["colony_id"])
        mask = masks_by_colony_id.get(cid)
        if mask is None or not np.asarray(mask).any():
            continue
        pix_gray = gray[mask].astype(float)
        pix_lab = lab[mask].astype(float)
        if pix_gray.size == 0:
            continue

        intensity_counts, _ = np.histogram(pix_gray, bins=bins, range=(0, 256))
        intensity_hist = _normalize_histogram(intensity_counts)
        L_counts, _ = np.histogram(pix_lab[:, 0], bins=bins, range=(0, 256))
        L_hist = _normalize_histogram(L_counts)
        contrast_center = row.get("local_background_intensity_mean", np.nan)
        if not np.isfinite(contrast_center):
            contrast_center = np.nanmedian(pix_gray)
        contrast_values = np.clip(pix_gray - float(contrast_center) + 128.0, 0, 255)
        contrast_counts, _ = np.histogram(contrast_values, bins=bins, range=(0, 256))
        contrast_hist = _normalize_histogram(contrast_counts)

        hist_store[cid] = {
            "intensity": intensity_hist,
            "L": L_hist,
            "contrast": contrast_hist,
        }
        features_df.at[idx, "hist_intensity_entropy"] = _hist_entropy_from_values(pix_gray, bins=bins, value_range=(0, 256))
        features_df.at[idx, "hist_intensity_peak_bin"] = int(np.argmax(intensity_hist))
        features_df.at[idx, "hist_intensity_peak_value"] = float(np.max(intensity_hist))
        features_df.at[idx, "hist_intensity_width"] = _hist_width_fraction(intensity_hist)
        features_df.at[idx, "hist_intensity_skewness"] = float(stats.skew(pix_gray, nan_policy="omit")) if pix_gray.size >= 3 else np.nan
        features_df.at[idx, "hist_intensity_kurtosis"] = float(stats.kurtosis(pix_gray, nan_policy="omit")) if pix_gray.size >= 4 else np.nan
        features_df.at[idx, "dark_fraction"] = float(np.mean(pix_gray < plate_p25))
        features_df.at[idx, "bright_fraction"] = float(np.mean(pix_gray > plate_p75))
        features_df.at[idx, "contrast_hist_entropy"] = _hist_entropy_from_values(contrast_values, bins=bins, value_range=(0, 256))
        features_df.at[idx, "contrast_dark_fraction"] = float(np.mean(contrast_values < 96))
        features_df.at[idx, "contrast_bright_fraction"] = float(np.mean(contrast_values > 160))

    valid_ids = features_df.loc[features_df.get("valid_for_anomaly", True).astype(bool), "colony_id"].astype(int).tolist() if "valid_for_anomaly" in features_df.columns else features_df["colony_id"].astype(int).tolist()
    valid_ids = [cid for cid in valid_ids if cid in hist_store]
    if not valid_ids:
        return features_df, hist_store

    median_hists: dict[str, np.ndarray] = {}
    for key in ["intensity", "L", "contrast"]:
        stack = np.vstack([hist_store[cid][key] for cid in valid_ids])
        median_hists[key] = _normalize_histogram(np.nanmedian(stack, axis=0))

    coords = features_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float) if {"centroid_x", "centroid_y"}.issubset(features_df.columns) else None
    tree = cKDTree(coords) if coords is not None and len(features_df) >= 2 else None

    for idx, row in features_df.iterrows():
        cid = int(row["colony_id"])
        if cid not in hist_store:
            continue
        features_df.at[idx, "intensity_hist_js_to_plate_median"] = _safe_jensen_shannon(hist_store[cid]["intensity"], median_hists["intensity"])
        features_df.at[idx, "intensity_hist_wasserstein_to_plate_median"] = _safe_wasserstein_hist(hist_store[cid]["intensity"], median_hists["intensity"])
        features_df.at[idx, "L_hist_js_to_plate_median"] = _safe_jensen_shannon(hist_store[cid]["L"], median_hists["L"])
        features_df.at[idx, "contrast_hist_js_to_plate_median"] = _safe_jensen_shannon(hist_store[cid]["contrast"], median_hists["contrast"])

        if tree is not None:
            k = min(int(config.neighbor_k) + 1, len(features_df))
            _, idxs = tree.query(coords[idx], k=k)
            idxs = np.atleast_1d(idxs).astype(int)
            neighbor_ids = [int(features_df.iloc[j]["colony_id"]) for j in idxs if j != idx and int(features_df.iloc[j]["colony_id"]) in hist_store]
            if neighbor_ids:
                nn_intensity = _normalize_histogram(np.nanmedian(np.vstack([hist_store[nid]["intensity"] for nid in neighbor_ids]), axis=0))
                nn_L = _normalize_histogram(np.nanmedian(np.vstack([hist_store[nid]["L"] for nid in neighbor_ids]), axis=0))
                features_df.at[idx, "intensity_hist_js_to_neighbors"] = _safe_jensen_shannon(hist_store[cid]["intensity"], nn_intensity)
                features_df.at[idx, "L_hist_js_to_neighbors"] = _safe_jensen_shannon(hist_store[cid]["L"], nn_L)

    if "intensity_hist_js_to_neighbors" in features_df.columns:
        vals = features_df["intensity_hist_js_to_neighbors"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
        mad = _finite_mad(vals)
        if np.isfinite(med) and np.isfinite(mad):
            rel = (vals - med) / (1.4826 * mad + eps)
            features_df["relative_histogram_vs_neighbors"] = np.clip(rel, -float(config.clip_relative_features), float(config.clip_relative_features))

    return features_df, hist_store


def _recompute_neighbor_relative_features(features_df: pd.DataFrame, config: AnomalyDetectionConfig) -> pd.DataFrame:
    """Локальные relative-признаки относительно ближайших валидных соседей.

    Нормирование выполняется через глобальную MAD по валидным объектам, а не через
    MAD только нескольких соседей. Это защищает от искусственно огромных значений,
    когда у ближайших соседей почти нулевой разброс.
    """
    if len(features_df) < 2 or not {"centroid_x", "centroid_y"}.issubset(features_df.columns):
        return features_df

    valid_mask = pd.Series(True, index=features_df.index)
    if "valid_for_anomaly" in features_df.columns:
        valid_mask &= features_df["valid_for_anomaly"].fillna(False).astype(bool)
    if "technical_warning" in features_df.columns:
        valid_mask &= ~features_df["technical_warning"].fillna(False).astype(bool)
    valid_idx = np.where(valid_mask.to_numpy())[0]
    if valid_idx.size < 2:
        return features_df

    coords_valid = features_df.iloc[valid_idx][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tree = cKDTree(coords_valid)
    feature_map = {
        "relative_area_vs_neighbors": "area",
        "relative_intensity_vs_neighbors": "intensity_mean",
        "relative_texture_vs_neighbors": "glcm_contrast",
        "relative_circularity_vs_neighbors": "circularity",
        "relative_solidity_vs_neighbors": "solidity",
        "relative_color_delta_lab_vs_neighbors": "local_color_delta_lab",
        "relative_local_contrast_vs_neighbors": "local_contrast",
        "relative_entropy_vs_neighbors": "intensity_entropy",
        "relative_histogram_vs_neighbors": "intensity_hist_js_to_neighbors",
    }
    clip = float(getattr(config, "clip_relative_features", 5.0))
    for out_col, source_col in feature_map.items():
        if source_col not in features_df.columns:
            continue
        valid_values = features_df.loc[valid_idx, source_col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        global_mad = _finite_mad(valid_values)
        if not np.isfinite(global_mad) or global_mad <= 0:
            continue
        for pos, global_i in enumerate(valid_idx):
            k = min(int(config.neighbor_k) + 1, len(valid_idx))
            _, local_idxs = tree.query(coords_valid[pos], k=k)
            local_idxs = np.atleast_1d(local_idxs).astype(int)
            nn_global = [valid_idx[j] for j in local_idxs if valid_idx[j] != global_i]
            if not nn_global:
                continue
            cur = float(features_df.at[global_i, source_col])
            nn_values = features_df.loc[nn_global, source_col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            nn_values = nn_values[np.isfinite(nn_values)]
            if not np.isfinite(cur) or nn_values.size == 0:
                continue
            med_nn = np.nanmedian(nn_values)
            value = (cur - med_nn) / (1.4826 * global_mad + 1e-8)
            features_df.at[global_i, out_col] = float(np.clip(value, -clip, clip))
    return features_df


def _augment_features_after_extraction(
    features_df: pd.DataFrame,
    image_rgb: np.ndarray,
    masks_by_colony_id: dict[int, np.ndarray],
    gray: np.ndarray,
    lab: np.ndarray,
    config: AnomalyDetectionConfig,
    image_area: float,
) -> pd.DataFrame:
    if features_df.empty:
        return features_df
    eps = 1e-8
    plate_area_median = np.nanmedian(features_df["area"].to_numpy(dtype=float))
    plate_equiv_median = np.nanmedian(features_df["equivalent_diameter"].to_numpy(dtype=float))
    if not np.isfinite(plate_equiv_median) or plate_equiv_median <= eps:
        plate_equiv_median = np.nan

    features_df["log_area"] = np.log1p(features_df["area"].astype(float))
    features_df["area_to_median_ratio"] = features_df["area"].astype(float) / (plate_area_median + eps) if np.isfinite(plate_area_median) else np.nan
    features_df["equivalent_diameter_to_median_ratio"] = features_df["equivalent_diameter"].astype(float) / (plate_equiv_median + eps) if np.isfinite(plate_equiv_median) else np.nan
    features_df["size_global_z"] = [
        _robust_z_value(v, features_df["log_area"].to_numpy(dtype=float)) for v in features_df["log_area"].to_numpy(dtype=float)
    ]

    features_df["possible_merged_colony"] = (
        (features_df["aspect_ratio"].astype(float) > 2.2)
        | ((features_df["eccentricity"].astype(float) > 0.88) & (features_df["circularity"].astype(float) < 0.70))
        | ((features_df["area_to_median_ratio"].astype(float) > 2.0) & (features_df["convexity_defect_ratio"].astype(float) > 0.15))
    )
    features_df["possible_merged_colony_score"] = features_df["possible_merged_colony"].astype(float)

    if len(features_df) >= 2:
        coords = features_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        radii = np.sqrt(np.maximum(features_df["area"].astype(float).to_numpy(), 0.0) / math.pi)
        tree = cKDTree(coords)
        edge_nearest = np.full(len(features_df), np.nan, dtype=float)
        mean_edge = np.full(len(features_df), np.nan, dtype=float)
        size_vs_neighbors = np.full(len(features_df), np.nan, dtype=float)
        for i in range(len(features_df)):
            k = min(int(config.neighbor_k) + 1, len(features_df))
            dists, idxs = tree.query(coords[i], k=k)
            dists = np.atleast_1d(dists).astype(float)
            idxs = np.atleast_1d(idxs).astype(int)
            keep = idxs != i
            nd = dists[keep]
            ni = idxs[keep]
            if nd.size:
                edge_d = nd - radii[i] - radii[ni]
                edge_nearest[i] = float(edge_d[0])
                mean_edge[i] = float(np.mean(edge_d[:5]))
                neighbor_log_area = features_df.loc[ni[: int(config.neighbor_k)], "log_area"].to_numpy(dtype=float)
                size_vs_neighbors[i] = _robust_neighbor_delta(float(features_df.at[i, "log_area"]), neighbor_log_area)
        features_df["edge_nearest_distance"] = edge_nearest
        features_df["mean_5nn_edge_distance"] = mean_edge
        features_df["size_vs_neighbors_z"] = np.clip(size_vs_neighbors, -float(config.clip_relative_features), float(config.clip_relative_features))
    else:
        features_df["edge_nearest_distance"] = np.nan
        features_df["mean_5nn_edge_distance"] = np.nan
        features_df["size_vs_neighbors_z"] = np.nan

    features_df["nearest_distance_to_median_diameter_ratio"] = features_df["nearest_neighbor_distance"].astype(float) / (plate_equiv_median + eps) if np.isfinite(plate_equiv_median) else np.nan
    features_df["edge_distance_to_median_diameter_ratio"] = features_df["edge_nearest_distance"].astype(float) / (plate_equiv_median + eps) if np.isfinite(plate_equiv_median) else np.nan

    # Update technical flags after derived mask/shape diagnostics.
    features_df["possible_segmentation_artifact"] = (
        features_df.get("mask_fragment_after_overlap", False).astype(bool)
        | features_df.get("invalid_geometry", False).astype(bool)
        | (features_df["possible_merged_colony"].astype(bool) & (features_df.get("mask_area_loss_fraction", 0).fillna(0).astype(float) > 0.30))
    )

    min_area_threshold = float(config.min_area)
    if np.isfinite(plate_area_median):
        min_area_threshold = max(min_area_threshold, float(plate_area_median) * float(config.min_area_relative_to_median))
    max_area_threshold = float(config.max_area_fraction) * image_area
    if np.isfinite(plate_area_median) and plate_area_median > 0:
        max_area_threshold = min(max_area_threshold, float(plate_area_median) * float(config.max_area_relative_to_median))
    features_df["is_too_small"] = features_df["area"].astype(float) < min_area_threshold
    features_df["is_too_large"] = features_df["area"].astype(float) > max_area_threshold
    if "yolo_conf" not in features_df.columns:
        features_df["yolo_conf"] = np.nan
    features_df["low_yolo_conf"] = features_df["yolo_conf"].fillna(1.0).astype(float) < float(config.min_yolo_conf_for_analysis)

    technical_flags = (
        features_df["is_too_small"].fillna(True).astype(bool)
        | features_df["is_too_large"].fillna(True).astype(bool)
        | features_df.get("suspicious_aspect_ratio", False).fillna(False).astype(bool)
        | features_df["low_yolo_conf"].fillna(False).astype(bool)
        | features_df.get("critical_poor_local_background", False).fillna(False).astype(bool)
        | features_df.get("mask_fragment_after_overlap", False).fillna(False).astype(bool)
        | features_df.get("invalid_geometry", False).fillna(False).astype(bool)
        | features_df["possible_segmentation_artifact"].fillna(False).astype(bool)
    )
    if config.exclude_border_touching_from_anomaly:
        technical_flags = technical_flags | features_df.get("touches_image_border", False).fillna(False).astype(bool)
    features_df["technical_warning"] = technical_flags.astype(bool)
    features_df["valid_for_anomaly"] = ~features_df["technical_warning"].astype(bool)

    features_df, hist_store = _compute_histograms_for_features(features_df, image_rgb, masks_by_colony_id, gray, lab, config)
    features_df = _recompute_neighbor_relative_features(features_df, config)
    features_df = _add_feature_knn_distance(features_df, config)
    features_df["technical_reason"] = features_df.apply(_build_technical_reason, axis=1)
    features_df.loc[~features_df["technical_warning"].astype(bool), "technical_reason"] = ""
    return features_df


def extract_colony_features(
    image_rgb: np.ndarray,
    masks,
    detections_df: pd.DataFrame | None = None,
    plate_mask: np.ndarray | None = None,
    config: AnomalyDetectionConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = AnomalyDetectionConfig()

    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Ожидалось RGB-изображение формы (H, W, 3)")

    h, w = image_rgb.shape[:2]
    masks_list = normalize_masks_input(masks)

    prepared_masks: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for source_colony_id, m in enumerate(masks_list, start=1):
        m_bool = np.asarray(m).astype(bool)
        if m_bool.shape != (h, w):
            m_bool = cv2.resize(
                m_bool.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        raw_area = float(m_bool.sum())
        cleaned, diag = clean_instance_mask(m_bool, config)
        cleaned_area = float(cleaned.sum())
        loss_fraction = float(diag.get("area_loss_fraction", 1.0 - cleaned_area / max(raw_area, 1.0)))
        mask_meta = {
            "mask_area_input": raw_area,
            "mask_area_after_feature_cleaning": cleaned_area,
            "mask_area_feature_loss_fraction": float(np.clip(loss_fraction, 0.0, 1.0)),
            "original_area": float(diag.get("original_area", raw_area)),
            "cleaned_area": float(diag.get("cleaned_area", cleaned_area)),
            "area_loss_fraction": float(diag.get("area_loss_fraction", loss_fraction)),
            "n_components_before": int(diag.get("n_components_before", 0)),
            "n_components_after": int(diag.get("n_components_after", 0)),
            "mask_fragment_after_overlap": bool(diag.get("mask_fragment_after_overlap", False)),
        }
        if cleaned.any():
            feature_mask = cleaned.astype(bool)
        else:
            # Preserve a fully discarded fragment as a technical warning row.
            # Otherwise it disappears from technical_warnings.csv and quality control.
            feature_mask = m_bool.astype(bool)
            mask_meta["invalid_geometry"] = True
            mask_meta["mask_fragment_after_overlap"] = True
        if feature_mask.any():
            prepared_masks.append((source_colony_id, feature_mask, mask_meta))

    if len(prepared_masks) == 0:
        return pd.DataFrame()
    analysis_masks = [m for _, m, _ in prepared_masks]
    masks_by_colony_id = {int(cid): m for cid, m, _ in prepared_masks}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    eps = 1e-8
    image_area = float(h * w)

    if plate_mask is None:
        plate_mask_bool = np.ones((h, w), dtype=bool)
        plate_center_x = (w - 1) / 2.0
        plate_center_y = (h - 1) / 2.0
        plate_dist_map = None
    else:
        plate_mask_bool = np.asarray(plate_mask).astype(bool)
        if plate_mask_bool.shape != (h, w):
            plate_mask_bool = cv2.resize(
                plate_mask_bool.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not plate_mask_bool.any():
            plate_mask_bool = np.ones((h, w), dtype=bool)

        ys_pm, xs_pm = np.where(plate_mask_bool)
        if len(xs_pm) > 0:
            plate_center_x = float(np.mean(xs_pm))
            plate_center_y = float(np.mean(ys_pm))
        else:
            plate_center_x = (w - 1) / 2.0
            plate_center_y = (h - 1) / 2.0

        plate_dist_map = cv2.distanceTransform(
            plate_mask_bool.astype(np.uint8), cv2.DIST_L2, 3
        )

    detections_lookup: dict[int, dict[str, float]] = {}
    if detections_df is not None and not detections_df.empty and "colony_id" in detections_df.columns:
        for _, row in detections_df.iterrows():
            colony_id = int(row["colony_id"])
            detections_lookup[colony_id] = row.to_dict()

    rows: list[dict[str, Any]] = []

    for colony_id, mask, mask_meta in prepared_masks:
        labeled_mask = label(mask.astype(np.uint8))
        props = regionprops(labeled_mask)
        if not props:
            continue

        prop = max(props, key=lambda p: p.area)
        area = float(prop.area)
        if area <= 0:
            continue

        perimeter = float(prop.perimeter)
        equivalent_diameter = float(prop.equivalent_diameter)

        min_row, min_col, max_row, max_col = prop.bbox
        bbox_width = float(max_col - min_col)
        bbox_height = float(max_row - min_row)
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else np.nan
        circularity = (
            4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else np.nan
        )

        eccentricity = float(getattr(prop, "eccentricity", np.nan))
        solidity = float(getattr(prop, "solidity", np.nan))
        extent = float(getattr(prop, "extent", np.nan))
        major_axis_length = float(getattr(prop, "major_axis_length", np.nan))
        minor_axis_length = float(getattr(prop, "minor_axis_length", np.nan))
        orientation = float(getattr(prop, "orientation", np.nan))
        convex_area = float(getattr(prop, "convex_area", np.nan))
        convexity_defect_area = convex_area - area if np.isfinite(convex_area) else np.nan
        convexity_defect_ratio = _safe_ratio(convexity_defect_area, area, eps=eps)
        convex_perimeter = _mask_contour_perimeter(getattr(prop, "image_convex", None))
        boundary_roughness = _safe_ratio(perimeter, convex_perimeter, eps=eps)

        centroid_y, centroid_x = prop.centroid
        centroid_x = float(centroid_x)
        centroid_y = float(centroid_y)
        centroid_x_norm = centroid_x / max(1.0, float(w - 1))
        centroid_y_norm = centroid_y / max(1.0, float(h - 1))

        distance_to_plate_center = float(
            np.hypot(centroid_x - plate_center_x, centroid_y - plate_center_y)
        )

        if plate_dist_map is None:
            distance_to_plate_edge = float(
                min(
                    centroid_x,
                    centroid_y,
                    (w - 1) - centroid_x,
                    (h - 1) - centroid_y,
                )
            )
        else:
            cyi = int(np.clip(round(centroid_y), 0, h - 1))
            cxi = int(np.clip(round(centroid_x), 0, w - 1))
            distance_to_plate_edge = float(plate_dist_map[cyi, cxi])
            if not np.isfinite(distance_to_plate_edge) or distance_to_plate_edge <= 0:
                distance_to_plate_edge = float(
                    min(
                        centroid_x,
                        centroid_y,
                        (w - 1) - centroid_x,
                        (h - 1) - centroid_y,
                    )
                )

        radial_contour_mean, radial_contour_std, radial_contour_cv = _radial_contour_features(
            mask, centroid_x=centroid_x, centroid_y=centroid_y
        )

        pix_rgb = image_rgb[mask]
        pix_gray = gray[mask]
        pix_hsv = hsv[mask]
        pix_lab = lab[mask]

        if pix_gray.size == 0:
            continue

        mean_R, mean_G, mean_B = [float(v) for v in np.mean(pix_rgb, axis=0)]
        std_R, std_G, std_B = [float(v) for v in np.std(pix_rgb, axis=0)]

        mean_H, mean_S, mean_V = [float(v) for v in np.mean(pix_hsv, axis=0)]
        std_H, std_S, std_V = [float(v) for v in np.std(pix_hsv, axis=0)]
        hue_rad = pix_hsv[:, 0].astype(float) / 180.0 * 2.0 * np.pi
        mean_H_sin = float(np.mean(np.sin(hue_rad))) if hue_rad.size else np.nan
        mean_H_cos = float(np.mean(np.cos(hue_rad))) if hue_rad.size else np.nan

        mean_L_lab, mean_a_lab, mean_b_lab = [float(v) for v in np.mean(pix_lab, axis=0)]
        std_L_lab, std_a_lab, std_b_lab = [float(v) for v in np.std(pix_lab, axis=0)]

        intensity_mean = float(np.mean(pix_gray))
        intensity_median = float(np.median(pix_gray))
        intensity_std = float(np.std(pix_gray))
        intensity_min = float(np.min(pix_gray))
        intensity_max = float(np.max(pix_gray))
        intensity_range = intensity_max - intensity_min
        intensity_cv = intensity_std / (abs(intensity_mean) + eps)
        intensity_p05, intensity_p25, intensity_p75, intensity_p95 = [
            float(v) for v in np.percentile(pix_gray, [5, 25, 75, 95])
        ]
        intensity_iqr = intensity_p75 - intensity_p25
        intensity_p95_p05_range = intensity_p95 - intensity_p05
        intensity_entropy = _hist_entropy_from_values(pix_gray, bins=32, value_range=(0, 256))

        ys_mask, xs_mask = np.where(mask)
        radial_distances = np.sqrt((xs_mask.astype(float) - centroid_x) ** 2 + (ys_mask.astype(float) - centroid_y) ** 2)
        max_radius = float(np.max(radial_distances)) if radial_distances.size else np.nan
        if radial_distances.size >= 3 and np.isfinite(max_radius) and max_radius > eps:
            normalized_radius = radial_distances / max_radius
            pix_gray_float = pix_gray.astype(float)
            center_vals = pix_gray_float[normalized_radius <= 0.35]
            rim_vals = pix_gray_float[normalized_radius >= 0.70]
            center_intensity_mean = float(np.mean(center_vals)) if center_vals.size else np.nan
            rim_intensity_mean = float(np.mean(rim_vals)) if rim_vals.size else np.nan
            center_rim_intensity_delta = center_intensity_mean - rim_intensity_mean if np.isfinite(center_intensity_mean) and np.isfinite(rim_intensity_mean) else np.nan
            center_rim_intensity_ratio = _safe_ratio(center_intensity_mean, rim_intensity_mean, eps=eps)
            radial_intensity_std = float(np.std(pix_gray_float)) if pix_gray_float.size else np.nan
            try:
                radial_intensity_slope = float(np.polyfit(normalized_radius, pix_gray_float, deg=1)[0])
            except Exception:
                radial_intensity_slope = np.nan
        else:
            center_intensity_mean = np.nan
            rim_intensity_mean = np.nan
            center_rim_intensity_delta = np.nan
            center_rim_intensity_ratio = np.nan
            radial_intensity_slope = np.nan
            radial_intensity_std = np.nan

        y1, y2 = int(min_row), int(max_row)
        x1, x2 = int(min_col), int(max_col)
        gray_crop = gray[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]

        if gray_crop.size == 0 or not mask_crop.any():
            laplacian_var = np.nan
            glcm_contrast = np.nan
            glcm_homogeneity = np.nan
            glcm_energy = np.nan
            glcm_correlation = np.nan
            glcm_dissimilarity = np.nan
            glcm_ASM = np.nan
            lbp_mean = np.nan
            lbp_std = np.nan
            lbp_entropy = np.nan
        else:
            inside_vals = gray_crop[mask_crop]
            fill_val = float(np.median(inside_vals)) if inside_vals.size else float(np.median(gray_crop))
            gray_crop_filled = gray_crop.astype(np.float32)
            gray_crop_filled[~mask_crop] = fill_val
            gray_crop_u8 = np.clip(gray_crop_filled, 0, 255).astype(np.uint8)

            laplacian_var = float(cv2.Laplacian(gray_crop_u8, cv2.CV_32F).var())

            try:
                glcm_levels = int(max(2, min(256, config.glcm_levels)))
                gray_crop_q = np.floor(gray_crop_u8.astype(np.float32) * glcm_levels / 256.0).astype(np.uint8)
                gray_crop_q = np.clip(gray_crop_q, 0, glcm_levels - 1)
                glcm = graycomatrix(
                    gray_crop_q,
                    distances=list(config.glcm_distances),
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=glcm_levels,
                    symmetric=True,
                    normed=True,
                )
                glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
                glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
                glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
                glcm_correlation = float(np.mean(graycoprops(glcm, "correlation")))
                glcm_dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
                glcm_ASM = float(np.mean(graycoprops(glcm, "ASM")))
            except Exception:
                glcm_contrast = np.nan
                glcm_homogeneity = np.nan
                glcm_energy = np.nan
                glcm_correlation = np.nan
                glcm_dissimilarity = np.nan
                glcm_ASM = np.nan

            try:
                lbp = local_binary_pattern(gray_crop_u8, P=config.lbp_p, R=config.lbp_r, method="uniform")
                lbp_vals = lbp[mask_crop] if mask_crop.any() else lbp.ravel()
                lbp_mean = float(np.mean(lbp_vals)) if lbp_vals.size else np.nan
                lbp_std = float(np.std(lbp_vals)) if lbp_vals.size else np.nan
                lbp_entropy = _hist_entropy_from_values(lbp_vals, bins=int(config.lbp_p) + 2, value_range=(0, int(config.lbp_p) + 2)) if lbp_vals.size else np.nan
            except Exception:
                lbp_mean = np.nan
                lbp_std = np.nan
                lbp_entropy = np.nan

        raw_background_ring = binary_dilation(mask, disk(config.local_background_radius)) & (~mask)
        ring = compute_local_background_ring(
            mask=mask,
            all_masks=analysis_masks,
            radius=config.local_background_radius,
        )
        local_background_area = int(ring.sum())
        raw_background_area = int(raw_background_ring.sum())
        local_background_valid_fraction = _safe_ratio(local_background_area, raw_background_area, eps=eps)

        if ring.any():
            bg_gray = gray[ring]
            bg_rgb = image_rgb[ring]
            bg_lab = lab[ring]

            local_background_intensity_mean = float(np.mean(bg_gray))
            local_background_intensity_std = float(np.std(bg_gray))
            local_contrast = float(abs(intensity_mean - local_background_intensity_mean))

            mean_obj_rgb = np.mean(pix_rgb, axis=0)
            mean_bg_rgb = np.mean(bg_rgb, axis=0)
            local_color_delta_rgb = float(np.linalg.norm(mean_obj_rgb - mean_bg_rgb))

            local_background_L_mean, local_background_a_mean, local_background_b_mean = [float(v) for v in np.mean(bg_lab, axis=0)]
            local_delta_L = mean_L_lab - local_background_L_mean
            local_delta_a = mean_a_lab - local_background_a_mean
            local_delta_b = mean_b_lab - local_background_b_mean
            local_color_delta_lab = float(np.sqrt(local_delta_L**2 + local_delta_a**2 + local_delta_b**2))
        else:
            local_background_intensity_mean = np.nan
            local_background_intensity_std = np.nan
            local_contrast = np.nan
            local_color_delta_rgb = np.nan
            local_background_L_mean = np.nan
            local_background_a_mean = np.nan
            local_background_b_mean = np.nan
            local_delta_L = np.nan
            local_delta_a = np.nan
            local_delta_b = np.nan
            local_color_delta_lab = np.nan

        poor_local_background = bool(local_background_area < max(20, 0.1 * area))
        critical_poor_local_background = bool(local_background_area < 5)

        edge_margin = max(0, int(config.edge_margin_px))
        if edge_margin == 0:
            touches_image_border = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
        else:
            touches_image_border = bool(
                mask[:edge_margin, :].any()
                or mask[-edge_margin:, :].any()
                or mask[:, :edge_margin].any()
                or mask[:, -edge_margin:].any()
            )
        is_too_small = bool(area < config.min_area)
        is_too_large = bool(area > config.max_area_fraction * image_area)

        if np.isfinite(aspect_ratio) and aspect_ratio > 0:
            inv_ar = 1.0 / max(aspect_ratio, eps)
            suspicious_aspect_ratio = bool(
                max(aspect_ratio, inv_ar) > config.suspicious_aspect_ratio_threshold
            )
        else:
            suspicious_aspect_ratio = False

        technical_warning = bool(
            touches_image_border
            or is_too_small
            or is_too_large
            or suspicious_aspect_ratio
        )

        row = {
            "colony_id": int(colony_id),
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": equivalent_diameter,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "extent": extent,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
            "orientation": orientation,
            "convex_area": convex_area,
            "convexity_defect_area": convexity_defect_area,
            "convexity_defect_ratio": convexity_defect_ratio,
            "convex_perimeter": convex_perimeter,
            "boundary_roughness": boundary_roughness,
            "radial_contour_mean": radial_contour_mean,
            "radial_contour_std": radial_contour_std,
            "radial_contour_cv": radial_contour_cv,
            "mean_R": mean_R,
            "mean_G": mean_G,
            "mean_B": mean_B,
            "std_R": std_R,
            "std_G": std_G,
            "std_B": std_B,
            "mean_H": mean_H,
            "mean_H_sin": mean_H_sin,
            "mean_H_cos": mean_H_cos,
            "mean_S": mean_S,
            "mean_V": mean_V,
            "std_H": std_H,
            "std_S": std_S,
            "std_V": std_V,
            "mean_L_lab": mean_L_lab,
            "mean_a_lab": mean_a_lab,
            "mean_b_lab": mean_b_lab,
            "std_L_lab": std_L_lab,
            "std_a_lab": std_a_lab,
            "std_b_lab": std_b_lab,
            "intensity_mean": intensity_mean,
            "intensity_median": intensity_median,
            "intensity_std": intensity_std,
            "intensity_min": intensity_min,
            "intensity_max": intensity_max,
            "intensity_range": intensity_range,
            "intensity_cv": intensity_cv,
            "intensity_p05": intensity_p05,
            "intensity_p25": intensity_p25,
            "intensity_p75": intensity_p75,
            "intensity_p95": intensity_p95,
            "intensity_iqr": intensity_iqr,
            "intensity_p95_p05_range": intensity_p95_p05_range,
            "intensity_entropy": intensity_entropy,
            "center_intensity_mean": center_intensity_mean,
            "rim_intensity_mean": rim_intensity_mean,
            "center_rim_intensity_delta": center_rim_intensity_delta,
            "center_rim_intensity_ratio": center_rim_intensity_ratio,
            "radial_intensity_slope": radial_intensity_slope,
            "radial_intensity_std": radial_intensity_std,
            "laplacian_var": laplacian_var,
            "glcm_contrast": glcm_contrast,
            "glcm_homogeneity": glcm_homogeneity,
            "glcm_energy": glcm_energy,
            "glcm_correlation": glcm_correlation,
            "glcm_dissimilarity": glcm_dissimilarity,
            "glcm_ASM": glcm_ASM,
            "lbp_mean": lbp_mean,
            "lbp_std": lbp_std,
            "lbp_entropy": lbp_entropy,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_x_norm": centroid_x_norm,
            "centroid_y_norm": centroid_y_norm,
            "distance_to_plate_center": distance_to_plate_center,
            "distance_to_plate_edge": distance_to_plate_edge,
            "nearest_neighbor_distance": np.nan,
            "mean_5nn_distance": np.nan,
            "log_nearest_neighbor_distance": np.nan,
            "log_mean_5nn_distance": np.nan,
            "edge_nearest_distance": np.nan,
            "mean_5nn_edge_distance": np.nan,
            "nearest_distance_to_median_diameter_ratio": np.nan,
            "edge_distance_to_median_diameter_ratio": np.nan,
            "local_density_r": np.nan,
            "voronoi_area": np.nan,
            "local_background_intensity_mean": local_background_intensity_mean,
            "local_background_intensity_std": local_background_intensity_std,
            "local_background_area": local_background_area,
            "local_background_valid_fraction": local_background_valid_fraction,
            "local_background_L_mean": local_background_L_mean,
            "local_background_a_mean": local_background_a_mean,
            "local_background_b_mean": local_background_b_mean,
            "local_delta_L": local_delta_L,
            "local_delta_a": local_delta_a,
            "local_delta_b": local_delta_b,
            "local_contrast": local_contrast,
            "local_color_delta_rgb": local_color_delta_rgb,
            "local_color_delta_lab": local_color_delta_lab,
            "relative_area_vs_plate_median": np.nan,
            "relative_intensity_vs_plate_median": np.nan,
            "relative_texture_vs_plate_median": np.nan,
            "relative_area_vs_neighbors": np.nan,
            "relative_intensity_vs_neighbors": np.nan,
            "relative_texture_vs_neighbors": np.nan,
            "relative_circularity_vs_neighbors": np.nan,
            "relative_solidity_vs_neighbors": np.nan,
            "relative_color_delta_lab_vs_neighbors": np.nan,
            "relative_local_contrast_vs_neighbors": np.nan,
            "relative_entropy_vs_neighbors": np.nan,
            "relative_histogram_vs_neighbors": np.nan,
            "feature_knn_distance": np.nan,
            "touches_image_border": touches_image_border,
            "is_too_small": is_too_small,
            "is_too_large": is_too_large,
            "suspicious_aspect_ratio": suspicious_aspect_ratio,
            "poor_local_background": poor_local_background,
            "critical_poor_local_background": critical_poor_local_background,
            "low_yolo_conf": False,
            "mask_fragment_after_overlap": bool(mask_meta.get("mask_fragment_after_overlap", False)),
            "possible_merged_colony": False,
            "possible_merged_colony_score": 0.0,
            "possible_segmentation_artifact": False,
            "invalid_geometry": bool(
                (not np.isfinite(area))
                or area <= 0
                or (not np.isfinite(perimeter))
                or perimeter <= 0
                or (np.isfinite(circularity) and circularity > 1.5)
                or (not np.isfinite(aspect_ratio))
            ),
            "mask_area_input": float(mask_meta.get("mask_area_input", np.nan)),
            "mask_area_after_feature_cleaning": float(mask_meta.get("mask_area_after_feature_cleaning", np.nan)),
            "mask_area_feature_loss_fraction": float(mask_meta.get("mask_area_feature_loss_fraction", np.nan)),
            "original_area": float(mask_meta.get("original_area", np.nan)),
            "cleaned_area": float(mask_meta.get("cleaned_area", np.nan)),
            "area_loss_fraction": float(mask_meta.get("area_loss_fraction", np.nan)),
            "n_components_before": float(mask_meta.get("n_components_before", np.nan)),
            "n_components_after": float(mask_meta.get("n_components_after", np.nan)),
            "technical_warning": technical_warning,
            "technical_reason": "",
        }

        if detections_df is not None:
            det_row = detections_lookup.get(int(colony_id), {})
            row["yolo_conf"] = float(det_row.get("yolo_conf", np.nan))
            row["x1"] = float(det_row.get("x1", np.nan))
            row["y1"] = float(det_row.get("y1", np.nan))
            row["x2"] = float(det_row.get("x2", np.nan))
            row["y2"] = float(det_row.get("y2", np.nan))
            row["bbox_area_yolo"] = float(det_row.get("bbox_area_yolo", np.nan))
            row["mask_area_raw"] = float(det_row.get("mask_area_raw", np.nan))
            row["mask_area_cleaned"] = float(det_row.get("mask_area_cleaned", np.nan))
            row["mask_area_after_overlap"] = float(det_row.get("mask_area_after_overlap", np.nan))
            row["mask_area_loss_fraction"] = float(det_row.get("mask_area_loss_fraction", np.nan))
            row["original_area"] = float(det_row.get("original_area", row.get("original_area", np.nan)))
            row["cleaned_area"] = float(det_row.get("cleaned_area", row.get("cleaned_area", np.nan)))
            row["area_loss_fraction"] = float(det_row.get("area_loss_fraction", row.get("area_loss_fraction", np.nan)))
            row["n_components_before"] = float(det_row.get("n_components_before", row.get("n_components_before", np.nan)))
            row["n_components_after"] = float(det_row.get("n_components_after", row.get("n_components_after", np.nan)))
            row["mask_fragment_after_overlap"] = bool(
                row.get("mask_fragment_after_overlap", False) or det_row.get("mask_fragment_after_overlap", False)
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    features_df = pd.DataFrame(rows).reset_index(drop=True)

    n = len(features_df)
    if n >= 2:
        coords = features_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        tree = cKDTree(coords)

        for i in range(n):
            k = min(config.neighbor_k + 1, n)
            dists, idxs = tree.query(coords[i], k=k)
            dists = np.atleast_1d(dists).astype(float)
            idxs = np.atleast_1d(idxs).astype(int)

            mask_non_self = idxs != i
            neighbor_dists = dists[mask_non_self]
            neighbor_idxs = idxs[mask_non_self]

            features_df.at[i, "nearest_neighbor_distance"] = (
                float(neighbor_dists[0]) if neighbor_dists.size else np.nan
            )
            features_df.at[i, "mean_5nn_distance"] = (
                float(np.mean(neighbor_dists[:5])) if neighbor_dists.size else np.nan
            )

            density_count = len(
                tree.query_ball_point(coords[i], r=float(config.local_density_radius))
            ) - 1
            features_df.at[i, "local_density_r"] = float(max(0, density_count))

            k_rel = min(config.neighbor_k, n - 1)
            if k_rel >= 1 and neighbor_idxs.size > 0:
                nn = neighbor_idxs[:k_rel]
                neighbor_feature_map = {
                    "relative_area_vs_neighbors": "area",
                    "relative_intensity_vs_neighbors": "intensity_mean",
                    "relative_texture_vs_neighbors": "glcm_contrast",
                    "relative_circularity_vs_neighbors": "circularity",
                    "relative_solidity_vs_neighbors": "solidity",
                    "relative_color_delta_lab_vs_neighbors": "local_color_delta_lab",
                    "relative_local_contrast_vs_neighbors": "local_contrast",
                    "relative_entropy_vs_neighbors": "intensity_entropy",
                }
                for out_col, source_col in neighbor_feature_map.items():
                    if source_col not in features_df.columns:
                        continue
                    cur_value = float(features_df.at[i, source_col])
                    neighbor_values = features_df.loc[nn, source_col].to_numpy(dtype=float)
                    features_df.at[i, out_col] = _robust_neighbor_delta(cur_value, neighbor_values, eps=eps)

    plate_area_median = np.nanmedian(features_df["area"].to_numpy(dtype=float))
    plate_equivalent_diameter_median = np.nanmedian(features_df["equivalent_diameter"].to_numpy(dtype=float))
    plate_intensity_median = np.nanmedian(
        features_df["intensity_mean"].to_numpy(dtype=float)
    )
    plate_texture_median = np.nanmedian(
        features_df["glcm_contrast"].to_numpy(dtype=float)
    )

    features_df["log_area"] = np.log1p(features_df["area"].astype(float))
    features_df["area_to_median_ratio"] = (
        features_df["area"].astype(float) / (plate_area_median + eps)
        if np.isfinite(plate_area_median) and abs(plate_area_median) > eps
        else np.nan
    )
    features_df["equivalent_diameter_to_median_ratio"] = (
        features_df["equivalent_diameter"].astype(float) / (plate_equivalent_diameter_median + eps)
        if np.isfinite(plate_equivalent_diameter_median) and abs(plate_equivalent_diameter_median) > eps
        else np.nan
    )
    features_df["relative_area_vs_plate_median"] = features_df["area_to_median_ratio"]
    features_df["relative_intensity_vs_plate_median"] = (
        features_df["intensity_mean"] / (plate_intensity_median + eps)
        if np.isfinite(plate_intensity_median) and abs(plate_intensity_median) > eps
        else np.nan
    )
    features_df["relative_texture_vs_plate_median"] = (
        features_df["glcm_contrast"] / (plate_texture_median + eps)
        if np.isfinite(plate_texture_median) and abs(plate_texture_median) > eps
        else np.nan
    )

    features_df["log_nearest_neighbor_distance"] = np.log1p(
        features_df["nearest_neighbor_distance"].replace([np.inf, -np.inf], np.nan).astype(float)
    )
    features_df["log_mean_5nn_distance"] = np.log1p(
        features_df["mean_5nn_distance"].replace([np.inf, -np.inf], np.nan).astype(float)
    )

    if n >= 4:
        features_df["voronoi_area"] = _compute_voronoi_areas(
            features_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float),
            image_shape=(h, w),
        )

    features_df = _add_feature_knn_distance(features_df, config)

    median_area = plate_area_median if len(features_df) else np.nan
    min_area_threshold = float(config.min_area)
    if np.isfinite(median_area):
        min_area_threshold = max(
            min_area_threshold,
            float(median_area) * float(config.min_area_relative_to_median),
        )
    max_area_threshold = float(config.max_area_fraction) * image_area
    if np.isfinite(median_area) and median_area > 0:
        max_area_threshold = min(
            max_area_threshold,
            float(median_area) * float(config.max_area_relative_to_median),
        )

    features_df["is_too_small"] = features_df["area"].astype(float) < min_area_threshold
    features_df["is_too_large"] = features_df["area"].astype(float) > max_area_threshold

    if "yolo_conf" not in features_df.columns:
        features_df["yolo_conf"] = np.nan
    features_df["low_yolo_conf"] = (
        features_df["yolo_conf"].fillna(1.0).astype(float) < float(config.min_yolo_conf_for_analysis)
    )

    for col, default in {
        "poor_local_background": False,
        "mask_fragment_after_overlap": False,
        "invalid_geometry": False,
        "touches_image_border": False,
        "suspicious_aspect_ratio": False,
    }.items():
        if col not in features_df.columns:
            features_df[col] = default

    technical_flags = (
        features_df["is_too_small"].fillna(True).astype(bool)
        | features_df["is_too_large"].fillna(True).astype(bool)
        | features_df["suspicious_aspect_ratio"].fillna(False).astype(bool)
        | features_df["low_yolo_conf"].fillna(False).astype(bool)
        | features_df["poor_local_background"].fillna(False).astype(bool)
        | features_df["mask_fragment_after_overlap"].fillna(False).astype(bool)
        | features_df["invalid_geometry"].fillna(False).astype(bool)
    )
    if config.exclude_border_touching_from_anomaly:
        technical_flags = technical_flags | features_df["touches_image_border"].fillna(False).astype(bool)

    features_df["technical_warning"] = technical_flags.astype(bool)
    features_df["technical_reason"] = features_df.apply(_build_technical_reason, axis=1)
    features_df.loc[~features_df["technical_warning"].astype(bool), "technical_reason"] = ""
    features_df["valid_for_anomaly"] = ~features_df["technical_warning"].astype(bool)

    if detections_df is not None:
        for col in [
            "yolo_conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "bbox_area_yolo",
            "mask_area_raw",
            "mask_area_cleaned",
            "mask_area_after_overlap",
            "mask_area_loss_fraction",
            "original_area",
            "cleaned_area",
            "area_loss_fraction",
            "n_components_before",
            "n_components_after",
        ]:
            if col not in features_df.columns:
                features_df[col] = np.nan

    features_df = _augment_features_after_extraction(
        features_df=features_df,
        image_rgb=image_rgb,
        masks_by_colony_id=masks_by_colony_id,
        gray=gray,
        lab=lab,
        config=config,
        image_area=image_area,
    )

    return features_df

FEATURE_GROUPS = {
    "size_shape_score": [
        "log_area",
        "equivalent_diameter",
        "area_to_median_ratio",
        "aspect_ratio",
        "circularity",
        "eccentricity",
        "solidity",
        "convexity_defect_ratio",
        "radial_contour_cv",
    ],
    "color_background_score": [
        "mean_L_lab",
        "mean_a_lab",
        "mean_b_lab",
        "std_L_lab",
        "std_a_lab",
        "std_b_lab",
        "mean_S",
        "mean_V",
        "mean_H_sin",
        "mean_H_cos",
        "local_delta_L",
        "local_delta_a",
        "local_delta_b",
        "local_color_delta_lab",
    ],
    "intensity_score": [
        "intensity_mean",
        "intensity_median",
        "intensity_iqr",
        "intensity_p95_p05_range",
        "intensity_cv",
        "intensity_entropy",
        "center_rim_intensity_delta",
        "center_rim_intensity_ratio",
        "radial_intensity_slope",
        "radial_intensity_std",
    ],
    "texture_score": [
        "laplacian_var",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        "glcm_dissimilarity",
        "glcm_ASM",
        "lbp_mean",
        "lbp_std",
        "lbp_entropy",
    ],
    "histogram_score": [
        "hist_intensity_entropy",
        "hist_intensity_width",
        "hist_intensity_skewness",
        "hist_intensity_kurtosis",
        "dark_fraction",
        "bright_fraction",
        "contrast_hist_entropy",
        "intensity_hist_js_to_plate_median",
        "intensity_hist_wasserstein_to_plate_median",
        "intensity_hist_js_to_neighbors",
        "L_hist_js_to_plate_median",
        "contrast_hist_js_to_plate_median",
    ],
    "spatial_context_score": [
        "log_nearest_neighbor_distance",
        "log_mean_5nn_distance",
        "edge_nearest_distance",
        "mean_5nn_edge_distance",
        "nearest_distance_to_median_diameter_ratio",
        "edge_distance_to_median_diameter_ratio",
        "local_density_r",
    ],
    "neighbor_difference_score": [
        "relative_area_vs_neighbors",
        "relative_intensity_vs_neighbors",
        "relative_texture_vs_neighbors",
        "relative_circularity_vs_neighbors",
        "relative_solidity_vs_neighbors",
        "relative_color_delta_lab_vs_neighbors",
        "relative_local_contrast_vs_neighbors",
        "relative_entropy_vs_neighbors",
        "relative_histogram_vs_neighbors",
        "feature_knn_distance",
    ],
    "morphotype_score": [
        "within_morphotype_anomaly_score",
        "size_within_morphotype_z",
        "texture_within_morphotype_z",
        "color_within_morphotype_z",
        "histogram_within_morphotype_z",
    ],
}

EXCLUDE_FROM_ANOMALY = {
    "colony_id",
    "technical_warning",
    "valid_for_anomaly",
    "selected_for_further_analysis",
    "technical_reason",
    "recommendation_status",
    "anomaly_type",
    "explanations",
    "explanations_text",
    "touches_image_border",
    "is_too_small",
    "is_too_large",
    "suspicious_aspect_ratio",
    "low_yolo_conf",
    "poor_local_background",
    "critical_poor_local_background",
    "mask_fragment_after_overlap",
    "possible_merged_colony",
    "possible_segmentation_artifact",
    "invalid_geometry",
    "rare_morphotype_flag",
    "yolo_conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_area_yolo",
    "mask_area_raw",
    "mask_area_cleaned",
    "mask_area_after_overlap",
    "mask_area_loss_fraction",
    "mask_area_input",
    "mask_area_after_feature_cleaning",
    "mask_area_feature_loss_fraction",
    "original_area",
    "cleaned_area",
    "area_loss_fraction",
    "n_components_before",
    "n_components_after",
    "centroid_x",
    "centroid_y",
    "centroid_x_norm",
    "centroid_y_norm",
    "orientation",
    "mean_H",
    "std_H",
    "intensity_min",
    "intensity_max",
    "distance_to_plate_center",
    "distance_to_plate_edge",
}


def _prepare_numeric_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        return np.empty((len(df), 0), dtype=float), []
    X = df[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite_columns = np.any(np.isfinite(X), axis=0)
    columns = [c for c, keep in zip(columns, finite_columns) if keep]
    X = X[:, finite_columns]
    if X.shape[1] == 0:
        return np.empty((len(df), 0), dtype=float), []
    try:
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = RobustScaler().fit_transform(X)
    except Exception:
        return np.empty((len(df), 0), dtype=float), []
    return X, columns


def _robust_outlier_score(X: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=float)
    eps = 1e-9
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.where(np.isfinite(mad) & (mad >= eps), mad, eps)
    z = np.abs((X - med) / (1.4826 * mad))
    z = np.where(np.isfinite(z), z, 0.0)
    return 0.7 * np.nanmean(z, axis=1) + 0.3 * np.nanmax(z, axis=1)


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.zeros(values.shape[0], dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return ranks
    ranks[finite] = pd.Series(values[finite]).rank(method="average", pct=True).to_numpy(dtype=float)
    return ranks


def _score_has_signal(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return bool(values.size >= 2 and np.nanstd(values) >= 1e-8)


def assign_morphotypes(features_df: pd.DataFrame, config: AnomalyDetectionConfig) -> pd.DataFrame:
    """Выделить морфологические подгруппы только при доказуемой кластерной структуре.

    Морфотип здесь не является видом бактерий. Это вычислимая группа колоний,
    похожих по размеру, форме, яркости, цвету и текстуре. Если разделение слабое
    (низкий silhouette), все объекты остаются в одном морфотипе.
    """
    df = features_df.copy()
    n = len(df)
    df["morphotype_id"] = 0
    df["morphotype_size"] = n
    df["morphotype_fraction"] = 1.0 if n else np.nan
    df["rare_morphotype_flag"] = False
    df["within_morphotype_anomaly_score"] = 0.0
    df["size_within_morphotype_z"] = np.nan
    df["texture_within_morphotype_z"] = np.nan
    df["color_within_morphotype_z"] = np.nan
    df["histogram_within_morphotype_z"] = np.nan
    df["morphotype_silhouette"] = np.nan

    if not bool(getattr(config, "use_morphotype_clustering", True)) or n == 0:
        return df

    valid_mask = df.get("valid_for_anomaly", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    if "technical_warning" in df.columns:
        valid_mask = valid_mask & (~df["technical_warning"].fillna(False).astype(bool))
    valid_idx = np.where(valid_mask.to_numpy())[0]
    n_valid = len(valid_idx)
    if n_valid < 10:
        df.loc[valid_idx, "morphotype_size"] = n_valid
        df.loc[valid_idx, "morphotype_fraction"] = 1.0 if n_valid else np.nan
        return df

    core_cols = [
        "log_area",
        "circularity",
        "eccentricity",
        "solidity",
        "intensity_mean",
        "intensity_iqr",
        "mean_L_lab",
        "mean_a_lab",
        "mean_b_lab",
        "glcm_contrast",
        "lbp_entropy",
        "local_color_delta_lab",
        "intensity_hist_js_to_plate_median",
    ]
    valid_df = df.iloc[valid_idx].copy()
    X, used_cols = _prepare_numeric_matrix(valid_df, core_cols)
    if X.shape[1] == 0:
        return df

    max_k = min(int(config.max_morphotypes), max(2, int(np.sqrt(n_valid)) + 1), n_valid - 1)
    best_labels = None
    best_score = -np.inf

    if str(config.morphotype_method).lower() == "dbscan":
        try:
            labels = DBSCAN(eps=float(config.dbscan_eps), min_samples=max(3, int(config.min_morphotype_size))).fit_predict(X)
            non_noise = labels[labels >= 0]
            if len(np.unique(non_noise)) >= 2:
                best_labels = labels
                best_score = float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else -np.inf
        except Exception:
            best_labels = None

    if best_labels is None:
        for k in range(2, max_k + 1):
            try:
                labels = KMeans(n_clusters=k, random_state=config.random_state, n_init=10).fit_predict(X)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(X, labels) if n_valid > k else -np.inf
                if np.isfinite(score) and score > best_score:
                    best_score = float(score)
                    best_labels = labels
            except Exception:
                continue

    # Если разделение слабое, не создаём искусственные морфотипы.
    min_sil = float(getattr(config, "min_morphotype_silhouette", 0.20))
    if best_labels is None or not np.isfinite(best_score) or best_score < min_sil:
        df.loc[valid_idx, "morphotype_id"] = 0
        df.loc[valid_idx, "morphotype_size"] = n_valid
        df.loc[valid_idx, "morphotype_fraction"] = 1.0
        df.loc[valid_idx, "morphotype_silhouette"] = best_score if np.isfinite(best_score) else np.nan
        return df

    labels = np.asarray(best_labels, dtype=int)
    if np.any(labels < 0):
        labels = labels.copy()
        labels[labels < 0] = labels.max() + 1

    df.loc[valid_idx, "morphotype_id"] = labels
    df.loc[valid_idx, "morphotype_silhouette"] = best_score

    for lab_id in np.unique(labels):
        member_local = np.where(labels == lab_id)[0]
        member_global = valid_idx[member_local]
        size = len(member_global)
        fraction = size / max(1, n_valid)
        rare = bool(size < int(config.min_morphotype_size) or fraction < float(config.rare_morphotype_fraction))
        df.loc[member_global, "morphotype_size"] = size
        df.loc[member_global, "morphotype_fraction"] = fraction
        df.loc[member_global, "rare_morphotype_flag"] = rare

        if size >= 3:
            X_cluster = X[member_local]
            df.loc[member_global, "within_morphotype_anomaly_score"] = _robust_outlier_score(X_cluster)
            for out_col, source_cols in {
                "size_within_morphotype_z": ["log_area"],
                "texture_within_morphotype_z": ["glcm_contrast", "lbp_entropy"],
                "color_within_morphotype_z": ["local_color_delta_lab", "mean_L_lab", "mean_a_lab", "mean_b_lab"],
                "histogram_within_morphotype_z": ["intensity_hist_js_to_plate_median", "hist_intensity_entropy"],
            }.items():
                vals = []
                for global_i in member_global:
                    z_parts = []
                    for source_col in source_cols:
                        if source_col in df.columns:
                            z = _robust_z_value(float(df.at[global_i, source_col]), df.loc[member_global, source_col].to_numpy(dtype=float))
                            if np.isfinite(z):
                                z_parts.append(abs(z))
                    vals.append(float(np.mean(z_parts)) if z_parts else np.nan)
                df.loc[member_global, out_col] = vals
    return df


def compute_weighted_final_anomaly_score(
    results_df: pd.DataFrame,
    valid_indices: np.ndarray,
    config: AnomalyDetectionConfig,
) -> pd.DataFrame:
    if valid_indices.size == 0:
        return results_df
    weights = {
        "lof_score_rank": 0.18,
        "isolation_forest_score_rank": 0.12,
        "robust_z_score_rank": 0.12,
        "size_shape_score_rank": 0.15,
        "texture_score_rank": 0.15,
        "intensity_score_rank": 0.10,
        "color_background_score_rank": 0.08,
        "histogram_score_rank": 0.08,
        "neighbor_difference_score_rank": float(getattr(config, "neighbor_difference_weight", 0.05)),
        "spatial_context_score_rank": float(getattr(config, "spatial_context_weight", 0.03)),
        "morphotype_score_rank": float(getattr(config, "morphotype_weight", 0.08)),
        "cluster_outlier_score_rank": 0.03,
    }
    used_values = []
    used_weights = []
    for rank_col, weight in weights.items():
        score_col = rank_col[:-5] if rank_col.endswith("_rank") else rank_col
        if rank_col not in results_df.columns:
            continue
        score_values = results_df.loc[valid_indices, score_col].to_numpy(dtype=float) if score_col in results_df.columns else results_df.loc[valid_indices, rank_col].to_numpy(dtype=float)
        if not _score_has_signal(score_values):
            continue
        ranks = results_df.loc[valid_indices, rank_col].to_numpy(dtype=float)
        if not _score_has_signal(ranks):
            continue
        used_values.append(ranks)
        used_weights.append(float(weight))
    if not used_values:
        results_df.loc[valid_indices, "final_anomaly_score_raw"] = 0.0
        results_df.loc[valid_indices, "final_anomaly_score_percentile"] = 0.0
        results_df.loc[valid_indices, "final_anomaly_score"] = 0.0
        return results_df
    w = np.asarray(used_weights, dtype=float)
    w = w / max(float(w.sum()), 1e-12)
    raw = np.average(np.vstack(used_values), axis=0, weights=w) if bool(config.use_weighted_final_score) else np.nanmean(np.vstack(used_values), axis=0)
    raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    percentile = _rank01(raw) if _score_has_signal(raw) else raw.copy()
    results_df.loc[valid_indices, "final_anomaly_score_raw"] = raw
    results_df.loc[valid_indices, "final_anomaly_score_percentile"] = percentile
    # Use raw for objective selection; percentile is stored only as within-plate context.
    results_df.loc[valid_indices, "final_anomaly_score"] = raw
    return results_df


def _assign_anomaly_types(results_df: pd.DataFrame) -> pd.Series:
    type_map = {
        "size_shape_score_rank": "size_shape",
        "texture_score_rank": "texture",
        "color_background_score_rank": "color_background",
        "intensity_score_rank": "intensity",
        "histogram_score_rank": "histogram",
        "spatial_context_score_rank": "spatial_isolation",
        "neighbor_difference_score_rank": "neighbor_difference",
        "morphotype_score_rank": "morphotype",
    }
    labels: list[str] = []
    for _, row in results_df.iterrows():
        if bool(row.get("technical_warning", False)) or not bool(row.get("valid_for_anomaly", False)):
            labels.append("technical_warning")
            continue
        scored = []
        for col, label_name in type_map.items():
            value = row.get(col, np.nan)
            if pd.notna(value) and np.isfinite(float(value)):
                scored.append((float(value), label_name))
        if not scored:
            labels.append("mixed")
            continue
        scored.sort(reverse=True, key=lambda x: x[0])
        best_value, best_label = scored[0]
        second_value = scored[1][0] if len(scored) > 1 else 0.0
        labels.append(best_label if best_value >= 0.65 and (best_value - second_value) >= 0.12 else "mixed")
    return pd.Series(labels, index=results_df.index, dtype=object)


def _compute_evidence_reliability_and_status(
    results_df: pd.DataFrame,
    valid_indices: np.ndarray,
    config: AnomalyDetectionConfig,
) -> pd.DataFrame:
    """Рассчитать доказательность, согласованность, надёжность и финальный статус.

    В отличие от чистого rank-based top-N, статус select_candidate требует:
    - высокий raw-score;
    - абсолютную выраженность отклонений;
    - минимум несколько независимых групп признаков;
    - достаточную надёжность маски/фона/текстуры;
    - согласованность нескольких методов.
    """
    evidence_rank_cols = [
        "size_shape_score_rank",
        "texture_score_rank",
        "intensity_score_rank",
        "color_background_score_rank",
        "histogram_score_rank",
        "neighbor_difference_score_rank",
        "morphotype_score_rank",
    ]
    evidence_score_cols = [c[:-5] for c in evidence_rank_cols if c.endswith("_rank")]
    consensus_rank_cols = [
        "lof_score_rank",
        "isolation_forest_score_rank",
        "robust_z_score_rank",
        "size_shape_score_rank",
        "texture_score_rank",
        "intensity_score_rank",
        "color_background_score_rank",
        "histogram_score_rank",
        "morphotype_score_rank",
    ]

    results_df["absolute_evidence_strength"] = 0.0
    results_df["independent_evidence_count"] = 0
    results_df["consensus_score"] = 0.0
    results_df["ablation_stability_score"] = 0.0
    results_df["segmentation_reliability"] = 0.0
    results_df["texture_reliability"] = 0.0
    results_df["background_reliability"] = 0.0
    results_df["shape_reliability"] = 0.0
    results_df["overall_reliability_score"] = 0.0
    results_df["recommendation_status"] = "technical_exclude"

    if valid_indices.size == 0:
        return results_df

    # Absolute evidence: robust z of group scores among valid objects, clipped to 0..1.
    absolute_components: dict[str, np.ndarray] = {}
    for score_col in evidence_score_cols:
        if score_col not in results_df.columns:
            continue
        vals = results_df.loc[valid_indices, score_col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
        mad = _finite_mad(vals)
        if not np.isfinite(med) or not np.isfinite(mad) or mad <= 0:
            continue
        z = (vals - med) / (1.4826 * mad + 1e-8)
        absolute_components[score_col] = np.clip(np.abs(z), 0, 5) / 5.0

    if absolute_components:
        abs_matrix = np.vstack(list(absolute_components.values()))
        results_df.loc[valid_indices, "absolute_evidence_strength"] = np.nanmean(abs_matrix, axis=0)

    for pos, idx in enumerate(valid_indices):
        row = results_df.loc[idx]

        evidence_count = 0
        evidence_values_for_ablation = []
        for rank_col in evidence_rank_cols:
            score_col = rank_col[:-5] if rank_col.endswith("_rank") else rank_col
            rank_val = float(row.get(rank_col, 0.0)) if pd.notna(row.get(rank_col, np.nan)) else 0.0

            # Spatial context is not treated as a strong independent proof.
            if rank_col == "spatial_context_score_rank":
                continue

            abs_component = np.nan
            if score_col in absolute_components:
                abs_component = float(absolute_components[score_col][pos])
            is_evidence = bool(rank_val >= 0.90 and (np.isfinite(abs_component) and abs_component >= 0.40))
            if is_evidence:
                evidence_count += 1
            if np.isfinite(rank_val):
                evidence_values_for_ablation.append(rank_val)

        results_df.at[idx, "independent_evidence_count"] = int(evidence_count)
        results_df.at[idx, "ablation_stability_score"] = float(evidence_count / max(1, len(evidence_values_for_ablation)))

        consensus_parts = []
        for col in consensus_rank_cols:
            if col in results_df.columns and pd.notna(row.get(col)):
                threshold = 0.95 if col in ["lof_score_rank", "isolation_forest_score_rank", "robust_z_score_rank"] else 0.90
                value = float(row.get(col))
                if np.isfinite(value):
                    consensus_parts.append(float(value >= threshold))
        results_df.at[idx, "consensus_score"] = float(np.mean(consensus_parts)) if consensus_parts else 0.0

        yolo_conf = row.get("yolo_conf", np.nan)
        seg_rel = float(np.clip(float(yolo_conf) / 0.5, 0.0, 1.0)) if pd.notna(yolo_conf) and np.isfinite(float(yolo_conf)) else 0.8

        area = float(row.get("area", 0.0)) if pd.notna(row.get("area", np.nan)) else 0.0
        texture_rel = float(np.clip(area / 100.0, 0.0, 1.0))

        bg_area = float(row.get("local_background_area", 0.0)) if pd.notna(row.get("local_background_area", np.nan)) else 0.0
        bg_rel = float(np.clip(bg_area / max(50.0, 0.2 * max(area, 1.0)), 0.0, 1.0))
        if bool(row.get("poor_local_background", False)):
            bg_rel *= 0.5
        if bool(row.get("critical_poor_local_background", False)):
            bg_rel = 0.0

        shape_rel = 0.6 if bool(row.get("possible_merged_colony", False)) else 1.0
        if bool(row.get("technical_warning", False)):
            seg_rel = texture_rel = bg_rel = shape_rel = 0.0

        overall = float(np.nanmean([seg_rel, texture_rel, bg_rel, shape_rel]))
        results_df.at[idx, "segmentation_reliability"] = seg_rel
        results_df.at[idx, "texture_reliability"] = texture_rel
        results_df.at[idx, "background_reliability"] = bg_rel
        results_df.at[idx, "shape_reliability"] = shape_rel
        results_df.at[idx, "overall_reliability_score"] = overall

    for i, row in results_df.iterrows():
        raw_score = float(row.get("final_anomaly_score_raw", 0.0))
        abs_strength = float(row.get("absolute_evidence_strength", 0.0))
        within = row.get("within_morphotype_anomaly_score", np.nan)
        within_val = float(within) if pd.notna(within) and np.isfinite(float(within)) else np.nan

        if bool(row.get("technical_warning", False)) or not bool(row.get("valid_for_anomaly", False)):
            status = "review_segmentation" if _is_segmentation_review_candidate(row, config) else "technical_exclude"
        elif bool(row.get("possible_merged_colony", False)) and raw_score >= 0.60:
            status = "review_segmentation"
        elif bool(row.get("rare_morphotype_flag", False)) and (not np.isfinite(within_val) or within_val < 0.75):
            status = "rare_morphotype"
        elif (
            raw_score >= float(config.min_final_score_raw)
            and abs_strength >= float(getattr(config, "min_absolute_evidence_strength", 0.35))
            and int(row.get("independent_evidence_count", 0)) >= int(config.min_independent_evidence_count)
            and float(row.get("overall_reliability_score", 0.0)) >= float(config.min_reliability_score)
            and float(row.get("consensus_score", 0.0)) >= float(config.min_consensus_score)
        ):
            status = "select_candidate"
        elif raw_score >= float(config.min_final_score_raw):
            status = "review_only"
        elif float(row.get("final_anomaly_score_percentile", 0.0)) >= 0.90:
            status = "weak_outlier"
        else:
            status = "no_valid_evidence"
        results_df.at[i, "recommendation_status"] = status

    results_df["selected_for_further_analysis"] = results_df["recommendation_status"].eq("select_candidate")
    return results_df


def compute_anomaly_scores(
    features_df: pd.DataFrame,
    config: AnomalyDetectionConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = AnomalyDetectionConfig()
    if features_df is None:
        return pd.DataFrame()
    if features_df.empty:
        return features_df.copy()

    results_df = assign_morphotypes(features_df.copy(), config)
    n_total = len(results_df)

    score_columns = list(FEATURE_GROUPS.keys()) + [
        "robust_z_score",
        "lof_score",
        "isolation_forest_score",
        "cluster_outlier_score",
    ]
    for col in score_columns:
        results_df[col] = 0.0
        results_df[f"{col}_rank"] = 0.0
    results_df["final_anomaly_score_raw"] = 0.0
    results_df["final_anomaly_score_percentile"] = 0.0
    results_df["final_anomaly_score"] = 0.0
    results_df["global_anomaly_score"] = 0.0
    results_df["absolute_evidence_strength"] = 0.0
    results_df["anomaly_rank"] = pd.NA
    results_df["selected_for_further_analysis"] = False

    valid_mask = results_df.get("valid_for_anomaly", pd.Series(True, index=results_df.index)).fillna(False).astype(bool).to_numpy()
    if bool(getattr(config, "exclude_technical_from_anomaly", True)) and "technical_warning" in results_df.columns:
        valid_mask = valid_mask & (~results_df["technical_warning"].fillna(False).astype(bool).to_numpy())
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        results_df["anomaly_type"] = _assign_anomaly_types(results_df)
        results_df = _compute_evidence_reliability_and_status(results_df, valid_indices, config)
        return results_df.sort_values("final_anomaly_score", ascending=False).reset_index(drop=True)

    valid_df = results_df.iloc[valid_indices].copy()
    n_colonies = len(valid_df)

    for group_name, cols in FEATURE_GROUPS.items():
        X_group, used_cols = _prepare_numeric_matrix(valid_df, [c for c in cols if c in valid_df.columns])
        group_score = _robust_outlier_score(X_group) if used_cols else np.zeros(n_colonies, dtype=float)
        results_df.loc[valid_indices, group_name] = group_score

    numeric_columns = [
        c for c in valid_df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_FROM_ANOMALY
        and not c.endswith("_score")
        and not c.endswith("_rank")
        and not c.startswith("final_anomaly_score")
    ]
    X_scaled, numeric_columns = _prepare_numeric_matrix(valid_df, numeric_columns)
    results_df.loc[valid_indices, "robust_z_score"] = _robust_outlier_score(X_scaled)
    results_df.loc[valid_indices, "global_anomaly_score"] = results_df.loc[valid_indices, "robust_z_score"].to_numpy(dtype=float)

    if n_colonies < 5 or X_scaled.shape[1] == 0:
        lof_score = np.zeros(n_colonies, dtype=float)
    else:
        try:
            n_neighbors = max(2, min(int(config.lof_neighbors), n_colonies - 1))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=min(max(config.contamination, 1e-3), 0.5))
            lof.fit_predict(X_scaled)
            lof_score = -lof.negative_outlier_factor_
        except Exception:
            lof_score = np.zeros(n_colonies, dtype=float)
    results_df.loc[valid_indices, "lof_score"] = lof_score

    if n_colonies < 8 or X_scaled.shape[1] == 0:
        iso_score = np.zeros(n_colonies, dtype=float)
    else:
        try:
            iso = IsolationForest(contamination=min(max(config.contamination, 1e-3), 0.5), random_state=config.random_state)
            iso.fit(X_scaled)
            iso_score = -iso.score_samples(X_scaled)
        except Exception:
            iso_score = np.zeros(n_colonies, dtype=float)
    results_df.loc[valid_indices, "isolation_forest_score"] = iso_score

    if n_colonies < 8 or X_scaled.shape[1] == 0:
        cluster_score = np.zeros(n_colonies, dtype=float)
    else:
        X_cluster = X_scaled
        try:
            n_components = min(5, X_cluster.shape[1], n_colonies - 1)
            if 1 <= n_components < X_cluster.shape[1]:
                X_cluster = PCA(n_components=n_components, random_state=config.random_state).fit_transform(X_cluster)
        except Exception:
            X_cluster = X_scaled
        try:
            min_samples = int(config.dbscan_min_samples) if config.dbscan_min_samples is not None else max(3, min(10, n_colonies // 20))
            min_samples = min(max(2, min_samples), n_colonies)
            labels = DBSCAN(eps=float(config.dbscan_eps), min_samples=min_samples).fit_predict(X_cluster)
            unique_non_noise = [lab for lab in np.unique(labels) if lab != -1]
            use_kmeans = np.all(labels == -1) or len(unique_non_noise) <= 1
        except Exception:
            labels = np.full(n_colonies, -1)
            use_kmeans = True
        if not use_kmeans:
            cluster_score = np.zeros(n_colonies, dtype=float)
            max_cluster_dist = 0.0
            for lab_id in [lab for lab in np.unique(labels) if lab != -1]:
                idx = np.where(labels == lab_id)[0]
                center = X_cluster[idx].mean(axis=0)
                d = np.linalg.norm(X_cluster[idx] - center, axis=1)
                cluster_score[idx] = d
                if d.size:
                    max_cluster_dist = max(max_cluster_dist, float(np.max(d)))
            noise_idx = np.where(labels == -1)[0]
            if noise_idx.size:
                cluster_score[noise_idx] = max_cluster_dist + 1.0
        else:
            try:
                k = min(3, max(2, int(np.sqrt(n_colonies))), n_colonies - 1)
                if k < 2:
                    cluster_score = np.zeros(n_colonies, dtype=float)
                else:
                    km = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
                    km_labels = km.fit_predict(X_cluster)
                    cluster_score = np.linalg.norm(X_cluster - km.cluster_centers_[km_labels], axis=1)
            except Exception:
                cluster_score = np.zeros(n_colonies, dtype=float)
    results_df.loc[valid_indices, "cluster_outlier_score"] = cluster_score

    for score_col in score_columns:
        values = results_df.loc[valid_indices, score_col].to_numpy(dtype=float)
        if _score_has_signal(values):
            results_df.loc[valid_indices, f"{score_col}_rank"] = _rank01(values)

    results_df = compute_weighted_final_anomaly_score(results_df, valid_indices, config)
    final_values = results_df.loc[valid_indices, "final_anomaly_score_raw"].to_numpy(dtype=float)
    if final_values.size:
        results_df.loc[valid_indices, "anomaly_rank"] = pd.Series(final_values).rank(method="dense", ascending=False).astype(int).to_numpy()
    results_df["anomaly_type"] = _assign_anomaly_types(results_df)
    results_df = _compute_evidence_reliability_and_status(results_df, valid_indices, config)

    STATUS_PRIORITY = {
        "select_candidate": 0,
        "review_segmentation": 1,
        "unstable_candidate": 2,
        "review_only": 3,
        "rare_morphotype": 4,
        "weak_outlier": 5,
        "no_valid_evidence": 6,
        "technical_exclude": 7,
    }
    results_df["status_priority"] = results_df["recommendation_status"].map(STATUS_PRIORITY).fillna(99).astype(int)
    results_df = results_df.sort_values(
        ["valid_for_anomaly", "status_priority", "final_anomaly_score_raw"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return results_df

def explain_anomaly(
    row: pd.Series,
    features_df: pd.DataFrame,
    top_k: int = 5,
) -> list[str]:
    """Сформировать русскоязычные причины аномальности с числовым контекстом."""
    technical_reason_map = {
        "too_small": "маска слишком мала и похожа на технический фрагмент сегментации",
        "too_large": "маска слишком велика относительно других объектов на чашке",
        "touches_border": "маска касается границы изображения, объект может быть обрезан",
        "suspicious_aspect_ratio": "маска имеет подозрительно вытянутую форму",
        "low_yolo_conf": "низкая уверенность YOLO для этой маски",
        "poor_local_background": "локальный фон вокруг колонии ограничен, интерпретация контрастных признаков менее надёжна",
        "critical_poor_local_background": "локальный фон вокруг колонии практически отсутствует, объект исключён из автоматического отбора",
        "mask_fragment_after_overlap": "маска стала фрагментом после удаления пересечений с другими масками",
        "invalid_geometry": "геометрия маски некорректна для интерпретации как отдельной колонии",
        "possible_segmentation_artifact": "объект похож на технический артефакт сегментации",
    }

    fallback = "интегральный показатель аномальности повышен за счёт совокупности слабых отклонений"

    if bool(row.get("technical_warning", False)) or not bool(row.get("valid_for_anomaly", True)):
        raw_reasons = str(row.get("technical_reason", "") or "")
        reasons = [technical_reason_map.get(r.strip(), r.strip()) for r in raw_reasons.split(";") if r.strip()]
        default_reason = "маска имеет техническое предупреждение, результат требует проверки"
        if default_reason not in reasons:
            reasons.insert(0, default_reason)
        return reasons[: max(1, int(top_k))]

    if features_df is None or features_df.empty:
        return [fallback]

    valid_df = features_df.copy()
    if "valid_for_anomaly" in valid_df.columns:
        valid_df = valid_df[valid_df["valid_for_anomaly"].fillna(False).astype(bool)]
    if "technical_warning" in valid_df.columns:
        valid_df = valid_df[~valid_df["technical_warning"].fillna(False).astype(bool)]
    if valid_df.empty:
        valid_df = features_df.copy()

    def _num(value: Any) -> float:
        try:
            value = float(value)
        except Exception:
            return np.nan
        return value if np.isfinite(value) else np.nan

    def _fmt(value: float) -> str:
        value = _num(value)
        if not np.isfinite(value):
            return "nan"
        if abs(value) >= 1000 or (0 < abs(value) < 0.01):
            return f"{value:.3g}"
        return f"{value:.3f}"

    def _context(col: str) -> tuple[float, float, float, float]:
        if col not in valid_df.columns or col not in row.index:
            return np.nan, np.nan, np.nan, np.nan
        value = _num(row.get(col, np.nan))
        values = valid_df[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float).to_numpy()
        if values.size < 3 or not np.isfinite(value):
            return value, np.nan, np.nan, np.nan
        median = float(np.nanmedian(values))
        mad = _finite_mad(values)
        z = float((value - median) / (1.4826 * mad + 1e-8)) if np.isfinite(mad) and mad > 0 else np.nan
        percentile = float(pd.Series(values).rank(pct=True).iloc[np.searchsorted(np.sort(values), value, side="left")] if False else np.mean(values <= value))
        return value, median, z, percentile

    def _reason(col: str, text: str, score: float | None = None) -> tuple[float, str] | None:
        value, median, z, percentile = _context(col)
        if score is None:
            score = abs(z) if np.isfinite(z) else 0.0
        if not np.isfinite(score) or score <= 0:
            return None
        detail = f"{text}: {col}={_fmt(value)}, медиана={_fmt(median)}, robust_z={_fmt(z)}, percentile={_fmt(percentile)}"
        return float(score), detail

    reasons_scored: list[tuple[float, str]] = []

    for col in ["log_area", "area_to_median_ratio"]:
        value, median, z, _ = _context(col)
        if np.isfinite(z) and z > 2.0:
            item = _reason(col, "площадь значительно выше медианы по чашке", abs(z))
            if item:
                reasons_scored.append(item)
        elif np.isfinite(z) and z < -2.0:
            item = _reason(col, "площадь значительно ниже медианы по чашке", abs(z))
            if item:
                reasons_scored.append(item)

    checks = [
        ("circularity", "форма менее округлая по сравнению с большинством колоний", "low", 2.0),
        ("eccentricity", "колония более вытянутая по сравнению с большинством", "high", 2.0),
        ("solidity", "контур колонии более неровный", "low", 2.0),
        ("convexity_defect_ratio", "контур колонии более неровный", "high", 2.0),
        ("radial_contour_cv", "контур колонии более неровный", "high", 2.0),
        ("intensity_iqr", "яркостная неоднородность выше типичного уровня", "high", 2.0),
        ("intensity_entropy", "яркостная неоднородность выше типичного уровня", "high", 2.0),
        ("center_rim_intensity_delta", "колония имеет выраженное отличие яркости между центром и периферией", "abs", 2.0),
        ("radial_intensity_slope", "колония имеет нетипичный радиальный профиль яркости", "abs", 2.0),
        ("glcm_contrast", "текстурная неоднородность выше типичного уровня", "high", 2.0),
        ("lbp_entropy", "текстурная неоднородность выше типичного уровня", "high", 2.0),
        ("hist_intensity_entropy", "гистограмма яркости имеет нетипичную форму", "high", 2.0),
        ("intensity_hist_js_to_plate_median", "гистограмма яркости отличается от типичной гистограммы чашки", "high", 1.5),
        ("intensity_hist_js_to_neighbors", "гистограмма яркости отличается от ближайших соседей", "high", 1.5),
        ("local_color_delta_lab", "цвет сильнее отличается от локального фона", "high", 2.0),
        ("local_delta_L", "яркость отличается от локального фона", "abs", 2.0),
        ("feature_knn_distance", "колония далека от других объектов в пространстве морфологических признаков", "high", 2.0),
    ]
    for col, text, direction, threshold in checks:
        value, median, z, _ = _context(col)
        take = (
            (direction == "high" and np.isfinite(z) and z > threshold)
            or (direction == "low" and np.isfinite(z) and z < -threshold)
            or (direction == "abs" and np.isfinite(z) and abs(z) > threshold)
        )
        if take:
            item = _reason(col, text, abs(z))
            if item:
                reasons_scored.append(item)

    neighbor_reason_map = {
        "relative_area_vs_neighbors": "колония отличается от ближайших соседей по площади",
        "relative_circularity_vs_neighbors": "колония отличается от ближайших соседей по форме",
        "relative_solidity_vs_neighbors": "колония отличается от ближайших соседей по форме",
        "relative_color_delta_lab_vs_neighbors": "колония отличается от ближайших соседей по цвету",
        "relative_entropy_vs_neighbors": "колония отличается от ближайших соседей по текстуре",
        "relative_texture_vs_neighbors": "колония отличается от ближайших соседей по текстуре",
        "relative_histogram_vs_neighbors": "колония отличается от ближайших соседей по гистограмме яркости",
    }
    for col, text in neighbor_reason_map.items():
        value = _num(row.get(col, np.nan))
        if np.isfinite(value) and abs(value) >= 2.0:
            reasons_scored.append((abs(value), f"{text}: {col}={_fmt(value)}"))

    nn_value, nn_median, nn_z, _ = _context("nearest_neighbor_distance")
    if np.isfinite(nn_z) and nn_z > 2.0:
        reasons_scored.append((
            abs(nn_z),
            f"колония пространственно изолирована, но этот признак используется только как дополнительный: nearest_neighbor_distance={_fmt(nn_value)}, медиана={_fmt(nn_median)}, robust_z={_fmt(nn_z)}",
        ))

    if bool(row.get("possible_merged_colony", False)):
        reasons_scored.append((2.0, "форма объекта может соответствовать слипшимся колониям или сомнительной маске"))

    if bool(row.get("rare_morphotype_flag", False)):
        morphotype_id = row.get("morphotype_id", np.nan)
        morphotype_fraction = row.get("morphotype_fraction", np.nan)
        reasons_scored.append((
            1.5,
            f"объект относится к редкому морфотипу, но не обязательно является аномалией: morphotype_id={morphotype_id}, доля={_fmt(morphotype_fraction)}",
        ))

    if bool(row.get("poor_local_background", False)):
        reasons_scored.append((1.0, "локальный фон вокруг колонии ограничен, интерпретация контрастных признаков менее надёжна"))

    if not reasons_scored:
        return [fallback]

    reasons_scored = sorted(reasons_scored, key=lambda x: x[0], reverse=True)
    reasons: list[str] = []
    for _, reason in reasons_scored:
        if reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= max(1, int(top_k)):
            break

    return reasons if reasons else [fallback]

def select_top_anomalies(
    results_df: pd.DataFrame,
    top_k: int | None = None,
    top_percent: float | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Выбрать только подтверждённые кандидаты, не технические и не слабые top-N."""
    if results_df is None:
        return pd.DataFrame()
    if results_df.empty:
        selected = results_df.copy()
        if "selected_for_further_analysis" not in selected.columns:
            selected["selected_for_further_analysis"] = pd.Series(dtype=bool)
        return selected

    score_col = "final_anomaly_score_raw" if "final_anomaly_score_raw" in results_df.columns else "final_anomaly_score"
    if score_col not in results_df.columns:
        raise ValueError("В results_df отсутствует колонка с итоговым anomaly score")

    df = results_df.copy()
    if "valid_for_anomaly" in df.columns:
        df = df[df["valid_for_anomaly"].fillna(False).astype(bool)].copy()
    if "technical_warning" in df.columns:
        df = df[~df["technical_warning"].fillna(False).astype(bool)].copy()
    if "recommendation_status" in df.columns:
        df = df[df["recommendation_status"].eq("select_candidate")].copy()

    if df.empty:
        selected = results_df.head(0).copy()
        selected["selected_for_further_analysis"] = pd.Series(dtype=bool)
        return selected

    df = df.sort_values(score_col, ascending=False, na_position="last").reset_index(drop=True)
    n_valid = int(results_df["valid_for_anomaly"].fillna(False).astype(bool).sum()) if "valid_for_anomaly" in results_df.columns else len(df)

    if top_k is not None:
        n_select = max(1, int(top_k))
        selected = df.head(n_select).copy()
    elif top_percent is not None:
        p = float(top_percent)
        if p > 1.0:
            p = p / 100.0
        p = min(max(p, 0.0), 1.0)
        n_select = max(1, int(math.ceil(n_valid * p)))
        selected = df.head(n_select).copy()
    elif min_score is not None:
        selected = df[df[score_col] >= float(min_score)].copy()
    else:
        if n_valid < 20:
            n_select = 1 if n_valid < 10 else 2
        elif n_valid <= 100:
            n_select = 3
        elif n_valid <= 300:
            n_select = 5
        else:
            n_select = max(5, int(math.ceil(n_valid * 0.02)))
        selected = df.head(n_select).copy()

    selected["selected_for_further_analysis"] = True
    return selected.reset_index(drop=True)


def collect_review_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    """Собрать объекты, которые стоит просмотреть, но не выбирать как подтверждённые аномалии."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    if "recommendation_status" not in results_df.columns:
        return results_df.head(0).copy()
    review_statuses = {"review_only", "review_segmentation", "unstable_candidate", "rare_morphotype", "weak_outlier"}
    df = results_df[results_df["recommendation_status"].isin(review_statuses)].copy()
    score_col = "final_anomaly_score_raw" if "final_anomaly_score_raw" in df.columns else "final_anomaly_score"
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=False, na_position="last")
    return df.reset_index(drop=True)


VISUAL_HIGHLIGHT_STATUSES: tuple[str, ...] = (
    "select_candidate",
    "review_segmentation",
    "unstable_candidate",
    "review_only",
    "rare_morphotype",
    "weak_outlier",
)

VISUAL_HIGHLIGHT_STATUS_PRIORITY: dict[str, int] = {
    "select_candidate": 0,
    "unstable_candidate": 1,
    "rare_morphotype": 2,
    "review_only": 3,
    "weak_outlier": 4,
    "review_segmentation": 5,
}


def visual_highlight_limit(
    n_colonies: int,
    max_fraction: float | None = 0.20,
    max_count: int | None = 20,
    min_count: int = 0,
) -> int:
    """Общий лимит визуально выделяемых объектов на изображении."""
    if n_colonies <= 0:
        return 0

    if max_fraction is not None:
        fraction = float(max_fraction)
        if fraction > 1.0:
            fraction = fraction / 100.0
        fraction = min(max(fraction, 0.0), 1.0)
        limit = int(math.floor(int(n_colonies) * fraction))
    elif max_count is not None:
        limit = int(max_count)
    else:
        limit = int(n_colonies)

    limit = max(int(min_count), int(limit))
    if max_count is not None:
        limit = min(limit, int(max_count))
    return max(0, min(limit, int(n_colonies)))


def _visual_diversity_radius(df: pd.DataFrame, radius_factor: float = 2.5, min_distance_px: float | None = None) -> float | None:
    if min_distance_px is not None and np.isfinite(float(min_distance_px)) and float(min_distance_px) > 0:
        return float(min_distance_px)
    if df is None or df.empty:
        return None

    radius = np.nan
    if "equivalent_diameter" in df.columns:
        diam = pd.to_numeric(df["equivalent_diameter"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not diam.empty:
            radius = float(np.nanmedian(diam.to_numpy(dtype=float))) * float(radius_factor)
    if (not np.isfinite(radius) or radius <= 0) and "nearest_neighbor_distance" in df.columns:
        nn = pd.to_numeric(df["nearest_neighbor_distance"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not nn.empty:
            radius = float(np.nanmedian(nn.to_numpy(dtype=float))) * 1.5
    if not np.isfinite(radius) or radius <= 0:
        return None
    return float(radius)


def select_visual_highlight_ids(
    results_df: pd.DataFrame,
    n_colonies: int | None = None,
    max_fraction: float | None = 0.20,
    max_count: int | None = 20,
    min_count: int = 0,
    statuses: tuple[str, ...] = VISUAL_HIGHLIGHT_STATUSES,
    max_per_neighborhood: int = 2,
    neighborhood_radius_factor: float = 2.5,
    min_distance_px: float | None = None,
    max_per_morphotype: int | None = 2,
    edge_band_diameters: float = 2.0,
    max_edge_fraction: float | None = 0.25,
    edge_score_penalty: float = 0.20,
    max_review_segmentation_fraction: float | None = 0.20,
    max_review_segmentation_count: int | None = 4,
) -> list[int]:
    """Выбрать ограниченный и пространственно разнообразный набор объектов для отрисовки.

    Лимит применяется к сумме всех визуальных классов: selected, review segmentation,
    unstable и прочие review-статусы. Если несколько кандидатов стоят рядом или
    относятся к одному морфотипу, selector оставляет только 1-2 представителя.
    """
    if results_df is None or results_df.empty or "colony_id" not in results_df.columns:
        return []

    total = int(n_colonies) if n_colonies is not None else int(len(results_df))
    limit = visual_highlight_limit(
        total,
        max_fraction=max_fraction,
        max_count=max_count,
        min_count=min_count,
    )
    if limit <= 0:
        return []

    if "recommendation_status" not in results_df.columns:
        if "selected_for_further_analysis" not in results_df.columns:
            return []
        df = results_df[results_df["selected_for_further_analysis"].fillna(False).astype(bool)].copy()
    else:
        df = results_df[results_df["recommendation_status"].isin(statuses)].copy()

    if df.empty:
        return []

    score_col = "final_anomaly_score_raw" if "final_anomaly_score_raw" in df.columns else "final_anomaly_score"
    if score_col not in df.columns:
        df["_visual_score"] = 0.0
        score_col = "_visual_score"
    else:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    if "recommendation_status" in df.columns:
        df["_visual_status_priority"] = (
            df["recommendation_status"].astype(str).map(VISUAL_HIGHLIGHT_STATUS_PRIORITY).fillna(99).astype(int)
        )
    else:
        df["_visual_status_priority"] = 0

    df["_visual_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).astype(float)
    df["_visual_is_edge_band"] = False

    edge_limit: int | None = None
    if (
        max_edge_fraction is not None
        and "distance_to_plate_edge" in df.columns
        and "equivalent_diameter" in results_df.columns
    ):
        diameters = pd.to_numeric(results_df["equivalent_diameter"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median_diameter = float(diameters.median()) if diameters.notna().any() else np.nan
        if np.isfinite(median_diameter) and median_diameter > 0:
            edge_threshold = max(0.0, float(edge_band_diameters)) * median_diameter
            if edge_threshold > 0:
                edge_fraction = min(max(float(max_edge_fraction), 0.0), 1.0)
                edge_limit = int(math.floor(limit * edge_fraction))
                if edge_fraction > 0.0 and limit > 0:
                    edge_limit = max(1, edge_limit)
                edge_dist = pd.to_numeric(df["distance_to_plate_edge"], errors="coerce")
                df["_visual_is_edge_band"] = edge_dist.lt(edge_threshold).fillna(False).astype(bool)
                edge_closeness = ((edge_threshold - edge_dist) / max(edge_threshold, 1e-8)).clip(lower=0.0, upper=1.0)
                df["_visual_score"] = df["_visual_score"] - float(edge_score_penalty) * edge_closeness.fillna(0.0)

    review_segmentation_limit: int | None = None
    if max_review_segmentation_fraction is not None or max_review_segmentation_count is not None:
        review_segmentation_limit = limit
        if max_review_segmentation_fraction is not None:
            status_fraction = min(max(float(max_review_segmentation_fraction), 0.0), 1.0)
            review_segmentation_limit = int(math.floor(limit * status_fraction))
            if status_fraction > 0.0 and limit > 0:
                review_segmentation_limit = max(1, review_segmentation_limit)
        if max_review_segmentation_count is not None:
            review_segmentation_limit = min(review_segmentation_limit, max(0, int(max_review_segmentation_count)))

    df = df.sort_values(
        ["_visual_status_priority", "_visual_score"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    radius = _visual_diversity_radius(
        results_df,
        radius_factor=neighborhood_radius_factor,
        min_distance_px=min_distance_px,
    )
    max_neighbors = max(1, int(max_per_neighborhood))
    morphotype_cap = None if max_per_morphotype is None else max(1, int(max_per_morphotype))

    selected_ids: list[int] = []
    selected_points: list[tuple[float, float]] = []
    morphotype_counts: dict[Any, int] = {}
    selected_set: set[int] = set()
    edge_count = 0
    review_segmentation_count = 0

    def can_add(
        row: pd.Series,
        enforce_morphotype: bool,
        enforce_neighborhood: bool,
    ) -> tuple[bool, int | None, tuple[float, float] | None, Any]:
        if len(selected_ids) >= limit:
            return False, None, None, None
        if pd.isna(row.get("colony_id")):
            return False, None, None, None
        colony_id = int(row["colony_id"])
        if colony_id in selected_set:
            return False, None, None, None

        status = str(row.get("recommendation_status", ""))
        if (
            review_segmentation_limit is not None
            and status == "review_segmentation"
            and review_segmentation_count >= review_segmentation_limit
        ):
            return False, None, None, None

        if (
            edge_limit is not None
            and bool(row.get("_visual_is_edge_band", False))
            and status != "select_candidate"
            and edge_count >= edge_limit
        ):
            return False, None, None, None

        if enforce_morphotype and morphotype_cap is not None and "morphotype_id" in row.index and pd.notna(row.get("morphotype_id")):
            morphotype_key = row.get("morphotype_id")
            current_count = int(morphotype_counts.get(morphotype_key, 0))
            if current_count >= morphotype_cap:
                return False, None, None, None
        else:
            morphotype_key = None

        point: tuple[float, float] | None = None
        if {"centroid_x", "centroid_y"}.issubset(df.columns):
            x = pd.to_numeric(pd.Series([row.get("centroid_x")]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([row.get("centroid_y")]), errors="coerce").iloc[0]
            if pd.notna(x) and pd.notna(y) and np.isfinite(float(x)) and np.isfinite(float(y)):
                point = (float(x), float(y))

        if enforce_neighborhood and point is not None and radius is not None and selected_points:
            nearby_count = sum(float(np.hypot(point[0] - sx, point[1] - sy)) <= radius for sx, sy in selected_points)
            if nearby_count >= max_neighbors:
                return False, None, None, None

        return True, colony_id, point, morphotype_key

    def add_row(row: pd.Series, colony_id: int, point: tuple[float, float] | None, morphotype_key: Any) -> None:
        nonlocal edge_count, review_segmentation_count
        selected_ids.append(colony_id)
        selected_set.add(colony_id)
        if point is not None:
            selected_points.append(point)
        if morphotype_key is not None:
            morphotype_counts[morphotype_key] = int(morphotype_counts.get(morphotype_key, 0)) + 1
        if bool(row.get("_visual_is_edge_band", False)):
            edge_count += 1
        if str(row.get("recommendation_status", "")) == "review_segmentation":
            review_segmentation_count += 1

    for _, row in df.iterrows():
        ok, colony_id, point, morphotype_key = can_add(row, enforce_morphotype=True, enforce_neighborhood=True)
        if ok and colony_id is not None:
            add_row(row, colony_id, point, morphotype_key)
        if len(selected_ids) >= limit:
            break

    if len(selected_ids) < limit:
        for _, row in df.iterrows():
            if len(selected_ids) >= limit:
                break
            ok, colony_id, point, morphotype_key = can_add(row, enforce_morphotype=False, enforce_neighborhood=True)
            if ok and colony_id is not None:
                add_row(row, colony_id, point, morphotype_key)

    return selected_ids[:limit]


def visualize_anomalies(
    image_rgb: np.ndarray,
    masks,
    results_df: pd.DataFrame,
    selected_ids: list[int] | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    show_scores: bool = True,
    show_review: bool = True,
    show_technical_warnings: bool = False,
    highlight_ids: list[int] | None = None,
):
    masks_list = normalize_masks_input(masks)
    if image_rgb is None:
        raise ValueError("image_rgb must not be None")

    selected_info: dict[int, dict[str, Any]] = {}
    technical_ids: set[int] = set()
    review_ids: set[int] = set()

    if results_df is not None and not results_df.empty and "colony_id" in results_df.columns:
        if "technical_warning" in results_df.columns:
            technical_ids = set(
                results_df.loc[
                    results_df["technical_warning"].fillna(False).astype(bool), "colony_id"
                ].dropna().astype(int).tolist()
            )
        if "recommendation_status" in results_df.columns:
            review_ids = set(
                results_df.loc[
                    results_df["recommendation_status"].isin(["review_only", "review_segmentation", "unstable_candidate", "rare_morphotype", "weak_outlier"]),
                    "colony_id",
                ].dropna().astype(int).tolist()
            )
        score_col = "final_anomaly_score_raw" if "final_anomaly_score_raw" in results_df.columns else "final_anomaly_score"
        if selected_ids is None:
            if "selected_for_further_analysis" in results_df.columns:
                selected_df = results_df[results_df["selected_for_further_analysis"].fillna(False).astype(bool)]
            else:
                selected_df = results_df.head(0)
            selected_ids = selected_df["colony_id"].dropna().astype(int).tolist()
        for _, row in results_df.iterrows():
            if pd.notna(row.get("colony_id")):
                cid = int(row["colony_id"])
                selected_info[cid] = {
                    "score": float(row.get(score_col)) if score_col in results_df.columns and pd.notna(row.get(score_col)) else np.nan,
                    "status": str(row.get("recommendation_status", "")),
                }
    elif selected_ids is None:
        selected_ids = []

    selected_set = {int(i) for i in selected_ids}
    if highlight_ids is not None:
        highlight_set = {int(i) for i in highlight_ids}
        selected_set = selected_set & highlight_set
        review_ids = review_ids & highlight_set

    smooth_sigma = 1.6
    contour_level = 0.35
    normal_fill_alpha = 0.08
    selected_fill_alpha = 0.06
    review_fill_alpha = 0.05

    def color_for_idx(i: int, n_total: int = 1) -> np.ndarray:
        if n_total < 1:
            n_total = 1
        t = ((i * 37) % n_total) / max(1, n_total - 1)
        hue = (0.02 + 0.96 * t) % 1.0
        sat = 0.85
        val = 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return np.array([r, g, b], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_rgb)
    ax.axis("off")
    if title:
        ax.set_title(title)

    n_total = len(masks_list)
    h, w = image_rgb.shape[:2]

    # Полная заливка инстансов остаётся полупрозрачной: морфология колоний не скрывается.
    fill_rgba = np.zeros((h, w, 4), dtype=float)
    draw_order = [i for i in range(n_total) if (i + 1) not in selected_set]
    draw_order += [i for i in range(n_total) if (i + 1) in selected_set]
    for k in draw_order:
        colony_id = k + 1
        is_technical = colony_id in technical_ids
        is_review = colony_id in review_ids
        is_selected = colony_id in selected_set
        if is_technical and not is_review and not is_selected and not show_technical_warnings:
            continue
        if is_review and not show_review and not is_selected:
            continue
        mask_bool = masks_list[k].astype(bool)
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if is_selected:
            fill_color = np.array([1.0, 0.95, 0.05])
            fill_alpha = selected_fill_alpha
        elif is_review:
            fill_color = np.array([1.0, 0.45, 0.0])
            fill_alpha = review_fill_alpha
        elif is_technical:
            fill_color = np.array([0.55, 0.55, 0.55])
            fill_alpha = 0.05
        else:
            fill_color = color_for_idx(k, n_total=n_total)
            fill_alpha = normal_fill_alpha
        fill_rgba[mask_bool, :3] = fill_color
        fill_rgba[mask_bool, 3] = fill_alpha
    ax.imshow(fill_rgba)

    for k, mask in enumerate(masks_list):
        colony_id = k + 1
        is_selected = colony_id in selected_set
        is_technical = colony_id in technical_ids
        is_review = colony_id in review_ids
        if is_technical and not is_review and not is_selected and not show_technical_warnings:
            continue
        if is_review and not show_review and not is_selected:
            continue

        if is_selected:
            color = "yellow"
            linewidth = 3.4
            linestyle = "-"
        elif is_review:
            color = "orange"
            linewidth = 2.2
            linestyle = "--"
        elif is_technical:
            color = "gray"
            linewidth = 1.0
            linestyle = ":"
        else:
            color = color_for_idx(k, n_total=n_total)
            linewidth = 0.7
            linestyle = "-"

        mask_prob = mask.astype(np.float32)
        mask_prob = cv2.GaussianBlur(mask_prob, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
        for contour in find_contours(mask_prob, level=contour_level):
            if is_selected:
                ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    color="black",
                    linewidth=linewidth + 1.6,
                    antialiased=True,
                    solid_joinstyle="round",
                    solid_capstyle="round",
                )
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                antialiased=True,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

        if is_selected or (show_review and is_review):
            ys, xs = np.where(mask)
            if len(xs) > 0:
                info = selected_info.get(colony_id, {})
                score = info.get("score", np.nan)
                status = info.get("status", "")
                if not show_scores or not np.isfinite(score):
                    continue
                if status == "review_segmentation":
                    label_text = "SEG"
                elif status == "unstable_candidate":
                    label_text = "UNST"
                elif status == "rare_morphotype":
                    label_text = "RARE"
                elif status == "review_only":
                    label_text = "REV"
                elif status == "weak_outlier":
                    label_text = "WEAK"
                else:
                    label_text = f"{score:.2f}"
                ax.text(
                    float(np.mean(xs)),
                    float(np.mean(ys)),
                    label_text,
                    color="black",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="yellow" if is_selected else "orange", edgecolor="black", alpha=0.82, pad=1.4),
                )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

def compute_plate_quality(scores_df: pd.DataFrame, detections_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Сводная оценка качества чашки/изображения для корректной интерпретации аномалий."""
    if scores_df is None:
        scores_df = pd.DataFrame()
    n_detected = int(len(scores_df))
    n_valid = int(scores_df["valid_for_anomaly"].fillna(False).astype(bool).sum()) if n_detected and "valid_for_anomaly" in scores_df.columns else 0
    n_technical = int(scores_df["technical_warning"].fillna(False).astype(bool).sum()) if n_detected and "technical_warning" in scores_df.columns else 0
    valid_fraction = float(n_valid / n_detected) if n_detected else 0.0
    technical_fraction = float(n_technical / n_detected) if n_detected else 0.0

    if "yolo_conf" in scores_df.columns and n_detected:
        median_yolo_conf = float(pd.to_numeric(scores_df["yolo_conf"], errors="coerce").median())
    elif detections_df is not None and not detections_df.empty and "yolo_conf" in detections_df.columns:
        median_yolo_conf = float(pd.to_numeric(detections_df["yolo_conf"], errors="coerce").median())
    else:
        median_yolo_conf = np.nan

    if "area" in scores_df.columns and n_detected:
        area_values = pd.to_numeric(scores_df["area"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        median_area = float(np.nanmedian(area_values)) if area_values.size else np.nan
        area_cv = float(np.nanstd(area_values) / (np.nanmean(area_values) + 1e-8)) if area_values.size else np.nan
    else:
        median_area = np.nan
        area_cv = np.nan

    if n_detected == 0:
        plate_quality_status = "no_masks"
    elif n_valid < 10:
        plate_quality_status = "low_colony_count"
    elif technical_fraction > 0.40:
        plate_quality_status = "too_many_technical_masks"
    elif np.isfinite(median_yolo_conf) and median_yolo_conf < 0.15:
        plate_quality_status = "low_detection_confidence"
    else:
        plate_quality_status = "ok"

    return pd.DataFrame([{
        "n_detected_colonies": n_detected,
        "n_valid_colonies": n_valid,
        "valid_fraction": valid_fraction,
        "n_technical_warnings": n_technical,
        "technical_fraction": technical_fraction,
        "median_yolo_conf": median_yolo_conf,
        "median_area": median_area,
        "area_cv": area_cv,
        "plate_quality_status": plate_quality_status,
    }])


def build_feature_correlation_report(features_df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Найти пары сильно коррелирующих признаков для контроля дублирования."""
    if features_df is None or features_df.empty:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    df = features_df.copy()
    if "valid_for_anomaly" in df.columns:
        df = df[df["valid_for_anomaly"].fillna(False).astype(bool)]
    if "technical_warning" in df.columns:
        df = df[~df["technical_warning"].fillna(False).astype(bool)]
    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric_df.shape[0] < 3 or numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    corr = numeric_df.corr(method="spearman", min_periods=3)
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": float(value)})
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])


def save_feature_space_pca(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Сохранить PCA-визуализацию признакового пространства валидных колоний."""
    if features_df is None or features_df.empty or scores_df is None or scores_df.empty:
        return
    df = scores_df.copy()
    if "valid_for_anomaly" in df.columns:
        df = df[df["valid_for_anomaly"].fillna(False).astype(bool)]
    if "technical_warning" in df.columns:
        df = df[~df["technical_warning"].fillna(False).astype(bool)]
    if len(df) < 3:
        return

    core_cols = [
        "log_area", "circularity", "eccentricity", "solidity",
        "intensity_mean", "intensity_iqr", "intensity_entropy",
        "mean_L_lab", "mean_a_lab", "mean_b_lab",
        "glcm_contrast", "lbp_entropy", "local_color_delta_lab",
        "intensity_hist_js_to_plate_median", "feature_knn_distance",
    ]
    cols = [c for c in core_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return
    try:
        X = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = RobustScaler().fit_transform(X)
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)

        fig, ax = plt.subplots(figsize=(7, 6))
        status = df["recommendation_status"].astype(str).to_numpy() if "recommendation_status" in df.columns else np.array(["object"] * len(df))
        unique_status = list(dict.fromkeys(status))
        for st in unique_status:
            mask = status == st
            ax.scatter(X2[mask, 0], X2[mask, 1], label=st, alpha=0.75, s=35)

        if "colony_id" in df.columns:
            label_mask = df["recommendation_status"].isin(["select_candidate", "review_segmentation", "unstable_candidate", "review_only"]) if "recommendation_status" in df.columns else pd.Series(False, index=df.index)
            for _, row in df[label_mask].head(20).iterrows():
                pos = df.index.get_loc(row.name)
                ax.text(X2[pos, 0], X2[pos, 1], str(int(row["colony_id"])), fontsize=8)

        ax.set_title("Пространство признаков колоний (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"Не удалось сохранить PCA-визуализацию: {exc}")


def save_colony_crops(
    image_rgb: np.ndarray,
    masks,
    scores_df: pd.DataFrame,
    output_dir: str | Path,
    statuses: tuple[str, ...] = ("select_candidate", "review_segmentation", "unstable_candidate", "review_only"),
    context_pad: int = 24,
) -> None:
    """Сохранить crop, context и mask для selected/review объектов."""
    if image_rgb is None or scores_df is None or scores_df.empty:
        return
    masks_list = normalize_masks_input(masks)
    if not masks_list:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_mask = {i + 1: m for i, m in enumerate(masks_list)}
    df = scores_df[scores_df.get("recommendation_status", "").isin(statuses)].copy() if "recommendation_status" in scores_df.columns else pd.DataFrame()
    if df.empty or "colony_id" not in df.columns:
        return

    h, w = image_rgb.shape[:2]
    for _, row in df.head(30).iterrows():
        try:
            colony_id = int(row["colony_id"])
            mask = id_to_mask.get(colony_id)
            if mask is None or not mask.any():
                continue
            ys, xs = np.where(mask)
            y1 = max(0, int(ys.min()) - context_pad)
            y2 = min(h, int(ys.max()) + context_pad + 1)
            x1 = max(0, int(xs.min()) - context_pad)
            x2 = min(w, int(xs.max()) + context_pad + 1)

            context = image_rgb[y1:y2, x1:x2].copy()
            local_mask = mask[y1:y2, x1:x2]
            crop = image_rgb[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
            mask_img = (local_mask.astype(np.uint8) * 255)

            prefix = output_dir / f"colony_id_{colony_id:03d}"
            cv2.imwrite(str(prefix) + "_context.png", cv2.cvtColor(context, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(prefix) + "_crop.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(prefix) + "_mask.png", mask_img)
        except Exception:
            continue

def run_single_image_pipeline(
    image_path: str | Path,
    model,
    config: AnomalyDetectionConfig,
    petri_detector_model=None,
) -> dict:
    image_path = Path(image_path)
    image_rgb_original = read_image_rgb(image_path)

    # 1) Предобработка изображения перед сегментацией.
    image_rgb, preprocess_info = prepare_image_for_segmentation(
        image_rgb=image_rgb_original,
        detector_model=petri_detector_model if config.use_petri_detector else None,
        config=config,
    )
    preprocess_info = dict(preprocess_info)
    apply_mask_smoothing = bool(getattr(config, "smooth_masks_before_anomaly", True))
    mask_smoothing_sigma = float(getattr(config, "mask_smoothing_sigma", 1.6))
    mask_smoothing_threshold = float(getattr(config, "mask_smoothing_threshold", 0.50))
    preprocess_info["mask_smoothing_before_anomaly"] = apply_mask_smoothing
    preprocess_info["mask_smoothing_sigma"] = mask_smoothing_sigma
    preprocess_info["mask_smoothing_threshold"] = mask_smoothing_threshold

    output_dir = Path(config.output_dir) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "preprocess_info.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_info, f, ensure_ascii=False, indent=2)

    if config.save_visualizations:
        pre_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "preprocessed_736.png"), pre_bgr)

    # 2) YOLO instance segmentation: сначала получаем маски, только затем считаем признаки.
    masks_yolo, detections_df = predict_colony_masks(model, image_rgb, config)
    if apply_mask_smoothing:
        masks_for_anomaly, smoothing_df = smooth_masks_before_anomaly(
            masks_yolo,
            config=config,
            sigma=mask_smoothing_sigma,
            threshold=mask_smoothing_threshold,
        )
    else:
        masks_for_anomaly = normalize_masks_input(masks_yolo)
        smoothing_df = build_mask_smoothing_diagnostics(
            masks_for_anomaly,
            sigma=mask_smoothing_sigma,
            threshold=mask_smoothing_threshold,
            applied=False,
        )

    detections_for_features = detections_df.copy()
    if (
        not detections_for_features.empty
        and not smoothing_df.empty
        and "colony_id" in detections_for_features.columns
    ):
        detections_for_features = detections_for_features.merge(smoothing_df, on="colony_id", how="left")

    # 3) Признаки, морфотипы, weighted anomaly score и статус рекомендации.
    features_df = extract_colony_features(
        image_rgb=image_rgb,
        masks=masks_for_anomaly,
        detections_df=detections_for_features,
        plate_mask=None,
        config=config,
    )
    scores_df = compute_anomaly_scores(features_df, config=config)
    stability_df = _empty_perturbation_stability_df()

    # 4) Оценка качества чашки. Если качество низкое, автоматический select_candidate
    # переводится в review_only, чтобы не выдавать слабое статистическое решение как подтверждённый отбор.
    plate_quality_df = compute_plate_quality(scores_df, detections_for_features)
    plate_quality_status = (
        str(plate_quality_df.loc[0, "plate_quality_status"])
        if not plate_quality_df.empty and "plate_quality_status" in plate_quality_df.columns
        else "ok"
    )
    if plate_quality_status != "ok" and not scores_df.empty and "recommendation_status" in scores_df.columns:
        bad_auto = scores_df["recommendation_status"].eq("select_candidate")
        scores_df.loc[bad_auto, "recommendation_status"] = "review_only"
        scores_df.loc[bad_auto, "selected_for_further_analysis"] = False
        if "explanations_text" in scores_df.columns:
            scores_df.loc[bad_auto, "explanations_text"] = (
                scores_df.loc[bad_auto, "explanations_text"].astype(str)
                + f"; качество чашки: {plate_quality_status}, автоматический отбор заменён на экспертный просмотр"
            )

    if (
        bool(getattr(config, "use_perturbation_stability", False))
        and plate_quality_status == "ok"
        and not scores_df.empty
        and "recommendation_status" in scores_df.columns
        and "colony_id" in scores_df.columns
    ):
        stability_candidate_ids = (
            scores_df.loc[scores_df["recommendation_status"].eq("select_candidate"), "colony_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        if stability_candidate_ids:
            stability_df = compute_perturbation_stability_for_candidates(
                image_rgb=image_rgb,
                model=model,
                config=config,
                baseline_scores=scores_df,
                candidate_ids=stability_candidate_ids,
            )
            scores_df = apply_perturbation_stability_filter(scores_df, stability_df, config)

    if not scores_df.empty:
        scores_df["explanations"] = scores_df.apply(
            lambda row: explain_anomaly(row, scores_df, top_k=5), axis=1
        )
        if "perturbation_stability_pass" in scores_df.columns:
            unstable_mask = scores_df["recommendation_status"].eq("unstable_candidate") if "recommendation_status" in scores_df.columns else pd.Series(False, index=scores_df.index)
            for idx in scores_df[unstable_mask].index:
                score = pd.to_numeric(pd.Series([scores_df.at[idx, "perturbation_stability_score"]]), errors="coerce").iloc[0]
                hits = pd.to_numeric(pd.Series([scores_df.at[idx, "perturbation_stability_hits"]]), errors="coerce").iloc[0]
                trials = pd.to_numeric(pd.Series([scores_df.at[idx, "perturbation_stability_trials"]]), errors="coerce").iloc[0]
                score_text = f"{float(score):.3f}" if pd.notna(score) and np.isfinite(float(score)) else "nan"
                note = f"candidate failed perturbation stability check: {int(hits) if pd.notna(hits) else 0}/{int(trials) if pd.notna(trials) else 0}, stability_score={score_text}"
                current = scores_df.at[idx, "explanations"]
                if isinstance(current, list):
                    current.append(note)
                    scores_df.at[idx, "explanations"] = current
                else:
                    scores_df.at[idx, "explanations"] = [note]
        if plate_quality_status != "ok" and "recommendation_status" in scores_df.columns:
            quality_mask = scores_df["recommendation_status"].eq("review_only")
            for idx in scores_df[quality_mask].index:
                note = f"plate quality is {plate_quality_status}; automatic selection moved to review"
                current = scores_df.at[idx, "explanations"]
                if isinstance(current, list):
                    current.append(note)
                    scores_df.at[idx, "explanations"] = current
                else:
                    scores_df.at[idx, "explanations"] = [note]
        scores_df["explanations_text"] = scores_df["explanations"].apply(lambda x: "; ".join(x))
    else:
        scores_df = scores_df.copy()
        scores_df["explanations"] = pd.Series(dtype=object)
        scores_df["explanations_text"] = pd.Series(dtype=object)

    selected_df = select_top_anomalies(
        scores_df,
        top_k=config.top_k,
        top_percent=config.top_percent,
        min_score=config.min_score,
    )

    if "selected_for_further_analysis" not in scores_df.columns:
        scores_df["selected_for_further_analysis"] = False
    scores_df["selected_for_further_analysis"] = False

    selected_ids = (
        selected_df["colony_id"].dropna().astype(int).tolist()
        if "colony_id" in selected_df.columns and not selected_df.empty
        else []
    )
    if selected_ids and "colony_id" in scores_df.columns:
        scores_df.loc[
            scores_df["colony_id"].astype(int).isin(selected_ids),
            "selected_for_further_analysis",
        ] = True
        selected_df = scores_df[scores_df["selected_for_further_analysis"].fillna(False).astype(bool)].copy()

    visual_highlight_ids = select_visual_highlight_ids(
        scores_df,
        n_colonies=len(masks_for_anomaly),
        max_fraction=getattr(config, "visual_highlight_percent", 0.20),
        max_count=getattr(config, "visual_highlight_max_count", 20),
        min_count=getattr(config, "visual_highlight_min_count", 0),
        max_per_neighborhood=getattr(config, "visual_highlight_max_per_neighborhood", 2),
        neighborhood_radius_factor=getattr(config, "visual_highlight_neighborhood_radius_factor", 2.5),
        min_distance_px=getattr(config, "visual_highlight_min_distance_px", None),
        max_per_morphotype=getattr(config, "visual_highlight_max_per_morphotype", 2),
        edge_band_diameters=getattr(config, "visual_highlight_edge_band_diameters", 2.0),
        max_edge_fraction=getattr(config, "visual_highlight_max_edge_fraction", 0.25),
        edge_score_penalty=getattr(config, "visual_highlight_edge_score_penalty", 0.20),
        max_review_segmentation_fraction=getattr(config, "visual_highlight_max_review_segmentation_fraction", 0.20),
        max_review_segmentation_count=getattr(config, "visual_highlight_max_review_segmentation_count", 4),
    )
    visual_highlights_df = scores_df.head(0).copy()
    if visual_highlight_ids and "colony_id" in scores_df.columns:
        order = {int(cid): i for i, cid in enumerate(visual_highlight_ids)}
        order_ids = set(order)
        colony_ids_numeric = pd.to_numeric(scores_df["colony_id"], errors="coerce")
        visual_highlights_df = scores_df[
            colony_ids_numeric.notna() & colony_ids_numeric.astype("Int64").isin(order_ids)
        ].copy()
        visual_highlights_df["visual_highlight_rank"] = (
            pd.to_numeric(visual_highlights_df["colony_id"], errors="coerce").astype(int).map(order).astype(int) + 1
        )
        visual_highlights_df = visual_highlights_df.sort_values("visual_highlight_rank").reset_index(drop=True)

    review_candidates_df = collect_review_candidates(scores_df)
    technical_warnings_df = (
        scores_df[scores_df["technical_warning"].fillna(False).astype(bool)].copy()
        if not scores_df.empty and "technical_warning" in scores_df.columns
        else pd.DataFrame()
    )
    valid_features_df = (
        features_df[features_df["valid_for_anomaly"].fillna(False).astype(bool)].copy()
        if not features_df.empty and "valid_for_anomaly" in features_df.columns
        else pd.DataFrame()
    )

    morphotype_cols = [
        "colony_id",
        "morphotype_id",
        "morphotype_size",
        "morphotype_fraction",
        "rare_morphotype_flag",
        "within_morphotype_anomaly_score",
        "size_within_morphotype_z",
        "texture_within_morphotype_z",
        "color_within_morphotype_z",
        "recommendation_status",
    ]
    morphotypes_df = scores_df[[c for c in morphotype_cols if c in scores_df.columns]].copy() if not scores_df.empty else pd.DataFrame()

    feature_correlation_df = (
        build_feature_correlation_report(scores_df)
        if bool(getattr(config, "save_feature_correlation_report", True))
        else pd.DataFrame()
    )

    fig_save_path = output_dir / "anomalies_visualization.png" if config.save_visualizations else None
    fig = visualize_anomalies(
        image_rgb=image_rgb,
        masks=masks_for_anomaly,
        results_df=scores_df,
        selected_ids=selected_ids,
        save_path=fig_save_path,
        title=f"Smoothed-mask anomaly candidates: {image_path.name}" if apply_mask_smoothing else f"Anomaly candidates: {image_path.name}",
        show_scores=True,
        show_review=True,
        show_technical_warnings=False,
        highlight_ids=visual_highlight_ids,
    )

    if bool(getattr(config, "save_feature_space_plot", True)):
        save_feature_space_pca(scores_df, scores_df, output_dir / "feature_space_pca.png")

    if bool(getattr(config, "save_colony_crops", True)):
        save_colony_crops(
            image_rgb=image_rgb,
            masks=masks_for_anomaly,
            scores_df=scores_df,
            output_dir=output_dir / "colony_crops",
        )

    masks_labeled = np.zeros(image_rgb.shape[:2], dtype=np.int32)
    for colony_id, mask in enumerate(normalize_masks_input(masks_for_anomaly), start=1):
        if mask.shape != masks_labeled.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (masks_labeled.shape[1], masks_labeled.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        masks_labeled[mask] = int(colony_id)
    np.save(output_dir / "masks_labeled.npy", masks_labeled)

    if config.save_csv:
        detections_for_features.to_csv(output_dir / "detections_yolo.csv", index=False, encoding="utf-8-sig")
        smoothing_df.to_csv(output_dir / "mask_smoothing_diagnostics.csv", index=False, encoding="utf-8-sig")
        stability_df.to_csv(output_dir / "perturbation_stability.csv", index=False, encoding="utf-8-sig")
        features_df.to_csv(output_dir / "colony_features.csv", index=False, encoding="utf-8-sig")
        valid_features_df.to_csv(output_dir / "colony_features_valid.csv", index=False, encoding="utf-8-sig")
        scores_df.to_csv(output_dir / "colony_anomaly_scores.csv", index=False, encoding="utf-8-sig")
        selected_df.to_csv(output_dir / "selected_anomalies.csv", index=False, encoding="utf-8-sig")
        visual_highlights_df.to_csv(output_dir / "visual_highlighted_objects.csv", index=False, encoding="utf-8-sig")
        review_candidates_df.to_csv(output_dir / "review_candidates.csv", index=False, encoding="utf-8-sig")
        technical_warnings_df.to_csv(output_dir / "technical_warnings.csv", index=False, encoding="utf-8-sig")
        morphotypes_df.to_csv(output_dir / "morphotypes.csv", index=False, encoding="utf-8-sig")

        plate_quality_df.to_csv(output_dir / "plate_quality.csv", index=False, encoding="utf-8-sig")
        feature_correlation_df.to_csv(output_dir / "feature_correlation_report.csv", index=False, encoding="utf-8-sig")

        if config.save_xlsx:
            try:
                with pd.ExcelWriter(output_dir / "colony_analysis_report.xlsx") as writer:
                    detections_for_features.to_excel(writer, sheet_name="detections_yolo", index=False)
                    smoothing_df.to_excel(writer, sheet_name="mask_smoothing", index=False)
                    stability_df.to_excel(writer, sheet_name="stability", index=False)
                    features_df.to_excel(writer, sheet_name="colony_features_all", index=False)
                    valid_features_df.to_excel(writer, sheet_name="colony_features_valid", index=False)
                    scores_df.to_excel(writer, sheet_name="anomaly_scores", index=False)
                    selected_df.to_excel(writer, sheet_name="selected_anomalies", index=False)
                    visual_highlights_df.to_excel(writer, sheet_name="visual_highlights", index=False)
                    review_candidates_df.to_excel(writer, sheet_name="review_candidates", index=False)
                    technical_warnings_df.to_excel(writer, sheet_name="technical_warnings", index=False)
                    morphotypes_df.to_excel(writer, sheet_name="morphotypes", index=False)
                    plate_quality_df.to_excel(writer, sheet_name="plate_quality", index=False)
                    feature_correlation_df.to_excel(writer, sheet_name="feature_correlation", index=False)
            except Exception as exc:
                print(f"Не удалось сохранить Excel-отчёт: {exc}")

    return {
        "image_path": image_path,
        "image_original": image_rgb_original,
        "image": image_rgb,
        "preprocess": preprocess_info,
        "masks_raw_yolo": masks_yolo,
        "masks": masks_for_anomaly,
        "mask_smoothing": smoothing_df,
        "perturbation_stability": stability_df,
        "detections": detections_for_features,
        "features": features_df,
        "scores": scores_df,
        "selected": selected_df,
        "visual_highlight_ids": visual_highlight_ids,
        "visual_highlights": visual_highlights_df,
        "review_candidates": review_candidates_df,
        "technical_warnings": technical_warnings_df,
        "morphotypes": morphotypes_df,
        "plate_quality": plate_quality_df,
        "feature_correlation": feature_correlation_df,
        "figure": fig,
    }

def run_batch_pipeline(config: AnomalyDetectionConfig) -> pd.DataFrame:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(config.input_path)

    if config.model_weights_path:
        seg_weights_path = Path(config.model_weights_path)
        if not seg_weights_path.exists():
            raise FileNotFoundError(f"Segmentation weights file not found: {seg_weights_path}")
    else:
        seg_weights_path = find_best_yolo_weights(config.model_dir)

    print("Используем веса модели сегментации для batch:", seg_weights_path)
    model = YOLO(str(seg_weights_path))

    petri_detector_model = None
    if config.use_petri_detector or str(config.preprocess_mode).lower() == "detect_petri":
        petri_weights_path = Path(config.petri_detector_weights_path)
        if not petri_weights_path.exists():
            raise FileNotFoundError(f"Petri detector weights file not found: {petri_weights_path}")
        print("Используем веса детектора чашки Петри для batch:", petri_weights_path)
        petri_detector_model = YOLO(str(petri_weights_path))

    selected_rows: list[pd.DataFrame] = []
    visual_rows: list[pd.DataFrame] = []
    review_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    selected_required_cols = [
        "image_name",
        "colony_id",
        "final_anomaly_score_raw",
        "final_anomaly_score_percentile",
        "final_anomaly_score",
        "anomaly_rank",
        "recommendation_status",
        "visual_highlight_rank",
        "anomaly_type",
        "independent_evidence_count",
        "consensus_score",
        "overall_reliability_score",
        "perturbation_stability_score",
        "perturbation_stability_hits",
        "perturbation_stability_trials",
        "perturbation_stability_pass",
        "explanations_text",
        "yolo_conf",
        "area",
        "log_area",
        "area_to_median_ratio",
        "circularity",
        "solidity",
        "intensity_mean",
        "intensity_iqr",
        "glcm_contrast",
        "lbp_entropy",
        "histogram_score",
        "local_color_delta_lab",
        "nearest_neighbor_distance",
        "feature_knn_distance",
    ]

    for image_path in tqdm(image_paths, desc="Batch processing"):
        try:
            result = run_single_image_pipeline(
                image_path=image_path,
                model=model,
                config=config,
                petri_detector_model=petri_detector_model,
            )
            scores_df = result["scores"]
            selected_df = result["selected"]
            visual_df = result.get("visual_highlights", pd.DataFrame())
            review_df = result.get("review_candidates", pd.DataFrame())
            plate_quality_df = result.get("plate_quality", pd.DataFrame())

            n_detected = int(len(scores_df))
            n_valid = int(scores_df["valid_for_anomaly"].fillna(False).astype(bool).sum()) if n_detected and "valid_for_anomaly" in scores_df.columns else 0
            n_technical = int(scores_df["technical_warning"].fillna(False).astype(bool).sum()) if n_detected and "technical_warning" in scores_df.columns else 0
            valid_fraction = float(n_valid / n_detected) if n_detected else 0.0
            technical_fraction = float(n_technical / n_detected) if n_detected else 0.0
            plate_quality_status = (
                str(plate_quality_df.loc[0, "plate_quality_status"])
                if plate_quality_df is not None and not plate_quality_df.empty and "plate_quality_status" in plate_quality_df.columns
                else ("no_masks" if n_detected == 0 else ("no_valid_colonies" if n_valid == 0 else "ok"))
            )
            n_selected = int(len(selected_df))
            n_visual = int(len(visual_df)) if isinstance(visual_df, pd.DataFrame) else 0
            n_review = int(len(review_df))
            n_rare = int(scores_df["rare_morphotype_flag"].fillna(False).astype(bool).sum()) if n_detected and "rare_morphotype_flag" in scores_df.columns else 0

            valid_scores = (
                scores_df[scores_df["valid_for_anomaly"].fillna(False).astype(bool)].copy()
                if n_detected and "valid_for_anomaly" in scores_df.columns
                else pd.DataFrame()
            )
            max_raw = float(valid_scores["final_anomaly_score_raw"].max()) if not valid_scores.empty and "final_anomaly_score_raw" in valid_scores.columns else np.nan
            mean_raw = float(valid_scores["final_anomaly_score_raw"].mean()) if not valid_scores.empty and "final_anomaly_score_raw" in valid_scores.columns else np.nan
            max_percentile = float(valid_scores["final_anomaly_score_percentile"].max()) if not valid_scores.empty and "final_anomaly_score_percentile" in valid_scores.columns else np.nan

            if n_detected == 0:
                status = "no_masks"
            elif n_valid == 0:
                status = "no_valid_colonies"
            elif plate_quality_status != "ok":
                status = "low_quality"
            else:
                status = "ok"

            type_source = selected_df if not selected_df.empty else review_df
            if not type_source.empty and "anomaly_type" in type_source.columns:
                main_types = ", ".join(type_source["anomaly_type"].dropna().astype(str).value_counts().head(3).index.tolist())
            else:
                main_types = ""

            summary_rows.append(
                {
                    "image_name": Path(image_path).name,
                    "n_detected_colonies": n_detected,
                    "n_valid_colonies": n_valid,
                    "valid_fraction": valid_fraction,
                    "n_technical_warnings": n_technical,
                    "technical_fraction": technical_fraction,
                    "n_selected_anomalies": n_selected,
                    "n_visual_highlights": n_visual,
                    "n_review_candidates": n_review,
                    "n_rare_morphotypes": n_rare,
                    "max_anomaly_score_raw": max_raw,
                    "mean_anomaly_score_raw": mean_raw,
                    "max_anomaly_score_percentile": max_percentile,
                    "main_anomaly_types": main_types,
                    "plate_quality_status": plate_quality_status,
                    "status": status,
                }
            )

            if not selected_df.empty:
                temp = selected_df.copy()
                temp.insert(0, "image_name", Path(image_path).name)
                for col in selected_required_cols:
                    if col not in temp.columns:
                        temp[col] = np.nan
                selected_rows.append(temp[selected_required_cols])

            if isinstance(visual_df, pd.DataFrame) and not visual_df.empty:
                temp = visual_df.copy()
                temp.insert(0, "image_name", Path(image_path).name)
                for col in selected_required_cols:
                    if col not in temp.columns:
                        temp[col] = np.nan
                visual_rows.append(temp[selected_required_cols])

            if not review_df.empty:
                temp = review_df.copy()
                temp.insert(0, "image_name", Path(image_path).name)
                for col in selected_required_cols:
                    if col not in temp.columns:
                        temp[col] = np.nan
                review_rows.append(temp[selected_required_cols])

        except Exception as exc:
            print(f"Ошибка при обработке {image_path}: {exc}")
            summary_rows.append(
                {
                    "image_name": Path(image_path).name,
                    "n_detected_colonies": 0,
                    "n_valid_colonies": 0,
                    "valid_fraction": 0.0,
                    "n_technical_warnings": 0,
                    "technical_fraction": 0.0,
                    "n_selected_anomalies": 0,
                    "n_visual_highlights": 0,
                    "n_review_candidates": 0,
                    "n_rare_morphotypes": 0,
                    "max_anomaly_score_raw": np.nan,
                    "mean_anomaly_score_raw": np.nan,
                    "max_anomaly_score_percentile": np.nan,
                    "main_anomaly_types": "",
                    "plate_quality_status": "error",
                    "status": "error",
                }
            )

    all_selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else pd.DataFrame(columns=selected_required_cols)
    )
    all_review = (
        pd.concat(review_rows, ignore_index=True)
        if review_rows
        else pd.DataFrame(columns=selected_required_cols)
    )
    all_visual = (
        pd.concat(visual_rows, ignore_index=True)
        if visual_rows
        else pd.DataFrame(columns=selected_required_cols)
    )
    if not all_selected.empty:
        all_selected = all_selected.sort_values("final_anomaly_score_raw", ascending=False, na_position="last").reset_index(drop=True)
    if not all_visual.empty:
        all_visual = all_visual.sort_values(["image_name", "visual_highlight_rank"], ascending=[True, True], na_position="last").reset_index(drop=True)
    if not all_review.empty:
        all_review = all_review.sort_values("final_anomaly_score_raw", ascending=False, na_position="last").reset_index(drop=True)

    batch_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "image_name",
            "n_detected_colonies",
            "n_valid_colonies",
            "valid_fraction",
            "n_technical_warnings",
            "technical_fraction",
            "n_selected_anomalies",
            "n_visual_highlights",
            "n_review_candidates",
            "n_rare_morphotypes",
            "max_anomaly_score_raw",
            "mean_anomaly_score_raw",
            "max_anomaly_score_percentile",
            "main_anomaly_types",
            "plate_quality_status",
            "status",
        ],
    )

    all_selected.to_csv(output_dir / "all_selected_anomalies.csv", index=False, encoding="utf-8-sig")
    all_visual.to_csv(output_dir / "all_visual_highlighted_objects.csv", index=False, encoding="utf-8-sig")
    all_review.to_csv(output_dir / "all_review_candidates.csv", index=False, encoding="utf-8-sig")
    batch_summary.to_csv(output_dir / "batch_summary.csv", index=False, encoding="utf-8-sig")

    if config.save_xlsx:
        try:
            with pd.ExcelWriter(output_dir / "batch_colony_analysis_report.xlsx") as writer:
                all_selected.to_excel(writer, sheet_name="selected_anomalies", index=False)
                all_visual.to_excel(writer, sheet_name="visual_highlights", index=False)
                all_review.to_excel(writer, sheet_name="review_candidates", index=False)
                batch_summary.to_excel(writer, sheet_name="batch_summary", index=False)
        except Exception as exc:
            print(f"Не удалось сохранить batch Excel-отчёт: {exc}")

    return all_selected

def _apply_stability_perturbation(image_rgb: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    img = image_rgb.astype(np.float32)
    if variant == "brightness_plus_5pct":
        img = img * 1.05
    elif variant == "brightness_minus_5pct":
        img = img * 0.95
    elif variant == "contrast_plus_5pct":
        img = (img - 127.5) * 1.05 + 127.5
    elif variant == "contrast_minus_5pct":
        img = (img - 127.5) * 0.95 + 127.5
    elif variant == "slight_blur":
        img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), sigmaX=0.6).astype(np.float32)
    elif variant == "slight_gaussian_noise":
        img = img + rng.normal(0.0, 3.0, size=img.shape)
    else:
        raise ValueError(f"неизвестный вариант perturbation: {variant}")
    return np.clip(img, 0, 255).astype(np.uint8)


def _selected_centroid_table(scores_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if scores_df is None or scores_df.empty:
        return pd.DataFrame()
    df = scores_df.copy()
    if "valid_for_anomaly" in df.columns:
        df = df[df["valid_for_anomaly"].fillna(False).astype(bool)]
    if "technical_warning" in df.columns:
        df = df[~df["technical_warning"].fillna(False).astype(bool)]
    required = {"colony_id", "centroid_x", "centroid_y", "final_anomaly_score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df.sort_values("final_anomaly_score", ascending=False, na_position="last").head(top_k).copy()


def _match_selected_to_baseline(
    baseline_scores: pd.DataFrame,
    perturbed_scores: pd.DataFrame,
    top_k: int,
    max_match_distance: float = 20.0,
) -> tuple[set[int], int, float]:
    base_top = _selected_centroid_table(baseline_scores, top_k=top_k)
    pert_top = _selected_centroid_table(perturbed_scores, top_k=top_k)
    if base_top.empty or pert_top.empty:
        return set(), 0, np.nan

    base_coords = base_top[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    base_ids = base_top["colony_id"].astype(int).to_numpy()
    tree = cKDTree(base_coords)

    mapped_ids: set[int] = set()
    for _, row in pert_top.iterrows():
        point = np.array([float(row["centroid_x"]), float(row["centroid_y"])])
        if not np.all(np.isfinite(point)):
            continue
        dist, idx = tree.query(point, k=1)
        if np.isfinite(dist) and dist <= max_match_distance:
            mapped_ids.add(int(base_ids[int(idx)]))

    base_set = set(base_ids.tolist())
    n_common = len(base_set & mapped_ids)
    union = base_set | mapped_ids
    jaccard = float(n_common / len(union)) if union else np.nan
    return mapped_ids, n_common, jaccard


def _spearman_by_centroid_matching(
    baseline_scores: pd.DataFrame,
    perturbed_scores: pd.DataFrame,
    max_match_distance: float = 20.0,
) -> tuple[float, int]:
    if baseline_scores is None or baseline_scores.empty or perturbed_scores is None or perturbed_scores.empty:
        return np.nan, 0
    needed = {"centroid_x", "centroid_y", "final_anomaly_score"}
    if not needed.issubset(baseline_scores.columns) or not needed.issubset(perturbed_scores.columns):
        return np.nan, 0

    base_df = baseline_scores.copy()
    pert_df = perturbed_scores.copy()
    if "valid_for_anomaly" in base_df.columns:
        base_df = base_df[base_df["valid_for_anomaly"].fillna(False).astype(bool)]
    if "valid_for_anomaly" in pert_df.columns:
        pert_df = pert_df[pert_df["valid_for_anomaly"].fillna(False).astype(bool)]
    if base_df.empty or pert_df.empty:
        return np.nan, 0

    base_coords = base_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tree = cKDTree(base_coords)
    base_scores = base_df["final_anomaly_score"].to_numpy(dtype=float)

    matched_base_scores: list[float] = []
    matched_pert_scores: list[float] = []
    for _, row in pert_df.iterrows():
        point = np.array([float(row["centroid_x"]), float(row["centroid_y"])])
        if not np.all(np.isfinite(point)):
            continue
        dist, idx = tree.query(point, k=1)
        if np.isfinite(dist) and dist <= max_match_distance:
            base_score = float(base_scores[int(idx)])
            pert_score = float(row["final_anomaly_score"])
            if np.isfinite(base_score) and np.isfinite(pert_score):
                matched_base_scores.append(base_score)
                matched_pert_scores.append(pert_score)

    if len(matched_base_scores) < 3:
        return np.nan, len(matched_base_scores)
    try:
        corr = stats.spearmanr(matched_base_scores, matched_pert_scores, nan_policy="omit").correlation
        return float(corr), len(matched_base_scores)
    except Exception:
        return np.nan, len(matched_base_scores)


PERTURBATION_STABILITY_COLUMNS = [
    "colony_id",
    "perturbation_stability_score",
    "perturbation_stability_hits",
    "perturbation_stability_trials",
    "perturbation_stability_pass",
    "perturbation_stability_variants",
    "perturbation_stability_selected_variants",
    "perturbation_stability_status",
]


def _empty_perturbation_stability_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PERTURBATION_STABILITY_COLUMNS)


def compute_perturbation_stability_for_candidates(
    image_rgb: np.ndarray,
    model,
    config: AnomalyDetectionConfig,
    baseline_scores: pd.DataFrame,
    candidate_ids: list[int],
) -> pd.DataFrame:
    if image_rgb is None or baseline_scores is None or baseline_scores.empty or not candidate_ids:
        return _empty_perturbation_stability_df()
    required = {"colony_id", "centroid_x", "centroid_y"}
    if not required.issubset(baseline_scores.columns):
        return _empty_perturbation_stability_df()

    candidates = baseline_scores[baseline_scores["colony_id"].astype(int).isin([int(c) for c in candidate_ids])].copy()
    candidates = candidates.dropna(subset=["centroid_x", "centroid_y"])
    if candidates.empty:
        return _empty_perturbation_stability_df()

    candidate_ids = candidates["colony_id"].astype(int).tolist()
    hits = {int(cid): 0 for cid in candidate_ids}
    selected_variants = {int(cid): [] for cid in candidate_ids}
    trials = 0
    variant_status: list[str] = []

    variants = tuple(getattr(config, "perturbation_stability_variants", ())) or (
        "brightness_plus_5pct",
        "contrast_minus_5pct",
        "slight_blur",
    )
    rng = np.random.default_rng(int(getattr(config, "random_state", 42)))
    match_distance = float(getattr(config, "perturbation_stability_match_distance", 20.0))

    stability_config = AnomalyDetectionConfig(**config.__dict__)
    stability_config.save_visualizations = False
    stability_config.save_csv = False
    stability_config.save_xlsx = False
    stability_config.save_colony_crops = False
    stability_config.save_feature_space_plot = False
    stability_config.save_feature_correlation_report = False
    stability_config.use_perturbation_stability = False

    candidate_coords = candidates[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    for variant in variants:
        try:
            perturbed_rgb = _apply_stability_perturbation(image_rgb, str(variant), rng)
            masks_yolo, detections_df = predict_colony_masks(model, perturbed_rgb, stability_config)
            if bool(getattr(stability_config, "smooth_masks_before_anomaly", True)):
                masks_for_anomaly, smoothing_df = smooth_masks_before_anomaly(
                    masks_yolo,
                    config=stability_config,
                    sigma=float(getattr(stability_config, "mask_smoothing_sigma", 1.6)),
                    threshold=float(getattr(stability_config, "mask_smoothing_threshold", 0.50)),
                )
            else:
                masks_for_anomaly = normalize_masks_input(masks_yolo)
                smoothing_df = build_mask_smoothing_diagnostics(
                    masks_for_anomaly,
                    sigma=float(getattr(stability_config, "mask_smoothing_sigma", 1.6)),
                    threshold=float(getattr(stability_config, "mask_smoothing_threshold", 0.50)),
                    applied=False,
                )

            detections_for_features = detections_df.copy()
            if (
                not detections_for_features.empty
                and not smoothing_df.empty
                and "colony_id" in detections_for_features.columns
            ):
                detections_for_features = detections_for_features.merge(smoothing_df, on="colony_id", how="left")

            features_df = extract_colony_features(
                image_rgb=perturbed_rgb,
                masks=masks_for_anomaly,
                detections_df=detections_for_features,
                plate_mask=None,
                config=stability_config,
            )
            pert_scores = compute_anomaly_scores(features_df, config=stability_config)
            pert_selected = select_top_anomalies(
                pert_scores,
                top_k=stability_config.top_k,
                top_percent=stability_config.top_percent,
                min_score=stability_config.min_score,
            )
            selected_ids = (
                set(pert_selected["colony_id"].dropna().astype(int).tolist())
                if not pert_selected.empty and "colony_id" in pert_selected.columns
                else set()
            )

            valid_pert = pert_scores.copy()
            if "valid_for_anomaly" in valid_pert.columns:
                valid_pert = valid_pert[valid_pert["valid_for_anomaly"].fillna(False).astype(bool)]
            if "technical_warning" in valid_pert.columns:
                valid_pert = valid_pert[~valid_pert["technical_warning"].fillna(False).astype(bool)]
            if valid_pert.empty or not {"colony_id", "centroid_x", "centroid_y"}.issubset(valid_pert.columns):
                trials += 1
                variant_status.append(f"{variant}:no_valid_matches")
                continue

            tree = cKDTree(valid_pert[["centroid_x", "centroid_y"]].to_numpy(dtype=float))
            pert_ids = valid_pert["colony_id"].astype(int).to_numpy()
            trials += 1
            for cid, point in zip(candidate_ids, candidate_coords):
                if not np.all(np.isfinite(point)):
                    continue
                dist, idx = tree.query(point, k=1)
                if np.isfinite(dist) and float(dist) <= match_distance:
                    matched_id = int(pert_ids[int(idx)])
                    if matched_id in selected_ids:
                        hits[int(cid)] += 1
                        selected_variants[int(cid)].append(str(variant))
            variant_status.append(f"{variant}:ok")
        except Exception as exc:
            variant_status.append(f"{variant}:error:{exc}")
            continue

    min_trials = int(getattr(config, "min_perturbation_stability_trials", 2))
    min_score = float(getattr(config, "min_perturbation_stability_score", 0.66))
    rows: list[dict[str, Any]] = []
    for cid in candidate_ids:
        score = float(hits[int(cid)] / max(trials, 1)) if trials else np.nan
        pass_flag = bool(trials < min_trials or (np.isfinite(score) and score >= min_score))
        rows.append(
            {
                "colony_id": int(cid),
                "perturbation_stability_score": score,
                "perturbation_stability_hits": int(hits[int(cid)]),
                "perturbation_stability_trials": int(trials),
                "perturbation_stability_pass": pass_flag,
                "perturbation_stability_variants": ";".join(map(str, variants)),
                "perturbation_stability_selected_variants": ";".join(selected_variants[int(cid)]),
                "perturbation_stability_status": ";".join(variant_status),
            }
        )
    return pd.DataFrame(rows, columns=PERTURBATION_STABILITY_COLUMNS)


def apply_perturbation_stability_filter(
    scores_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    config: AnomalyDetectionConfig,
) -> pd.DataFrame:
    if scores_df is None or scores_df.empty or stability_df is None or stability_df.empty:
        return scores_df
    if "colony_id" not in scores_df.columns or "colony_id" not in stability_df.columns:
        return scores_df

    result = scores_df.copy()
    stability_cols = [c for c in PERTURBATION_STABILITY_COLUMNS if c != "colony_id" and c in stability_df.columns]
    for col in stability_cols:
        if col not in result.columns:
            result[col] = np.nan

    lookup = stability_df.set_index(stability_df["colony_id"].astype(int))
    for idx, row in result.iterrows():
        try:
            colony_id = int(row.get("colony_id"))
        except Exception:
            continue
        if colony_id not in lookup.index:
            continue
        stable_row = lookup.loc[colony_id]
        if isinstance(stable_row, pd.DataFrame):
            stable_row = stable_row.iloc[0]
        for col in stability_cols:
            result.at[idx, col] = stable_row.get(col, np.nan)

        trials = int(stable_row.get("perturbation_stability_trials", 0) or 0)
        passed = bool(stable_row.get("perturbation_stability_pass", False))
        if (
            trials >= int(getattr(config, "min_perturbation_stability_trials", 2))
            and not passed
            and str(row.get("recommendation_status", "")) == "select_candidate"
        ):
            result.at[idx, "recommendation_status"] = "unstable_candidate"
            result.at[idx, "selected_for_further_analysis"] = False

    return result


def run_stability_test(
    image_path: str | Path,
    model,
    config: AnomalyDetectionConfig,
    top_k: int = 5,
    petri_detector_model=None,
) -> pd.DataFrame:
    image_path = Path(image_path)
    rng = np.random.default_rng(config.random_state)

    stability_config = AnomalyDetectionConfig(**config.__dict__)
    stability_config.top_k = int(top_k)
    stability_config.output_dir = str(Path(config.output_dir) / "stability_test" / image_path.stem)
    stability_config.save_visualizations = False
    stability_config.save_csv = False
    stability_config.save_xlsx = False

    baseline_result = run_single_image_pipeline(
        image_path=image_path,
        model=model,
        config=stability_config,
        petri_detector_model=petri_detector_model,
    )
    baseline_scores = baseline_result["scores"]
    baseline_top = _selected_centroid_table(baseline_scores, top_k=top_k)
    baseline_top_ids = set(baseline_top["colony_id"].astype(int).tolist()) if not baseline_top.empty else set()

    original_rgb = read_image_rgb(image_path)
    temp_dir = Path(config.output_dir) / "_stability_inputs" / image_path.stem
    temp_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        "brightness_plus_5pct",
        "brightness_minus_5pct",
        "contrast_plus_5pct",
        "contrast_minus_5pct",
        "slight_blur",
        "slight_gaussian_noise",
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        perturbed_rgb = _apply_stability_perturbation(original_rgb, variant, rng)
        perturbed_path = temp_dir / f"{image_path.stem}_{variant}.png"
        cv2.imwrite(str(perturbed_path), cv2.cvtColor(perturbed_rgb, cv2.COLOR_RGB2BGR))

        try:
            result = run_single_image_pipeline(
                image_path=perturbed_path,
                model=model,
                config=stability_config,
                petri_detector_model=petri_detector_model,
            )
            perturbed_scores = result["scores"]
            mapped_ids, n_common, jaccard = _match_selected_to_baseline(
                baseline_scores=baseline_scores,
                perturbed_scores=perturbed_scores,
                top_k=top_k,
            )
            spearman_corr, n_matched = _spearman_by_centroid_matching(
                baseline_scores=baseline_scores,
                perturbed_scores=perturbed_scores,
            )
            rows.append(
                {
                    "variant": variant,
                    "baseline_top_k_ids": ";".join(map(str, sorted(baseline_top_ids))),
                    "matched_perturbed_top_k_as_baseline_ids": ";".join(map(str, sorted(mapped_ids))),
                    "top_k_jaccard": jaccard,
                    "n_common_top_k": n_common,
                    "spearman_corr_scores": spearman_corr,
                    "n_matched_for_spearman": n_matched,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "variant": variant,
                    "baseline_top_k_ids": ";".join(map(str, sorted(baseline_top_ids))),
                    "matched_perturbed_top_k_as_baseline_ids": "",
                    "top_k_jaccard": np.nan,
                    "n_common_top_k": 0,
                    "spearman_corr_scores": np.nan,
                    "n_matched_for_spearman": 0,
                    "status": f"error: {exc}",
                }
            )

    stability_df = pd.DataFrame(rows)
    if not stability_df.empty:
        spearman_component = (stability_df["spearman_corr_scores"].clip(-1, 1) + 1.0) / 2.0
        stability_df["stability_score"] = np.nanmean(
            np.vstack([
                stability_df["top_k_jaccard"].to_numpy(dtype=float),
                spearman_component.to_numpy(dtype=float),
            ]),
            axis=0,
        )
    output_dir = Path(config.output_dir) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(output_dir / "stability_test.csv", index=False, encoding="utf-8-sig")
    return stability_df

DEFAULT_PETRI_IMAGES: list[Path] = []


def make_full_pipeline_config(
    output_dir: str | Path = "outputs/colony_anomaly_detection/petri_full_pipeline",
) -> AnomalyDetectionConfig:
    """Create config for raw Petri photos: detector first, then 736x736 segmentation."""
    return replace(
        AnomalyDetectionConfig(),
        preprocess_mode="detect_petri",
        use_petri_detector=True,
        output_dir=str(output_dir),
    )


def load_pipeline_models(config: AnomalyDetectionConfig):
    """Load segmentation model X and Petri dish detector."""
    if config.model_weights_path:
        seg_weights_path = Path(config.model_weights_path)
        if not seg_weights_path.exists():
            raise FileNotFoundError(f"Segmentation weights file not found: {seg_weights_path}")
    else:
        seg_weights_path = find_best_yolo_weights(config.model_dir)

    petri_weights_path = Path(config.petri_detector_weights_path)
    if not petri_weights_path.exists():
        raise FileNotFoundError(f"Petri detector weights file not found: {petri_weights_path}")

    print("Segmentation weights:", seg_weights_path)
    print("Petri detector weights:", petri_weights_path)
    segmentation_model = YOLO(str(seg_weights_path))
    petri_detector_model = YOLO(str(petri_weights_path))
    return segmentation_model, petri_detector_model


def run_full_pipeline_for_images(
    image_paths: list[str | Path] | None = None,
    config: AnomalyDetectionConfig | None = None,
) -> dict[str, Any]:
    """Run full two-model pipeline for a fixed list of Petri images."""
    if config is None:
        config = make_full_pipeline_config()
    else:
        config = replace(config, preprocess_mode="detect_petri", use_petri_detector=True)

    paths = [Path(p) for p in (image_paths or DEFAULT_PETRI_IMAGES)]
    if not paths:
        raise ValueError("Provide input image paths explicitly; no private sample images are bundled.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input images: " + ", ".join(str(p) for p in missing))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmentation_model, petri_detector_model = load_pipeline_models(config)

    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    selected_tables: list[pd.DataFrame] = []
    review_tables: list[pd.DataFrame] = []
    technical_tables: list[pd.DataFrame] = []

    for image_path in paths:
        print(f"\n=== Full pipeline: {image_path.name} ===")
        result = run_single_image_pipeline(
            image_path=image_path,
            model=segmentation_model,
            config=config,
            petri_detector_model=petri_detector_model,
        )
        results[image_path.name] = result

        scores_df = result.get("scores", pd.DataFrame())
        selected_df = result.get("selected", pd.DataFrame())
        review_df = result.get("review_candidates", pd.DataFrame())
        technical_df = result.get("technical_warnings", pd.DataFrame())
        plate_quality_df = result.get("plate_quality", pd.DataFrame())

        n_detected = int(len(scores_df))
        n_valid = (
            int(scores_df["valid_for_anomaly"].fillna(False).astype(bool).sum())
            if n_detected and "valid_for_anomaly" in scores_df.columns
            else 0
        )
        n_selected = int(len(selected_df))
        n_review = int(len(review_df))
        n_technical = int(len(technical_df))
        max_score = (
            float(scores_df["final_anomaly_score_raw"].max())
            if n_detected and "final_anomaly_score_raw" in scores_df.columns
            else np.nan
        )
        plate_quality_status = (
            str(plate_quality_df.loc[0, "plate_quality_status"])
            if plate_quality_df is not None and not plate_quality_df.empty and "plate_quality_status" in plate_quality_df.columns
            else ("no_masks" if n_detected == 0 else ("no_valid_colonies" if n_valid == 0 else "ok"))
        )

        image_output_dir = output_dir / image_path.stem
        summary_rows.append(
            {
                "image_name": image_path.name,
                "n_detected_colonies": n_detected,
                "n_valid_colonies": n_valid,
                "n_selected_anomalies": n_selected,
                "n_review_candidates": n_review,
                "n_technical_warnings": n_technical,
                "max_anomaly_score_raw": max_score,
                "plate_quality_status": plate_quality_status,
                "output_dir": str(image_output_dir),
            }
        )

        if not selected_df.empty:
            tmp = selected_df.copy()
            tmp.insert(0, "image_name", image_path.name)
            selected_tables.append(tmp)
        if not review_df.empty:
            tmp = review_df.copy()
            tmp.insert(0, "image_name", image_path.name)
            review_tables.append(tmp)
        if not technical_df.empty:
            tmp = technical_df.copy()
            tmp.insert(0, "image_name", image_path.name)
            technical_tables.append(tmp)

        print(
            f"Detected: {n_detected}, valid: {n_valid}, selected: {n_selected}, "
            f"review: {n_review}, technical: {n_technical}, quality: {plate_quality_status}"
        )
        print("Output folder:", image_output_dir)
        fig = result.get("figure")
        if fig is not None:
            plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    all_selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    all_review = pd.concat(review_tables, ignore_index=True) if review_tables else pd.DataFrame()
    all_technical = pd.concat(technical_tables, ignore_index=True) if technical_tables else pd.DataFrame()

    summary_df.to_csv(output_dir / "petri_5_images_summary.csv", index=False, encoding="utf-8-sig")
    all_selected.to_csv(output_dir / "petri_5_images_selected_anomalies.csv", index=False, encoding="utf-8-sig")
    all_review.to_csv(output_dir / "petri_5_images_review_candidates.csv", index=False, encoding="utf-8-sig")
    all_technical.to_csv(output_dir / "petri_5_images_technical_warnings.csv", index=False, encoding="utf-8-sig")

    if config.save_xlsx:
        try:
            with pd.ExcelWriter(output_dir / "petri_5_images_report.xlsx") as writer:
                summary_df.to_excel(writer, sheet_name="summary", index=False)
                all_selected.to_excel(writer, sheet_name="selected_anomalies", index=False)
                all_review.to_excel(writer, sheet_name="review_candidates", index=False)
                all_technical.to_excel(writer, sheet_name="technical_warnings", index=False)
        except Exception as exc:
            print(f"Could not save Excel report for 5 images: {exc}")

    return {
        "config": config,
        "results": results,
        "summary": summary_df,
        "selected": all_selected,
        "review_candidates": all_review,
        "technical_warnings": all_technical,
        "output_dir": output_dir,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Local Petri dish detection, segmentation and anomaly analysis")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/full_pipeline"))
    args = parser.parse_args()
    result = run_full_pipeline_for_images(args.images, make_full_pipeline_config(args.output))
    print("\n=== Summary ===")
    print(result["summary"].to_string(index=False))
    print("\nSaved to:", result["output_dir"])


if __name__ == "__main__":
    main()

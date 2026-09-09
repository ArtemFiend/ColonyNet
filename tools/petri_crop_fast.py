from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower() or ".png"
    if not path.suffix:
        path = path.with_suffix(ext)

    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def draw_circle_overlay(
    img_bgr: np.ndarray,
    circle: tuple[float, float, float] | None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray:
    out = img_bgr.copy()
    if circle is None:
        return out

    cx, cy, r = circle
    center = (int(round(cx)), int(round(cy)))
    radius = int(round(r))
    cv2.circle(out, center, radius, color, thickness)
    cv2.circle(out, center, max(4, thickness + 1), (255, 0, 0), -1)
    return out


def resize_for_detection(
    img_bgr: np.ndarray,
    max_side: int = 1400,
) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img_bgr.copy(), scale

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    values = np.clip(values, low, high)
    values = (values - low) * (255.0 / (high - low))
    return values.astype(np.uint8)


def make_clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def build_candidate_masks(img_bgr_small: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], dict[str, Any]]:
    h, w = img_bgr_small.shape[:2]
    lab = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2HSV)

    border = max(10, int(round(min(h, w) * 0.05)))
    border_pixels = np.concatenate(
        [
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg_lab = np.median(border_pixels, axis=0)

    color_dist = np.linalg.norm(lab.astype(np.float32) - bg_lab.reshape(1, 1, 3), axis=2)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    color_dist_u8 = normalize_to_uint8(color_dist)
    saturation_u8 = normalize_to_uint8(saturation)
    value_u8 = normalize_to_uint8(value)

    fused = cv2.addWeighted(color_dist_u8, 0.75, saturation_u8, 0.25, 0)
    fused = cv2.GaussianBlur(fused, (9, 9), 0)

    _, mask_fused = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_dist = cv2.threshold(color_dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_sat = cv2.threshold(saturation_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_dark_bg = cv2.threshold(value_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_size = max(5, int(round(min(h, w) * 0.03)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    masks = [
        ("fused", make_clean_mask(mask_fused, kernel_size)),
        ("distance", make_clean_mask(mask_dist, kernel_size)),
        ("saturation", make_clean_mask(mask_sat, kernel_size)),
        ("dark_bg", make_clean_mask(mask_dark_bg, kernel_size)),
    ]
    debug = {
        "bg_lab": tuple(float(v) for v in bg_lab),
        "border": border,
        "kernel_size": kernel_size,
    }
    return masks, debug


def contour_score(
    contour: np.ndarray,
    img_shape: tuple[int, int],
    min_r: float,
    max_r: float,
) -> tuple[float, tuple[float, float, float]] | None:
    h, w = img_shape
    area = cv2.contourArea(contour)
    if area < math.pi * (min_r**2) * 0.35:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    hull = cv2.convexHull(contour)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    if not (min_r <= r <= max_r):
        return None

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    fill_ratio = area / max(math.pi * r * r, 1e-6)
    size_score = r / max(max_r, 1e-6)

    dist_to_center = math.hypot(cx - w / 2.0, cy - h / 2.0) / (0.5 * min(h, w))
    center_score = max(0.0, 1.0 - dist_to_center)

    score = (
        0.35 * fill_ratio
        + 0.25 * circularity
        + 0.25 * size_score
        + 0.15 * center_score
    )
    return score, (float(cx), float(cy), float(r))


def detect_petri_circle_fast(
    img_bgr: np.ndarray,
    min_r_frac: float = 0.18,
    max_r_frac: float = 0.52,
    detection_max_side: int = 1400,
) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
    small, scale = resize_for_detection(img_bgr, max_side=detection_max_side)
    h, w = small.shape[:2]
    min_r = min(h, w) * min_r_frac
    max_r = min(h, w) * max_r_frac

    candidate_masks, debug = build_candidate_masks(small)

    best_circle = None
    best_score = -1.0
    best_method = None
    contour_count = 0

    for method_name, mask in candidate_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count += len(contours)

        for contour in contours:
            scored = contour_score(contour, (h, w), min_r=min_r, max_r=max_r)
            if scored is None:
                continue

            score, circle_small = scored
            if score > best_score:
                cx, cy, r = circle_small
                best_score = score
                best_circle = (cx / scale, cy / scale, r / scale)
                best_method = method_name

    meta: dict[str, Any] = {
        "method": best_method,
        "score": None if best_circle is None else float(best_score),
        "scale": float(scale),
        "small_shape": (int(h), int(w)),
        "contours_seen": int(contour_count),
    }
    meta.update(debug)
    return best_circle, meta


def crop_petri_square_bgr(
    img_bgr: np.ndarray,
    pad_frac: float = 0.03,
    fill_value: int = 255,
    mask_outside: bool = False,
    min_r_frac: float = 0.18,
    max_r_frac: float = 0.52,
    detection_max_side: int = 1400,
) -> tuple[np.ndarray, dict[str, Any] | None, str, dict[str, Any]]:
    circle, meta = detect_petri_circle_fast(
        img_bgr,
        min_r_frac=min_r_frac,
        max_r_frac=max_r_frac,
        detection_max_side=detection_max_side,
    )
    if circle is None:
        return img_bgr.copy(), None, "no_circle", meta

    cx, cy, r = circle
    r = r * (1.0 + pad_frac)

    h, w = img_bgr.shape[:2]
    side = int(math.ceil(2.0 * r))
    half = side / 2.0

    x1 = int(math.floor(cx - half))
    y1 = int(math.floor(cy - half))
    x2 = x1 + side
    y2 = y1 + side

    left_pad = max(0, -x1)
    top_pad = max(0, -y1)
    right_pad = max(0, x2 - w)
    bottom_pad = max(0, y2 - h)

    if any(v > 0 for v in (left_pad, top_pad, right_pad, bottom_pad)):
        img_bgr = cv2.copyMakeBorder(
            img_bgr,
            top=top_pad,
            bottom=bottom_pad,
            left=left_pad,
            right=right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=(fill_value, fill_value, fill_value),
        )
        cx += left_pad
        cy += top_pad
        x1 += left_pad
        x2 += left_pad
        y1 += top_pad
        y2 += top_pad

    crop = img_bgr[y1:y2, x1:x2].copy()

    if mask_outside:
        yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
        local_cx = cx - x1
        local_cy = cy - y1
        circle_mask = (xx - local_cx) ** 2 + (yy - local_cy) ** 2 <= r**2
        crop[~circle_mask] = (fill_value, fill_value, fill_value)

    info = {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "side": int(side),
        "method": meta["method"],
        "score": meta["score"],
    }
    return crop, info, "cropped", meta


def resize_to_square(img_bgr: np.ndarray, out_size: int = 512) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    interpolation = cv2.INTER_AREA if max(h, w) > out_size else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (out_size, out_size), interpolation=interpolation)


def preprocess_petri_image_file(
    input_path: str | Path,
    output_path: str | Path,
    out_size: int = 512,
    pad_frac: float = 0.03,
    fill_value: int = 255,
    mask_outside: bool = False,
    fallback_resize_full_image: bool = False,
    save_debug: bool = False,
    debug_dir: str | Path | None = None,
    detection_max_side: int = 1400,
) -> dict[str, Any]:
    img_bgr = imread_unicode(input_path)
    if img_bgr is None:
        return {"file": str(input_path), "status": "read_error"}

    crop, info, status, meta = crop_petri_square_bgr(
        img_bgr=img_bgr,
        pad_frac=pad_frac,
        fill_value=fill_value,
        mask_outside=mask_outside,
        detection_max_side=detection_max_side,
    )

    if status == "no_circle":
        if fallback_resize_full_image:
            result = resize_to_square(img_bgr, out_size=out_size)
            final_status = "fallback_full_image"
        else:
            if save_debug and debug_dir is not None:
                debug_path = Path(debug_dir) / f"{Path(input_path).stem}_FAIL.jpg"
                imwrite_unicode(debug_path, img_bgr)
            return {
                "file": str(input_path),
                "status": "no_circle",
                "info": None,
                "meta": meta,
            }
    else:
        result = resize_to_square(crop, out_size=out_size)
        final_status = "cropped"

    ok = imwrite_unicode(output_path, result)

    if save_debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        circle = None if info is None else (info["cx"], info["cy"], info["r"] / (1.0 + pad_frac))
        overlay = draw_circle_overlay(img_bgr, circle)
        debug_path = debug_dir / f"{Path(input_path).stem}_{final_status}.jpg"
        imwrite_unicode(debug_path, overlay)

    return {
        "file": str(input_path),
        "saved_to": str(output_path) if ok else None,
        "status": final_status if ok else "save_error",
        "info": info,
        "meta": meta,
    }


def process_petri_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    out_size: int = 512,
    pad_frac: float = 0.03,
    fill_value: int = 255,
    mask_outside: bool = False,
    recursive: bool = False,
    fallback_resize_full_image: bool = False,
    save_debug: bool = False,
    detection_max_side: int = 1400,
    exts: tuple[str, ...] = DEFAULT_EXTS,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "_debug" if save_debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return []

    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    files = sorted(files)
    if not files:
        print("No images found.")
        return []

    results: list[dict[str, Any]] = []
    for index, input_path in enumerate(files, start=1):
        output_path = output_dir / input_path.relative_to(input_dir) if recursive else output_dir / input_path.name
        output_path = output_path.with_suffix(".png")

        result = preprocess_petri_image_file(
            input_path=input_path,
            output_path=output_path,
            out_size=out_size,
            pad_frac=pad_frac,
            fill_value=fill_value,
            mask_outside=mask_outside,
            fallback_resize_full_image=fallback_resize_full_image,
            save_debug=save_debug,
            debug_dir=debug_dir,
            detection_max_side=detection_max_side,
        )
        results.append(result)
        print(f"[{index}/{len(files)}] {input_path.name} -> {result['status']}")

    total = len(results)
    cropped = sum(item["status"] == "cropped" for item in results)
    fallback = sum(item["status"] == "fallback_full_image" for item in results)
    no_circle = sum(item["status"] == "no_circle" for item in results)
    errors = sum(item["status"] in {"read_error", "save_error"} for item in results)

    print("\nDone.")
    print(f"Total files: {total}")
    print(f"Cropped: {cropped}")
    print(f"Fallback resize: {fallback}")
    print(f"No circle: {no_circle}")
    print(f"Read/save errors: {errors}")

    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop Petri dishes from images.")
    parser.add_argument("--input_dir", required=True, help="Folder with source images.")
    parser.add_argument("--output_dir", required=True, help="Folder for cropped images.")
    parser.add_argument("--out_size", type=int, default=512, help="Final square output size.")
    parser.add_argument("--pad_frac", type=float, default=0.03, help="Extra padding around detected circle.")
    parser.add_argument("--fill_value", type=int, default=255, help="Border and mask fill value.")
    parser.add_argument("--mask_outside", action="store_true", help="Paint pixels outside the dish white.")
    parser.add_argument("--recursive", action="store_true", help="Process files recursively.")
    parser.add_argument(
        "--fallback_resize_full_image",
        action="store_true",
        help="Resize the original image if no dish is found.",
    )
    parser.add_argument("--save_debug", action="store_true", help="Save overlay images with the detected circle.")
    parser.add_argument(
        "--detection_max_side",
        type=int,
        default=1400,
        help="Resize long edge to this size before detection for speed.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    process_petri_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        out_size=args.out_size,
        pad_frac=args.pad_frac,
        fill_value=args.fill_value,
        mask_outside=args.mask_outside,
        recursive=args.recursive,
        fallback_resize_full_image=args.fallback_resize_full_image,
        save_debug=args.save_debug,
        detection_max_side=args.detection_max_side,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMG_KEY_RE = re.compile(r"(IMG_\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


@dataclass(frozen=True)
class MaskRef:
    root: Path
    root_alias: str
    path: Path


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, flags)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    try:
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def extract_img_key(text: str) -> str:
    match = IMG_KEY_RE.search(text)
    if match:
        return match.group(1).upper()
    return Path(text).stem.upper()


def infer_key_from_path(path: Path) -> str:
    candidates = [
        path.stem,
        path.name,
        path.parent.name,
        path.parent.parent.name if path.parent != path else path.parent.name,
    ]
    for candidate in candidates:
        key = extract_img_key(candidate)
        if key.startswith("IMG_"):
            return key
    return path.stem.upper()


def collect_files(root: Path, exts: set[str]) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def pick_best_box(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
) -> CropBox | None:
    kwargs = {
        "source": image_bgr,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
        "max_det": max_det,
    }
    if device:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if not results:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    xyxy = boxes.xyxy[best_idx].detach().cpu().numpy()
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]

    h, w = image_bgr.shape[:2]
    x1 = min(max(x1, 0), max(w - 1, 0))
    y1 = min(max(y1, 0), max(h - 1, 0))
    x2 = min(max(x2, x1 + 1), w)
    y2 = min(max(y2, y1 + 1), h)

    return CropBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(confs[best_idx]))


def expand_box(box: CropBox, width: int, height: int, pad_frac: float) -> CropBox:
    box_w = box.x2 - box.x1
    box_h = box.y2 - box.y1
    pad_x = int(round(box_w * pad_frac))
    pad_y = int(round(box_h * pad_frac))

    x1 = max(0, box.x1 - pad_x)
    y1 = max(0, box.y1 - pad_y)
    x2 = min(width, box.x2 + pad_x)
    y2 = min(height, box.y2 + pad_y)

    if x2 <= x1:
        x1, x2 = 0, width
    if y2 <= y1:
        y1, y2 = 0, height

    return CropBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=box.conf)


def resize_crop(image: np.ndarray, out_size: int | None, is_mask: bool) -> np.ndarray:
    if out_size is None:
        return image
    interpolation = cv2.INTER_NEAREST if is_mask else (
        cv2.INTER_AREA if max(image.shape[:2]) > out_size else cv2.INTER_CUBIC
    )
    return cv2.resize(image, (out_size, out_size), interpolation=interpolation)


def scale_box_between_shapes(
    box: CropBox,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> CropBox:
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)

    x1 = int(round(box.x1 * sx))
    y1 = int(round(box.y1 * sy))
    x2 = int(round(box.x2 * sx))
    y2 = int(round(box.y2 * sy))

    x1 = min(max(x1, 0), max(dst_w - 1, 0))
    y1 = min(max(y1, 0), max(dst_h - 1, 0))
    x2 = min(max(x2, x1 + 1), dst_w)
    y2 = min(max(y2, y1 + 1), dst_h)
    return CropBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=box.conf)


def save_overlay(image_bgr: np.ndarray, box: CropBox, out_path: Path) -> bool:
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 4)
    cv2.putText(
        overlay,
        f"conf={box.conf:.3f}",
        (box.x1, max(24, box.y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return imwrite_unicode(out_path, overlay)


def build_image_index(image_dirs: Sequence[Path]) -> tuple[Dict[str, Path], Dict[str, List[Path]]]:
    primary: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for image_dir in image_dirs:
        for path in collect_files(image_dir, IMAGE_EXTS):
            key = infer_key_from_path(path)
            if key not in primary:
                primary[key] = path
            else:
                duplicates.setdefault(key, []).append(path)
    return primary, duplicates


def make_root_aliases(mask_dirs: Sequence[Path]) -> Dict[Path, str]:
    counts: Dict[str, int] = {}
    aliases: Dict[Path, str] = {}
    for root in mask_dirs:
        base = root.name or "mask_root"
        counts[base] = counts.get(base, 0) + 1
        suffix = counts[base]
        alias = f"{base}_{suffix}" if suffix > 1 else base
        aliases[root] = alias
    return aliases


def build_mask_index(mask_dirs: Sequence[Path]) -> Dict[str, List[MaskRef]]:
    aliases = make_root_aliases(mask_dirs)
    index: Dict[str, List[MaskRef]] = {}
    for root in mask_dirs:
        alias = aliases[root]
        for path in collect_files(root, MASK_EXTS):
            key = infer_key_from_path(path)
            index.setdefault(key, []).append(MaskRef(root=root, root_alias=alias, path=path))
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop ROI by trained YOLO detector and apply the same crop to images and masks."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to YOLO weights (.pt)")
    parser.add_argument(
        "--image_dirs",
        required=True,
        nargs="+",
        type=Path,
        help="One or more image directories; files are scanned recursively.",
    )
    parser.add_argument(
        "--mask_dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Zero or more mask directories; crop is matched by IMG_XXXX key.",
    )
    parser.add_argument("--out_dir", required=True, type=Path, help="Output root directory")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--max_det", type=int, default=10, help="Max detections per image")
    parser.add_argument("--pad_frac", type=float, default=0.03, help="Relative bbox padding")
    parser.add_argument(
        "--output_size",
        type=int,
        default=None,
        help="Optional square output size; if omitted, keeps crop size.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or 0")
    parser.add_argument(
        "--fallback_full_image",
        action="store_true",
        help="If no detection, use full image as ROI.",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save image overlays with the selected ROI box.",
    )
    return parser.parse_args()


def log_json(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "key",
        "status",
        "source_image",
        "conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_saved",
        "masks_total",
        "masks_saved",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_dirs = [p.resolve() for p in args.image_dirs]
    mask_dirs = [p.resolve() for p in args.mask_dirs]
    out_dir: Path = args.out_dir.resolve()
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    out_overlay = out_dir / "overlay"

    for directory in image_dirs + mask_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

    print(f"Model: {model_path}")
    print("Image dirs:")
    for d in image_dirs:
        print(f"  - {d}")
    if mask_dirs:
        print("Mask dirs:")
        for d in mask_dirs:
            print(f"  - {d}")
    else:
        print("Mask dirs: none")
    print(f"Output dir: {out_dir}")

    model = YOLO(str(model_path))
    image_index, duplicates = build_image_index(image_dirs)
    mask_index = build_mask_index(mask_dirs)

    if not image_index:
        print("No input images found.")
        return

    if duplicates:
        print(f"Duplicate image keys (kept first match): {len(duplicates)}")

    rows: List[dict] = []
    keys = sorted(image_index.keys())
    total = len(keys)
    saved_images = 0
    saved_masks = 0
    missed = 0

    for i, key in enumerate(keys, start=1):
        image_path = image_index[key]
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                {
                    "key": key,
                    "status": "image_read_error",
                    "source_image": str(image_path),
                    "note": "failed to read image",
                    "masks_total": len(mask_index.get(key, [])),
                    "masks_saved": 0,
                }
            )
            missed += 1
            print(f"[{i}/{total}] {key}: image_read_error")
            continue

        best = pick_best_box(
            model=model,
            image_bgr=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )

        status = "cropped"
        note = ""
        if best is None:
            if args.fallback_full_image:
                h, w = image.shape[:2]
                best = CropBox(0, 0, w, h, conf=0.0)
                status = "fallback_full_image"
                note = "no_detection_full_image_used"
            else:
                rows.append(
                    {
                        "key": key,
                        "status": "no_detection",
                        "source_image": str(image_path),
                        "note": "no bbox detected",
                        "masks_total": len(mask_index.get(key, [])),
                        "masks_saved": 0,
                    }
                )
                missed += 1
                print(f"[{i}/{total}] {key}: no_detection")
                continue

        h, w = image.shape[:2]
        box = expand_box(best, width=w, height=h, pad_frac=args.pad_frac)
        image_crop = image[box.y1:box.y2, box.x1:box.x2]
        image_crop = resize_crop(image_crop, args.output_size, is_mask=False)

        image_out_path = out_images / f"{key}.png"
        image_ok = imwrite_unicode(image_out_path, image_crop)
        if image_ok:
            saved_images += 1

        if args.save_overlay:
            save_overlay(image, box, out_overlay / f"{key}.png")

        masks_for_key = mask_index.get(key, [])
        masks_saved_for_key = 0
        for mask_ref in masks_for_key:
            mask = imread_unicode(mask_ref.path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            mh, mw = mask.shape[:2]
            if mh == h and mw == w:
                box_m = box
            else:
                box_m = scale_box_between_shapes(box, src_w=w, src_h=h, dst_w=mw, dst_h=mh)

            mask_crop = mask[box_m.y1:box_m.y2, box_m.x1:box_m.x2]
            mask_crop = resize_crop(mask_crop, args.output_size, is_mask=True)

            rel = mask_ref.path.relative_to(mask_ref.root)
            mask_out = out_masks / mask_ref.root_alias / rel
            mask_out = mask_out.with_suffix(".png")
            if imwrite_unicode(mask_out, mask_crop):
                masks_saved_for_key += 1
                saved_masks += 1

        rows.append(
            {
                "key": key,
                "status": status,
                "source_image": str(image_path),
                "conf": round(box.conf, 6),
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
                "image_saved": int(image_ok),
                "masks_total": len(masks_for_key),
                "masks_saved": masks_saved_for_key,
                "note": note,
            }
        )
        print(
            f"[{i}/{total}] {key}: {status}, "
            f"box=({box.x1},{box.y1},{box.x2},{box.y2}), masks={masks_saved_for_key}/{len(masks_for_key)}"
        )

    log_csv(out_dir / "crop_report.csv", rows)
    log_json(out_dir / "crop_report.jsonl", rows)

    summary = {
        "total_keys": total,
        "saved_images": saved_images,
        "saved_masks": saved_masks,
        "missed": missed,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

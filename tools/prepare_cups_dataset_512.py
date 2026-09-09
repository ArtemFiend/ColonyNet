from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import rawpy
except ImportError:  # pragma: no cover - optional dependency for CR3 support
    rawpy = None


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".cr3"}
DEFAULT_INPUT = Path("датасеты_чашки")
DEFAULT_OUTPUT = Path("датасеты_чашки_cropped_512")


@dataclass
class ProcessResult:
    source: str
    output: str
    status: str
    orig_hw: tuple[int, int]
    crop_hw: tuple[int, int]
    final_hw: tuple[int, int]
    circle: tuple[int, int, int] | None
    bbox: tuple[int, int, int, int] | None
    error: str | None = None


def detect_petri_circle(
    img_rgb: np.ndarray,
    min_r_frac: float = 0.35,
    max_r_frac: float = 0.55,
    center_tol: float = 0.25,
) -> tuple[int, int, int] | None:
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


def crop_petri(img_rgb: np.ndarray, pad: float = 0.02, mask_outside: bool = True) -> tuple[np.ndarray, dict[str, Any], str]:
    h, w = img_rgb.shape[:2]
    circ = detect_petri_circle(img_rgb)
    if circ is None:
        return img_rgb.copy(), {"orig_hw": (h, w), "circle": None, "bbox": None}, "no_circle"

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
        "orig_hw": (h, w),
        "circle": (int(cx), int(cy), int(r)),
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
    }
    return crop, meta, "cropped"


def overlay_boundaries(img: np.ndarray, lbl: np.ndarray) -> np.ndarray:
    out = img.copy()
    b = boundary_mask(lbl)
    out[b] = (0, 255, 0)
    return out


def boundary_mask(lbl: np.ndarray) -> np.ndarray:
    lbl = np.asarray(lbl)
    if lbl.ndim != 2:
        raise ValueError("Label image must be 2D")

    b = np.zeros(lbl.shape, dtype=bool)
    b[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    b[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    return b


def read_image_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".cr3":
        if rawpy is None:
            raise RuntimeError("rawpy is required to read CR3 files")
        with rawpy.imread(str(path)) as raw:
            img = raw.postprocess(use_camera_wb=True, output_bps=8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: Path, img_rgb: np.ndarray, output_ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    ext = output_ext.lower()
    if ext in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(ext, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        ok, encoded = cv2.imencode(ext, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, encoded = cv2.imencode(ext, img_bgr)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def pad_to_square(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb

    side = max(h, w)
    padded = np.zeros((side, side, img_rgb.shape[2]), dtype=img_rgb.dtype)
    y = (side - h) // 2
    x = (side - w) // 2
    padded[y : y + h, x : x + w] = img_rgb
    return padded


def resize_square(img_rgb: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if max(img_rgb.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img_rgb, (size, size), interpolation=interp)


def is_source_image(path: Path, input_root: Path, output_root: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_EXTS:
        return False
    if output_root in path.parents:
        return False

    rel_parts = [part.casefold() for part in path.relative_to(input_root).parts]
    if "разметка" in rel_parts:
        return False
    return True


def iter_source_images(input_root: Path, output_root: Path) -> list[Path]:
    files = [p for p in input_root.rglob("*") if p.is_file() and is_source_image(p, input_root, output_root)]
    return sorted(files)


def process_one(path: Path, input_root: Path, output_root: Path, size: int, pad: float, output_ext: str) -> ProcessResult:
    rel = path.relative_to(input_root)
    out_path = output_root / rel.with_suffix(output_ext)

    try:
        img_rgb = read_image_rgb(path)
        crop, meta, status = crop_petri(img_rgb, pad=pad, mask_outside=True)
        squared = pad_to_square(crop)
        resized = resize_square(squared, size=size)
        write_image_rgb(out_path, resized, output_ext=output_ext)

        return ProcessResult(
            source=str(path),
            output=str(out_path),
            status=status,
            orig_hw=tuple(int(v) for v in meta["orig_hw"]),
            crop_hw=(int(crop.shape[0]), int(crop.shape[1])),
            final_hw=(int(resized.shape[0]), int(resized.shape[1])),
            circle=None if meta["circle"] is None else tuple(int(v) for v in meta["circle"]),
            bbox=None if meta["bbox"] is None else tuple(int(v) for v in meta["bbox"]),
        )
    except Exception as exc:
        return ProcessResult(
            source=str(path),
            output=str(out_path),
            status="error",
            orig_hw=(0, 0),
            crop_hw=(0, 0),
            final_hw=(0, 0),
            circle=None,
            bbox=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def write_metadata(output_root: Path, results: list[ProcessResult]) -> None:
    metadata_path = output_root / "metadata.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def build_summary(results: list[ProcessResult], output_root: Path, size: int, output_ext: str) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    errors: list[dict[str, str]] = []

    for item in results:
        by_status[item.status] = by_status.get(item.status, 0) + 1
        if item.error is not None:
            errors.append({"source": item.source, "error": item.error})

    summary = {
        "output_root": str(output_root),
        "image_size": size,
        "output_ext": output_ext,
        "total": len(results),
        "by_status": by_status,
        "errors": errors,
    }

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop Petri dishes from the cups dataset and resize images to 512x512."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input dataset root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset root")
    parser.add_argument("--size", type=int, default=512, help="Final square image size")
    parser.add_argument("--pad", type=float, default=0.02, help="Relative circle padding")
    parser.add_argument(
        "--output-ext",
        default=".png",
        choices=[".png", ".jpg", ".jpeg"],
        help="Output image extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    files = iter_source_images(input_root, output_root)
    if not files:
        raise RuntimeError(f"No source images found in {input_root}")

    results: list[ProcessResult] = []
    total = len(files)
    for idx, path in enumerate(files, start=1):
        result = process_one(
            path=path,
            input_root=input_root,
            output_root=output_root,
            size=args.size,
            pad=args.pad,
            output_ext=args.output_ext,
        )
        results.append(result)
        print(f"[{idx}/{total}] {result.status}: {path}")

    write_metadata(output_root, results)
    summary = build_summary(results, output_root=output_root, size=args.size, output_ext=args.output_ext)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

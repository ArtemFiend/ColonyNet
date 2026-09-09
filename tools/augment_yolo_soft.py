from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SOFT_SUFFIX_RE = re.compile(r"__soft\d{2}$")


@dataclass
class Stats:
    source_pairs: int = 0
    requested_new_pairs: int = 0
    generated_pairs: int = 0
    skipped_existing: int = 0
    skipped_unreadable: int = 0
    write_failures: int = 0


def read_image_any(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_any(path: Path, image_bgr: np.ndarray) -> bool:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(ext, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        ok, encoded = cv2.imencode(ext, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array([((x / 255.0) ** gamma) * 255.0 for x in range(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def mild_color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.astype(np.float32)
    alpha = float(rng.uniform(0.95, 1.07))
    beta = float(rng.uniform(-8.0, 8.0))
    out = np.clip(out * alpha + beta, 0, 255).astype(np.uint8)
    out = gamma_correct(out, gamma=float(rng.uniform(0.93, 1.07)))

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(0.93, 1.08))
    hsv[..., 2] *= float(rng.uniform(0.95, 1.05))
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def mild_noise_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma_noise = float(rng.uniform(2.0, 5.0))
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma_noise, size=image.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    k = 3 if float(rng.random()) < 0.75 else 5
    sigma_blur = float(rng.uniform(0.2, 0.9))
    return cv2.GaussianBlur(noisy, (k, k), sigmaX=sigma_blur)


def mild_clahe(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip_limit = float(rng.uniform(1.2, 1.8))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    return mild_color_jitter(out, rng)


def mild_sharpen(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=float(rng.uniform(0.4, 0.8)))
    out = cv2.addWeighted(image, 1.12, blur, -0.12, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def soft_augment(image: np.ndarray, variant_idx: int, rng: np.random.Generator) -> np.ndarray:
    mode = (variant_idx - 1) % 4
    if mode == 0:
        return mild_color_jitter(image, rng)
    if mode == 1:
        return mild_noise_blur(image, rng)
    if mode == 2:
        return mild_clahe(image, rng)
    return mild_sharpen(mild_color_jitter(image, rng), rng)


def collect_pairs(split_images_dir: Path, split_labels_dir: Path, include_soft_source: bool) -> list[tuple[str, Path, Path]]:
    images_by_stem: dict[str, Path] = {}
    for p in sorted(split_images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem not in images_by_stem:
            images_by_stem[p.stem] = p

    labels_by_stem: dict[str, Path] = {}
    for p in sorted(split_labels_dir.glob("*.txt")):
        if p.is_file() and p.stem not in labels_by_stem:
            labels_by_stem[p.stem] = p

    stems = sorted(set(images_by_stem) & set(labels_by_stem))
    if not include_soft_source:
        stems = [s for s in stems if SOFT_SUFFIX_RE.search(s) is None]
    return [(stem, images_by_stem[stem], labels_by_stem[stem]) for stem in stems]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softly augment YOLO dataset splits (photometric only) to expand image count.")
    parser.add_argument("--dataset_root", default="Petri_curcle/split_dataset")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to augment (e.g., train val).")
    parser.add_argument("--multiplier", type=int, default=5, help="Total multiplier including originals (>=2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include_soft_source", action="store_true")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.multiplier) < 2:
        raise ValueError("--multiplier must be >= 2")

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    rng = np.random.default_rng(int(args.seed))
    stats_by_split: dict[str, Stats] = {}
    summary_before: dict[str, int] = {}
    summary_after: dict[str, int] = {}
    new_per_source = int(args.multiplier) - 1

    for split in args.splits:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            raise RuntimeError(f"Missing split folders: {images_dir} and {labels_dir}")

        pairs = collect_pairs(images_dir, labels_dir, include_soft_source=bool(args.include_soft_source))
        split_stats = Stats(
            source_pairs=len(pairs),
            requested_new_pairs=len(pairs) * new_per_source,
        )
        stats_by_split[split] = split_stats
        summary_before[split] = len(pairs)

        for idx, (stem, img_path, lbl_path) in enumerate(pairs, start=1):
            img = read_image_any(img_path)
            if img is None:
                split_stats.skipped_unreadable += new_per_source
                continue

            label_text = lbl_path.read_text(encoding="utf-8")
            for aug_idx in range(1, new_per_source + 1):
                out_stem = f"{stem}__soft{aug_idx:02d}"
                out_img = images_dir / f"{out_stem}{img_path.suffix.lower()}"
                out_lbl = labels_dir / f"{out_stem}.txt"

                if (out_img.exists() or out_lbl.exists()) and not bool(args.overwrite):
                    split_stats.skipped_existing += 1
                    continue

                aug = soft_augment(img, aug_idx, rng)
                ok_img = write_image_any(out_img, aug)
                if not ok_img:
                    split_stats.write_failures += 1
                    if out_img.exists():
                        out_img.unlink()
                    continue
                out_lbl.write_text(label_text, encoding="utf-8")
                split_stats.generated_pairs += 1

            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                print(
                    f"[{split}] [{idx}/{len(pairs)}] generated={split_stats.generated_pairs} "
                    f"skipped_existing={split_stats.skipped_existing} write_failures={split_stats.write_failures}"
                )

        final_images = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        final_labels = {p.stem for p in labels_dir.glob("*.txt") if p.is_file()}
        summary_after[split] = len(final_images & final_labels)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "settings": {
            "splits": list(args.splits),
            "multiplier": int(args.multiplier),
            "seed": int(args.seed),
            "overwrite": bool(args.overwrite),
            "include_soft_source": bool(args.include_soft_source),
            "augmentation": "photometric_only_mild",
        },
        "before_pairs_by_split": summary_before,
        "after_pairs_by_split": summary_after,
        "stats_by_split": {k: asdict(v) for k, v in stats_by_split.items()},
    }
    report_path = root / "soft_aug_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

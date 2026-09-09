from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SOFT_SUFFIX_RE = re.compile(r"__soft\d{2}$")


@dataclass
class AugmentStats:
    source_pairs: int = 0
    requested_new_pairs: int = 0
    generated_pairs: int = 0
    skipped_existing: int = 0
    skipped_unreadable: int = 0
    write_failures: int = 0


def read_image_any(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image_any(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def ensure_color_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def normalize_instance_ids(mask: np.ndarray) -> np.ndarray:
    uniq = np.unique(mask)
    uniq_fg = [int(x) for x in uniq if int(x) != 0]
    out = np.zeros(mask.shape, dtype=np.uint16)
    for i, lab in enumerate(uniq_fg, start=1):
        out[mask == lab] = i
    return out


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


def mild_affine(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.98, 1.02))
    tx = float(rng.uniform(-0.015, 0.015) * w)
    ty = float(rng.uniform(-0.015, 0.015) * h)
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    mat[:, 2] += (tx, ty)

    img2 = cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    msk2 = cv2.warpAffine(
        mask,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img2 = mild_color_jitter(img2, rng)
    return img2, msk2


def mild_noise_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma_noise = float(rng.uniform(2.0, 5.0))
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma_noise, size=image.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    k = 3 if float(rng.random()) < 0.75 else 5
    sigma_blur = float(rng.uniform(0.2, 0.9))
    return cv2.GaussianBlur(noisy, (k, k), sigmaX=sigma_blur)


def mild_clahe_sharpen(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip_limit = float(rng.uniform(1.2, 1.8))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=float(rng.uniform(0.4, 0.8)))
    out = cv2.addWeighted(out, 1.12, blur, -0.12, 0.0)
    return mild_color_jitter(out, rng)


def soft_augment(image: np.ndarray, mask: np.ndarray, variant_idx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mode = (variant_idx - 1) % 4
    if mode == 0:
        return mild_color_jitter(image, rng), mask.copy()
    if mode == 1:
        return mild_affine(image, mask, rng)
    if mode == 2:
        return mild_noise_blur(image, rng), mask.copy()
    return mild_clahe_sharpen(image, rng), mask.copy()


def find_pairs(pool_root: Path, include_soft_source: bool) -> tuple[list[tuple[str, Path, Path]], dict[str, int]]:
    images_dir = pool_root / "images"
    instances_dir = pool_root / "instances"
    if not images_dir.exists() or not instances_dir.exists():
        raise RuntimeError(f"Expected folders not found: {images_dir} and {instances_dir}")

    images_by_stem: dict[str, Path] = {}
    duplicate_images = 0
    for p in sorted(images_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if p.stem in images_by_stem:
            duplicate_images += 1
            continue
        images_by_stem[p.stem] = p

    masks_by_stem: dict[str, Path] = {}
    duplicate_masks = 0
    for p in sorted(instances_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        if p.stem in masks_by_stem:
            duplicate_masks += 1
            continue
        masks_by_stem[p.stem] = p

    matched_stems = sorted(set(images_by_stem) & set(masks_by_stem))
    if not include_soft_source:
        matched_stems = [s for s in matched_stems if SOFT_SUFFIX_RE.search(s) is None]

    pairs = [(stem, images_by_stem[stem], masks_by_stem[stem]) for stem in matched_stems]
    all_stats = {
        "image_stems_total": len(images_by_stem),
        "mask_stems_total": len(masks_by_stem),
        "matched_pairs_total": len(set(images_by_stem) & set(masks_by_stem)),
        "duplicate_image_stems_ignored": duplicate_images,
        "duplicate_mask_stems_ignored": duplicate_masks,
    }
    return pairs, all_stats


def validation_stats(pool_root: Path) -> dict[str, int]:
    images_dir = pool_root / "images"
    instances_dir = pool_root / "instances"
    image_stems = {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    mask_stems = {p.stem for p in instances_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    return {
        "images_total": len(image_stems),
        "instances_total": len(mask_stems),
        "matched_pairs": len(image_stems & mask_stems),
        "only_images": len(image_stems - mask_stems),
        "only_instances": len(mask_stems - image_stems),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand dataset with mild augmentations. By default makes x5 total samples (original + 4 soft variants)."
    )
    parser.add_argument("--pool_root", default="trainable_pool", help="Dataset root with images/ and instances/")
    parser.add_argument("--multiplier", type=int, default=5, help="Total multiplier including originals (>=2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated __softXX files.")
    parser.add_argument(
        "--include_soft_source",
        action="store_true",
        help="Also augment files that already end with __softXX.",
    )
    parser.add_argument("--progress_every", type=int, default=200, help="Progress log period by source pairs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.multiplier) < 2:
        raise ValueError("--multiplier must be >= 2")

    pool_root = Path(args.pool_root).resolve()
    images_dir = pool_root / "images"
    instances_dir = pool_root / "instances"

    pairs, pair_stats = find_pairs(pool_root, include_soft_source=bool(args.include_soft_source))
    if not pairs:
        raise RuntimeError("No matched pairs found for augmentation.")

    stats = AugmentStats(
        source_pairs=len(pairs),
        requested_new_pairs=len(pairs) * (int(args.multiplier) - 1),
    )

    before = validation_stats(pool_root)
    rng = np.random.default_rng(int(args.seed))
    new_per_source = int(args.multiplier) - 1

    for idx, (stem, img_path, msk_path) in enumerate(pairs, start=1):
        img = read_image_any(img_path, cv2.IMREAD_UNCHANGED)
        msk = read_image_any(msk_path, cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            stats.skipped_unreadable += new_per_source
            continue
        img = ensure_color_bgr(img)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = msk.astype(np.uint16)
        msk = normalize_instance_ids(msk)

        for aug_idx in range(1, new_per_source + 1):
            out_stem = f"{stem}__soft{aug_idx:02d}"
            out_img = images_dir / f"{out_stem}{img_path.suffix.lower()}"
            out_msk = instances_dir / f"{out_stem}.png"

            if (out_img.exists() or out_msk.exists()) and not bool(args.overwrite):
                stats.skipped_existing += 1
                continue

            img_aug, msk_aug = soft_augment(img, msk, aug_idx, rng)
            msk_aug = normalize_instance_ids(msk_aug.astype(np.uint16))

            ok_img = write_image_any(out_img, img_aug)
            ok_msk = write_image_any(out_msk, msk_aug)
            if not (ok_img and ok_msk):
                stats.write_failures += 1
                if out_img.exists():
                    out_img.unlink()
                if out_msk.exists():
                    out_msk.unlink()
                continue
            stats.generated_pairs += 1

        if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
            print(
                f"[{idx}/{len(pairs)}] generated={stats.generated_pairs} "
                f"skipped_existing={stats.skipped_existing} write_failures={stats.write_failures}"
            )

    after = validation_stats(pool_root)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_root": str(pool_root),
        "settings": {
            "multiplier": int(args.multiplier),
            "seed": int(args.seed),
            "overwrite": bool(args.overwrite),
            "include_soft_source": bool(args.include_soft_source),
        },
        "pair_discovery": pair_stats,
        "before": before,
        "after": after,
        "stats": asdict(stats),
        "target_total_pairs": before["matched_pairs"] * int(args.multiplier),
        "actual_total_pairs": after["matched_pairs"],
    }
    report_path = pool_root / "soft_aug_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

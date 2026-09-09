from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Stats:
    base_pairs: int = 0
    added_pairs: int = 0
    skipped_pairs: int = 0
    write_failures: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def center_crop_or_pad(image: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h > size:
        top = (h - size) // 2
        image = image[top : top + size, ...]
    elif h < size:
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        if image.ndim == 3:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    h, w = image.shape[:2]
    if w > size:
        left = (w - size) // 2
        image = image[:, left : left + size, ...]
    elif w < size:
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        if image.ndim == 3:
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    if image.shape[0] != size or image.shape[1] != size:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        image = cv2.resize(image, (size, size), interpolation=interp)
    return image


def normalize_instance_ids(mask: np.ndarray) -> np.ndarray:
    # Keep background as 0 and remap all non-zero labels to contiguous 1..N.
    uniq = np.unique(mask)
    uniq_fg = [int(x) for x in uniq if int(x) != 0]
    out = np.zeros(mask.shape, dtype=np.uint16)
    for i, lab in enumerate(uniq_fg, start=1):
        out[mask == lab] = i
    return out


def collect_pairs(root: Path) -> list[tuple[str, str, str, Path, Path]]:
    pairs: list[tuple[str, str, str, Path, Path]] = []
    for modality in ("brightfield_dataset", "fluorescence_dataset"):
        mod_root = root / modality
        if not mod_root.exists():
            continue
        for split_name, img_sub in (
            ("train_full", ("train", "full_images", "brightfield" if "brightfield" in modality else "fluorescence")),
            ("train_patches", ("train", "patches", "brightfield" if "brightfield" in modality else "fluorescence")),
            ("test", ("test", "brightfield" if "brightfield" in modality else "fluorescence")),
        ):
            img_dir = mod_root.joinpath(*img_sub)
            msk_dir = mod_root.joinpath(*img_sub[:-1], "masks")
            if not img_dir.exists() or not msk_dir.exists():
                continue
            for img_path in sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS):
                msk_path = msk_dir / f"{img_path.stem}.tif"
                if not msk_path.exists():
                    msk_path = msk_dir / f"{img_path.stem}.tiff"
                if not msk_path.exists():
                    continue
                modality_short = "bf" if "brightfield" in modality else "fl"
                pairs.append((modality_short, split_name, img_path.stem, img_path, msk_path))
    return pairs


def augment_pair(img: np.ndarray, msk: np.ndarray, aug_idx: int, n_aug: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # a00 is identity; the rest are deterministic geometric transforms with optional tiny affine jitter.
    if aug_idx == 0:
        img2, msk2 = img.copy(), msk.copy()
    elif aug_idx == 1:
        img2, msk2 = cv2.flip(img, 1), cv2.flip(msk, 1)
    elif aug_idx == 2:
        img2, msk2 = cv2.flip(img, 0), cv2.flip(msk, 0)
    elif aug_idx == 3:
        img2, msk2 = cv2.flip(img, -1), cv2.flip(msk, -1)
    elif aug_idx == 4:
        img2, msk2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
    elif aug_idx == 5:
        img2, msk2 = cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(msk, cv2.ROTATE_180)
    elif aug_idx == 6:
        img2, msk2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(msk, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_idx == 7:
        img2, msk2 = cv2.transpose(img), cv2.transpose(msk)
    else:
        h, w = img.shape[:2]
        angle = float(rng.uniform(-10.0, 10.0))
        scale = float(rng.uniform(0.95, 1.05))
        tx = float(rng.uniform(-0.03, 0.03) * w)
        ty = float(rng.uniform(-0.03, 0.03) * h)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[:, 2] += (tx, ty)
        img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        msk2 = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, msk2


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert DeepBacs Staph Aureus dataset to trainable_pool with 512 resize and augmentation.")
    ap.add_argument("--src_root", default="DeepBacs_Data_Segmentation_Staph_Aureus_dataset")
    ap.add_argument("--pool_root", default="trainable_pool")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_aug", type=int, default=10, help="Number of variants per source sample (includes identity a00).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean_prefix", action="store_true", help="Remove previously added data_deepbacs_staph__ files before import.")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    pool_root = Path(args.pool_root).resolve()
    out_images = pool_root / "images"
    out_instances = pool_root / "instances"
    ensure_dir(out_images)
    ensure_dir(out_instances)

    if args.clean_prefix:
        for p in out_images.glob("data_deepbacs_staph__*.png"):
            p.unlink()
        for p in out_instances.glob("data_deepbacs_staph__*.png"):
            p.unlink()

    pairs = collect_pairs(src_root)
    stats = Stats(base_pairs=len(pairs))
    rng = np.random.default_rng(args.seed)
    id_map: list[dict[str, str]] = []

    for idx, (modality_short, split_name, stem, img_path, msk_path) in enumerate(pairs, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            stats.skipped_pairs += 1
            continue
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = msk.astype(np.uint16)
        msk = normalize_instance_ids(msk)

        img = center_crop_or_pad(img, size=args.size, is_mask=False)
        msk = center_crop_or_pad(msk, size=args.size, is_mask=True).astype(np.uint16)
        base_short = f"data_deepbacs_staph__{modality_short}__{split_name}__{idx:04d}"
        id_map.append(
            {
                "base_short": base_short,
                "modality": modality_short,
                "split": split_name,
                "source_image": str(img_path),
                "source_mask": str(msk_path),
                "source_stem": stem,
            }
        )

        for i in range(max(1, int(args.n_aug))):
            out_stem = f"{base_short}__a{i:02d}"
            out_img = out_images / f"{out_stem}.png"
            out_msk = out_instances / f"{out_stem}.png"
            img_i, msk_i = augment_pair(img, msk, i, args.n_aug, rng)
            msk_i = normalize_instance_ids(msk_i.astype(np.uint16))
            ok_img = cv2.imwrite(str(out_img), img_i)
            ok_msk = cv2.imwrite(str(out_msk), msk_i)
            if not (ok_img and ok_msk):
                stats.write_failures += 1
                if out_img.exists():
                    out_img.unlink()
                if out_msk.exists():
                    out_msk.unlink()
                continue
            stats.added_pairs += 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src_root),
        "pool_root": str(pool_root),
        "settings": {"size": int(args.size), "n_aug": int(args.n_aug), "seed": int(args.seed)},
        "base_pairs": int(stats.base_pairs),
        "added_pairs": int(stats.added_pairs),
        "skipped_pairs": int(stats.skipped_pairs),
        "write_failures": int(stats.write_failures),
    }
    report_path = pool_root / "deepbacs_staph_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    map_path = pool_root / "deepbacs_staph_id_map.json"
    map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved id map: {map_path}")


if __name__ == "__main__":
    main()

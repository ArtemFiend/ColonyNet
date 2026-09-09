from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class MergeStats:
    dataset: str
    added: int = 0
    skipped: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_prefix(path_str: str) -> str:
    x = path_str.replace("\\", "_").replace("/", "_").replace(":", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_").lower()


def image_files(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def center_crop_or_pad(image: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h > size:
        top = (h - size) // 2
        image = image[top: top + size, ...]
    elif h < size:
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        border_mode = cv2.BORDER_CONSTANT
        if image.ndim == 3:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, border_mode, value=(0, 0, 0))
        else:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, border_mode, value=0)

    h, w = image.shape[:2]
    if w > size:
        left = (w - size) // 2
        image = image[:, left: left + size, ...]
    elif w < size:
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        border_mode = cv2.BORDER_CONSTANT
        if image.ndim == 3:
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, border_mode, value=(0, 0, 0))
        else:
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, border_mode, value=0)

    if image.shape[0] != size or image.shape[1] != size:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        image = cv2.resize(image, (size, size), interpolation=interp)
    return image


def copy_pair(img_path: Path, inst_path: Path, out_images: Path, out_instances: Path, out_stem: str) -> None:
    out_img = out_images / f"{out_stem}{img_path.suffix.lower()}"
    out_inst = out_instances / f"{out_stem}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
    if inst is None:
        raise RuntimeError(f"Failed to read instance mask: {inst_path}")
    img = center_crop_or_pad(img, size=512, is_mask=False)
    inst = center_crop_or_pad(inst, size=512, is_mask=True)
    cv2.imwrite(str(out_img), img)
    cv2.imwrite(str(out_inst), inst)


def find_pair_datasets(root: Path, exclude_roots: set[Path] | None = None) -> list[Path]:
    exclude_roots = exclude_roots or set()
    found: list[Path] = []
    for images_dir in root.rglob("images"):
        if not images_dir.is_dir():
            continue
        parent = images_dir.parent
        if any(str(parent).startswith(str(ex)) for ex in exclude_roots):
            continue
        if (parent / "instances").exists():
            found.append(parent)
    return sorted(found)


def merge_existing_pairs(dataset_dir: Path, source_root: Path, out_images: Path, out_instances: Path) -> MergeStats:
    rel = dataset_dir.relative_to(source_root)
    dataset_key = f"{source_root.name}/{rel.as_posix()}"
    prefix = sanitize_prefix(dataset_key)
    stats = MergeStats(dataset=dataset_key)

    images_dir = dataset_dir / "images"
    instances_dir = dataset_dir / "instances"

    for img_path in image_files(images_dir):
        inst_path = instances_dir / f"{img_path.stem}.png"
        if not inst_path.exists():
            stats.skipped += 1
            continue
        out_stem = f"{prefix}__{img_path.stem}"
        copy_pair(img_path, inst_path, out_images, out_instances, out_stem)
        stats.added += 1

    return stats


def convert_stage1_train(stage1_root: Path, out_images: Path, out_instances: Path) -> MergeStats:
    stats = MergeStats(dataset="stage1_train/converted")
    if not stage1_root.exists():
        stats.skipped += 1
        return stats

    sample_dirs = sorted([d for d in stage1_root.iterdir() if d.is_dir()])
    for sample_dir in sample_dirs:
        img_dir = sample_dir / "images"
        msk_dir = sample_dir / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            stats.skipped += 1
            continue

        imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if len(imgs) != 1:
            stats.skipped += 1
            continue

        img_path = imgs[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            stats.skipped += 1
            continue

        mask_files = sorted(msk_dir.glob("*.png"))
        if len(mask_files) == 0:
            stats.skipped += 1
            continue

        h, w = img.shape[:2]
        inst = np.zeros((h, w), dtype=np.uint16)
        cur = 1
        for mf in mask_files:
            m = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[:, :, 0]
            fg = (m > 0)
            if not fg.any():
                continue
            inst[fg] = cur
            cur += 1
            if cur >= 65535:
                break

        out_stem = f"stage1_train__{sample_dir.name}"
        out_img = out_images / f"{out_stem}.png"
        out_inst = out_instances / f"{out_stem}.png"
        img = center_crop_or_pad(img, size=512, is_mask=False)
        inst = center_crop_or_pad(inst, size=512, is_mask=True)
        cv2.imwrite(str(out_img), img)
        cv2.imwrite(str(out_inst), inst)
        stats.added += 1

    return stats


def validate_pairs(out_images: Path, out_instances: Path) -> dict[str, int]:
    image_stems = {p.stem for p in out_images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    instance_stems = {p.stem for p in out_instances.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
    return {
        "images_total": len(image_stems),
        "instances_total": len(instance_stems),
        "matched_pairs": len(image_stems & instance_stems),
        "only_images": len(image_stems - instance_stems),
        "only_instances": len(instance_stems - image_stems),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build one merged trainable folder from data + all_datasets + stage1_train.")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--all_root", default="all_datasets")
    ap.add_argument("--stage1_root", default="stage1_train")
    ap.add_argument("--out_root", default="data/trainable_pool")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    all_root = Path(args.all_root).resolve()
    stage1_root = Path(args.stage1_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_images = out_root / "images"
    out_instances = out_root / "instances"

    ensure_dir(out_images)
    ensure_dir(out_instances)

    existing = find_pair_datasets(data_root, exclude_roots={out_root})
    existing += find_pair_datasets(all_root, exclude_roots={out_root})

    stats_all: list[MergeStats] = []
    for ds in existing:
        src_root = data_root if str(ds).startswith(str(data_root)) else all_root
        stats_all.append(merge_existing_pairs(ds, src_root, out_images, out_instances))

    stats_all.append(convert_stage1_train(stage1_root, out_images, out_instances))
    validation = validate_pairs(out_images, out_instances)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out_root),
        "sources": [{"dataset": s.dataset, "added": s.added, "skipped": s.skipped} for s in stats_all],
        "totals": {
            "added_all": int(sum(s.added for s in stats_all)),
            "skipped_all": int(sum(s.skipped for s in stats_all)),
        },
        "validation": validation,
    }

    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

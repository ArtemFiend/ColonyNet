from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


RAW_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
ANY_IMG_EXTS = RAW_IMG_EXTS | {".tif", ".tiff"}
COORD_RE = re.compile(r"\((\d+),\s*(\d+)\)")
IMG_ID_RE = re.compile(r"(IMG_\d+)")


@dataclass
class ConvertStats:
    base_tif: int = 0
    colored_tif: int = 0
    coordinates_txt: int = 0
    skipped: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_image_any(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def write_png_any(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def to_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        fg = np.any(mask > 0, axis=2)
    else:
        fg = mask > 0
    return fg.astype(np.uint8)


def binary_to_instances(binary: np.ndarray) -> np.ndarray:
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    mx = int(labels.max())
    if mx > np.iinfo(np.uint16).max:
        raise RuntimeError(f"Too many connected components: {mx}")
    return labels.astype(np.uint16)


def sanitize_id(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return out or "sample"


def unique_id(base_id: str, used: set[str]) -> str:
    if base_id not in used:
        used.add(base_id)
        return base_id
    i = 2
    while True:
        cand = f"{base_id}__{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def validation_stats(images_dir: Path, instances_dir: Path) -> dict[str, int]:
    image_stems = {
        p.stem for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ANY_IMG_EXTS
    }
    inst_stems = {
        p.stem for p in instances_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }
    return {
        "images_total": len(image_stems),
        "instances_total": len(inst_stems),
        "matched_pairs": len(image_stems & inst_stems),
        "only_images": len(image_stems - inst_stems),
        "only_instances": len(inst_stems - image_stems),
    }


def auto_find_raw_root(base_dir: Path) -> Path:
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]

    for d in dirs:
        child_names = {c.name.lower() for c in d.iterdir() if c.is_dir()}
        if any("s. aureus" in x for x in child_names) and any("coli" in x for x in child_names):
            return d

    best: tuple[int, Path] | None = None
    for d in dirs:
        cnt = len(list(d.rglob("*_coordinates.txt")))
        if cnt > 0 and (best is None or cnt > best[0]):
            best = (cnt, d)

    if best is not None:
        return best[1]
    raise RuntimeError("Could not auto-detect raw cups dataset root. Pass --raw_root explicitly.")


def collect_raw_images(raw_root: Path) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in RAW_IMG_EXTS:
            continue
        name_l = p.name.lower()
        if "colored" in name_l or "marked" in name_l:
            continue
        by_stem[p.stem].append(p)
    return by_stem


def pick_image(stem: str, images_by_stem: dict[str, list[Path]]) -> Path | None:
    cand = images_by_stem.get(stem)
    if not cand:
        return None
    # Prefer shortest path depth, then shortest full path.
    return sorted(cand, key=lambda p: (len(p.parts), len(str(p))))[0]


def parse_coordinates_to_binary(txt_path: Path, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = COORD_RE.search(line)
            if m is None:
                continue
            x = int(m.group(1))
            y = int(m.group(2))
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = 1
    return mask


def save_pair(
    out_images: Path,
    out_instances: Path,
    out_id: str,
    src_img: Path,
    inst: np.ndarray,
) -> None:
    out_img = out_images / f"{out_id}{src_img.suffix.lower()}"
    out_inst = out_instances / f"{out_id}.png"
    shutil.copy2(src_img, out_img)
    write_png_any(out_inst, inst)


def convert_cups(raw_root: Path, out_root: Path) -> dict[str, object]:
    out_images = out_root / "images"
    out_instances = out_root / "instances"
    reset_dir(out_images)
    reset_dir(out_instances)

    images_by_stem = collect_raw_images(raw_root)
    used_ids: set[str] = set()
    converted_img_stems: set[str] = set()
    stats = ConvertStats()
    skips: list[str] = []

    def add_from_binary(mask_bin: np.ndarray, img_stem: str, img_path: Path, source_kind: str) -> None:
        base_id = sanitize_id(img_stem)
        out_id = unique_id(base_id, used_ids)
        inst = binary_to_instances(mask_bin)
        save_pair(out_images, out_instances, out_id, img_path, inst)
        converted_img_stems.add(img_stem)
        if source_kind == "base_tif":
            stats.base_tif += 1
        elif source_kind == "colored_tif":
            stats.colored_tif += 1
        elif source_kind == "coordinates_txt":
            stats.coordinates_txt += 1

    # 1) Base TIFF masks (binary) -> connected components
    base_masks: list[Path] = []
    for ext in ("*.tif", "*.tiff"):
        base_masks.extend(raw_root.rglob(ext))
    for mask_path in sorted(base_masks):
        name_l = mask_path.name.lower()
        if name_l.endswith("_colored.tif") or name_l.endswith("_colored.tiff"):
            continue
        if name_l.endswith("_marked.tif") or name_l.endswith("_marked.tiff"):
            continue

        img_stem = mask_path.stem
        if img_stem in converted_img_stems:
            continue
        img_path = pick_image(img_stem, images_by_stem)
        if img_path is None:
            stats.skipped += 1
            skips.append(f"base_tif:no_image:{mask_path}")
            continue

        mask = read_image_any(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            stats.skipped += 1
            skips.append(f"base_tif:read_fail:{mask_path}")
            continue

        add_from_binary(to_binary(mask), img_stem, img_path, "base_tif")

    # 2) Colored TIFF fallback (for images without base masks)
    colored_masks: list[Path] = []
    for ext in ("*_colored.tif", "*_colored.tiff"):
        colored_masks.extend(raw_root.rglob(ext))
    for mask_path in sorted(colored_masks):
        stem_l = mask_path.stem.lower()
        if stem_l.endswith("_non_black_colored"):
            img_stem = mask_path.stem[: -len("_non_black_colored")]
        elif stem_l.endswith("_colored"):
            img_stem = mask_path.stem[: -len("_colored")]
        else:
            continue
        if img_stem in converted_img_stems:
            continue

        img_path = pick_image(img_stem, images_by_stem)
        if img_path is None:
            stats.skipped += 1
            skips.append(f"colored_tif:no_image:{mask_path}")
            continue

        mask = read_image_any(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            stats.skipped += 1
            skips.append(f"colored_tif:read_fail:{mask_path}")
            continue

        add_from_binary(to_binary(mask), img_stem, img_path, "colored_tif")

    # 3) Coordinates fallback (for images without any TIFF mask)
    for txt_path in sorted(raw_root.rglob("*_coordinates.txt")):
        m = IMG_ID_RE.search(txt_path.name)
        if m is None:
            continue
        img_stem = m.group(1)
        if img_stem in converted_img_stems:
            continue

        img_path = pick_image(img_stem, images_by_stem)
        if img_path is None:
            stats.skipped += 1
            skips.append(f"coordinates:no_image:{txt_path}")
            continue

        img = read_image_any(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            stats.skipped += 1
            skips.append(f"coordinates:image_read_fail:{img_path}")
            continue
        h, w = img.shape[:2]
        mask_bin = parse_coordinates_to_binary(txt_path, h, w)
        add_from_binary(mask_bin, img_stem, img_path, "coordinates_txt")

    validation = validation_stats(out_images, out_instances)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(raw_root.resolve()),
        "output_root": str(out_root.resolve()),
        "target_format": "images/* + instances/*.png (0=background, 1..N=instance ids)",
        "mask_building": {
            "base_tif": "binary mask (>0) converted to instance ids with connected components",
            "colored_tif_fallback": "non-black mask converted to instance ids with connected components",
            "coordinates_txt_fallback": "parsed (x,y) pixels converted to instance ids with connected components",
        },
        "converted": {
            "base_tif": stats.base_tif,
            "colored_tif": stats.colored_tif,
            "coordinates_txt": stats.coordinates_txt,
            "total": stats.base_tif + stats.colored_tif + stats.coordinates_txt,
        },
        "skipped": {
            "count": stats.skipped,
            "examples": skips[:100],
        },
        "validation": validation,
    }
    (out_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def merge_into_pool(out_root: Path, pool_root: Path) -> dict[str, object]:
    src_images = out_root / "images"
    src_instances = out_root / "instances"
    dst_images = pool_root / "images"
    dst_instances = pool_root / "instances"
    ensure_dir(dst_images)
    ensure_dir(dst_instances)

    copied = 0
    skipped = 0
    for img_path in sorted(src_images.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in ANY_IMG_EXTS:
            continue
        stem = img_path.stem
        inst_path = src_instances / f"{stem}.png"
        if not inst_path.exists():
            skipped += 1
            continue
        pool_stem = f"data_cups_dataset__{stem}"
        out_img = dst_images / f"{pool_stem}{img_path.suffix.lower()}"
        out_inst = dst_instances / f"{pool_stem}.png"
        shutil.copy2(img_path, out_img)
        shutil.copy2(inst_path, out_inst)
        copied += 1

    validation = validation_stats(dst_images, dst_instances)
    report_path = pool_root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}

    sources = report.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    cups_count = len(
        [
            p for p in dst_images.iterdir()
            if p.is_file() and p.suffix.lower() in ANY_IMG_EXTS and p.stem.startswith("data_cups_dataset__")
        ]
    )

    updated = False
    for s in sources:
        if isinstance(s, dict) and s.get("dataset") == "data/cups_dataset":
            s["added"] = int(cups_count)
            s["skipped"] = 0
            updated = True
            break
    if not updated:
        sources.append({"dataset": "data/cups_dataset", "added": int(cups_count), "skipped": 0})

    total_added = int(sum(int(s.get("added", 0)) for s in sources if isinstance(s, dict)))
    total_skipped = int(sum(int(s.get("skipped", 0)) for s in sources if isinstance(s, dict)))

    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["output_root"] = str(pool_root.resolve())
    report["sources"] = sources
    report["totals"] = {"added_all": total_added, "skipped_all": total_skipped}
    report["validation"] = validation
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "copied_to_pool": copied,
        "skipped_missing_pairs": skipped,
        "pool_cups_pairs_total": int(cups_count),
        "pool_validation": validation,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare cups dataset to ColonyNet image+instances format and merge into trainable_pool.")
    ap.add_argument("--raw_root", default=None, help="Raw cups dataset root folder")
    ap.add_argument("--out_root", default="data/cups_dataset", help="Output folder with images/ and instances/")
    ap.add_argument("--pool_root", default="trainable_pool", help="Trainable pool folder with images/ and instances/")
    ap.add_argument("--skip_pool", action="store_true", help="Only prepare dataset, do not copy into trainable_pool")
    args = ap.parse_args()

    cwd = Path.cwd()
    raw_root = Path(args.raw_root).resolve() if args.raw_root else auto_find_raw_root(cwd)
    out_root = Path(args.out_root).resolve()
    pool_root = Path(args.pool_root).resolve()

    convert_report = convert_cups(raw_root, out_root)
    final = {"cups_dataset": convert_report}

    if not args.skip_pool:
        final["trainable_pool"] = merge_into_pool(out_root, pool_root)

    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

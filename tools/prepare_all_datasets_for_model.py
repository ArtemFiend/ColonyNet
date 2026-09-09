from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ConvertResult:
    dataset: str
    converted_count: int
    converted_ids: list[str]
    skipped_count: int
    skipped_items: list[str]
    notes: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_bgr_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))

    if img.ndim == 2:
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return img
    return img[:, :, :3]


def clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    nw = max(0, x2 - x1)
    nh = max(0, y2 - y1)
    return x1, y1, nw, nh


def draw_bbox_ellipse(mask: np.ndarray, x: int, y: int, w: int, h: int, obj_id: int) -> None:
    cx = x + w // 2
    cy = y + h // 2
    ax1 = max(1, w // 2)
    ax2 = max(1, h // 2)
    cv2.ellipse(mask, (cx, cy), (ax1, ax2), 0, 0, 360, int(obj_id), thickness=-1)


def write_pair(out_images: Path, out_instances: Path, sample_id: str, img: np.ndarray, img_ext: str, inst: np.ndarray) -> None:
    out_img = out_images / f"{sample_id}{img_ext}"
    out_inst = out_instances / f"{sample_id}.png"
    cv2.imwrite(str(out_img), img)
    cv2.imwrite(str(out_inst), inst)


def convert_top_level_coco(all_root: Path, out_images: Path, out_instances: Path) -> ConvertResult:
    coco_path = all_root / "annot_COCO.json"
    if not coco_path.exists():
        return ConvertResult("top_level_coco_bbox", 0, [], 1, [str(coco_path)], "annot_COCO.json not found")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_map[int(ann["image_id"])].append(ann)

    converted: list[str] = []
    skipped: list[str] = []

    for im in coco.get("images", []):
        image_id = int(im["id"])
        file_name = im["file_name"]
        src = all_root / file_name
        if not src.exists():
            skipped.append(f"missing_image:{file_name}")
            continue

        img = to_bgr_u8(src)
        h, w = img.shape[:2]
        inst = np.zeros((h, w), dtype=np.uint16)

        obj_id = 1
        for ann in ann_map.get(image_id, []):
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, bw, bh = (int(round(float(v))) for v in bbox)
            x, y, bw, bh = clip_box(x, y, bw, bh, w, h)
            if bw <= 0 or bh <= 0:
                continue
            draw_bbox_ellipse(inst, x, y, bw, bh, obj_id)
            obj_id += 1
            if obj_id >= 65535:
                break

        sample_id = f"topcoco_{Path(file_name).stem}"
        write_pair(out_images, out_instances, sample_id, img, ".jpg", inst)
        converted.append(sample_id)

    return ConvertResult("top_level_coco_bbox", len(converted), converted, len(skipped), skipped, "bbox->ellipse instances")


def convert_agar_representative(all_root: Path, out_images: Path, out_instances: Path) -> ConvertResult:
    agar_root = all_root / "AGAR_representative"
    if not agar_root.exists():
        return ConvertResult("AGAR_representative", 0, [], 1, [str(agar_root)], "folder not found")

    converted: list[str] = []
    skipped: list[str] = []

    jpgs = sorted(list(agar_root.rglob("*.jpg")) + list(agar_root.rglob("*.jpeg")))
    for jpg in jpgs:
        js = jpg.with_suffix(".json")
        if not js.exists():
            skipped.append(f"missing_json:{js}")
            continue

        try:
            with js.open("r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            skipped.append(f"bad_json:{js}:{e}")
            continue

        try:
            img = to_bgr_u8(jpg)
        except Exception as e:
            skipped.append(f"bad_image:{jpg}:{e}")
            continue

        h, w = img.shape[:2]
        inst = np.zeros((h, w), dtype=np.uint16)
        obj_id = 1
        for obj in ann.get("labels", []):
            x = int(obj.get("x", 0))
            y = int(obj.get("y", 0))
            bw = int(obj.get("width", 0))
            bh = int(obj.get("height", 0))
            if bw <= 0 or bh <= 0:
                continue
            x, y, bw, bh = clip_box(x, y, bw, bh, w, h)
            if bw <= 0 or bh <= 0:
                continue
            draw_bbox_ellipse(inst, x, y, bw, bh, obj_id)
            obj_id += 1
            if obj_id >= 65535:
                break

        rel = jpg.relative_to(agar_root)
        tags = [p.lower().replace("-", "") for p in rel.parts[:-1]]
        sample_id = "agarrep_" + "_".join(tags + [jpg.stem])
        write_pair(out_images, out_instances, sample_id, img, ".jpg", inst)
        converted.append(sample_id)

    return ConvertResult("AGAR_representative", len(converted), converted, len(skipped), skipped, "bbox json->ellipse instances")


def decode_coco_segmentation(segmentation: Any, h: int, w: int) -> np.ndarray:
    if isinstance(segmentation, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    if isinstance(segmentation, dict) and mask_utils is not None:
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, h, w)
            decoded = mask_utils.decode(rle)
        else:
            decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2).astype(np.uint8)
        else:
            decoded = (decoded > 0).astype(np.uint8)
        if decoded.shape != (h, w):
            if decoded.shape == (w, h):
                decoded = decoded.T
            else:
                raise ValueError(f"Bad decoded shape: {decoded.shape}, expected {(h, w)}")
        return decoded

    return np.zeros((h, w), dtype=np.uint8)


def convert_livecell(all_root: Path, out_images: Path, out_instances: Path) -> ConvertResult:
    live_root = all_root / "livecell_train_val"
    ann_path = live_root / "livecell_annotations" / "3_train25percent.json"
    images_dir = live_root / "livecell_train_val_images"
    if not ann_path.exists() or not images_dir.exists():
        return ConvertResult(
            "livecell_train_val",
            0,
            [],
            1,
            [f"missing:{ann_path}", f"missing:{images_dir}"],
            "missing livecell inputs",
        )

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_map[int(ann["image_id"])].append(ann)

    converted: list[str] = []
    skipped: list[str] = []

    for im in coco.get("images", []):
        image_id = int(im["id"])
        file_name = im["file_name"]
        src = images_dir / file_name
        if not src.exists():
            src = images_dir / Path(file_name).name
        if not src.exists():
            skipped.append(f"missing_image:{file_name}")
            continue

        try:
            img = to_bgr_u8(src)
        except Exception as e:
            skipped.append(f"bad_image:{src}:{e}")
            continue

        h, w = img.shape[:2]
        inst = np.zeros((h, w), dtype=np.uint16)
        obj_id = 1
        for ann in ann_map.get(image_id, []):
            seg = ann.get("segmentation")
            m = decode_coco_segmentation(seg, h, w)
            if m.sum() == 0:
                continue
            inst[m > 0] = obj_id
            obj_id += 1
            if obj_id >= 65535:
                break

        sample_id = f"livecell_{Path(file_name).stem}"
        write_pair(out_images, out_instances, sample_id, img, ".png", inst)
        converted.append(sample_id)

    note = "COCO segmentation->instances"
    if mask_utils is None:
        note += "; pycocotools missing (RLE would be skipped)"
    return ConvertResult("livecell_train_val", len(converted), converted, len(skipped), skipped, note)


def convert_bbbc005(all_root: Path, out_images: Path, out_instances: Path) -> ConvertResult:
    img_dir = all_root / "BBBC005_v1_images"
    gt_dir = all_root / "BBBC005_v1_ground_truth" / "BBBC005_v1_ground_truth"
    if not img_dir.exists() or not gt_dir.exists():
        return ConvertResult(
            "BBBC005_v1",
            0,
            [],
            1,
            [f"missing:{img_dir}", f"missing:{gt_dir}"],
            "missing BBBC005 inputs",
        )

    converted: list[str] = []
    skipped: list[str] = []

    gt_files = sorted([p for p in gt_dir.glob("*.TIF") if not p.name.startswith(".")])
    for gt in gt_files:
        src = img_dir / gt.name
        if not src.exists():
            skipped.append(f"missing_image:{src}")
            continue

        try:
            img = to_bgr_u8(src)
        except Exception as e:
            skipped.append(f"bad_image:{src}:{e}")
            continue

        gt_raw = cv2.imread(str(gt), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            skipped.append(f"bad_gt:{gt}")
            continue
        if gt_raw.ndim == 3:
            gt_raw = gt_raw[:, :, 0]

        fg = (gt_raw > 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(fg, connectivity=8)
        labels = labels.astype(np.uint16)

        sample_id = f"bbbc005_{gt.stem.lower()}"
        write_pair(out_images, out_instances, sample_id, img, ".png", labels)
        converted.append(sample_id)

    return ConvertResult(
        "BBBC005_v1",
        len(converted),
        converted,
        len(skipped),
        skipped,
        "binary GT->connected-components instances",
    )


def find_immediately_suitable(all_root: Path) -> list[str]:
    suitable: list[str] = []
    images_dirs = sorted([p for p in all_root.rglob("*") if p.is_dir() and p.name.lower() == "images"])
    for images_dir in images_dirs:
        parent = images_dir.parent
        inst_dir = parent / "instances"
        if not inst_dir.exists():
            continue
        for img in sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]):
            inst = inst_dir / f"{img.stem}.png"
            if not inst.exists():
                continue
            suitable.append(str(img))
    return suitable


def validate_output_pairs(out_images: Path, out_instances: Path) -> dict[str, Any]:
    image_stems = {p.stem for p in out_images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    inst_stems = {p.stem for p in out_instances.iterdir() if p.is_file() and p.suffix.lower() == ".png"}

    only_images = sorted(image_stems - inst_stems)
    only_instances = sorted(inst_stems - image_stems)
    matched = sorted(image_stems & inst_stems)
    return {
        "images_total": len(image_stems),
        "instances_total": len(inst_stems),
        "matched_pairs": len(matched),
        "only_images_count": len(only_images),
        "only_instances_count": len(only_instances),
        "only_images_examples": only_images[:20],
        "only_instances_examples": only_instances[:20],
    }


def delete_unusable(all_root: Path) -> list[str]:
    deleted: list[str] = []
    sartorius = all_root / "sartorius"
    if sartorius.exists():
        shutil.rmtree(sartorius)
        deleted.append(str(sartorius))
    return deleted


def save_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all_datasets into image+instances format for ColonyNet.")
    parser.add_argument("--all_root", default="all_datasets", help="Input root with raw datasets")
    parser.add_argument("--out_root", default="data/unified_from_all_datasets", help="Output root with images/instances")
    parser.add_argument("--delete_unusable", action="store_true", help="Delete unusable folders after conversion")
    args = parser.parse_args()

    all_root = Path(args.all_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_images = out_root / "images"
    out_instances = out_root / "instances"
    reports_dir = out_root / "reports"

    ensure_dir(out_images)
    ensure_dir(out_instances)
    ensure_dir(reports_dir)

    suitable = find_immediately_suitable(all_root)

    results = [
        convert_top_level_coco(all_root, out_images, out_instances),
        convert_agar_representative(all_root, out_images, out_instances),
        convert_livecell(all_root, out_images, out_instances),
        convert_bbbc005(all_root, out_images, out_instances),
    ]

    for res in results:
        save_list(reports_dir / f"reformatted_{res.dataset}.txt", res.converted_ids)
        save_list(reports_dir / f"skipped_{res.dataset}.txt", res.skipped_items)

    deleted = delete_unusable(all_root) if args.delete_unusable else []
    validation = validate_output_pairs(out_images, out_instances)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_format": "images/* + instances/*.png with integer instance ids (0 background, 1..N objects)",
        "suitable_immediately": {
            "count": len(suitable),
            "items": suitable,
            "note": "Scanned for existing images/ + instances/ folder pairs inside all_datasets.",
        },
        "reformatted": [
            {
                "dataset": r.dataset,
                "converted_count": r.converted_count,
                "skipped_count": r.skipped_count,
                "notes": r.notes,
                "converted_list_file": str((reports_dir / f"reformatted_{r.dataset}.txt").name),
                "skipped_list_file": str((reports_dir / f"skipped_{r.dataset}.txt").name),
            }
            for r in results
        ],
        "deleted": {"count": len(deleted), "paths": deleted},
        "output_root": str(out_root),
        "validation": validation,
    }

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()

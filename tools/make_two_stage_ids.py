from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def collect_pair_ids(images_dir: Path, instances_dir: Path) -> list[str]:
    image_ids = {
        os.path.splitext(p.name)[0]
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }
    instance_ids = {
        os.path.splitext(p.name)[0]
        for p in instances_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }
    return sorted(image_ids & instance_ids)


def split_ids(ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not ids:
        return [], []
    rng = random.Random(seed)
    x = ids[:]
    rng.shuffle(x)
    n_val = max(1, int(len(x) * val_fraction))
    return x[n_val:], x[:n_val]


def save_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 2-stage train/val ID lists for trainable_pool.")
    ap.add_argument("--images_dir", default="trainable_pool/images")
    ap.add_argument("--instances_dir", default="trainable_pool/instances")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--cups_prefix", default="data_cups")
    ap.add_argument("--val_split_stage1", type=float, default=0.15)
    ap.add_argument("--val_split_stage2", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    instances_dir = Path(args.instances_dir)
    out_dir = Path(args.out_dir)

    all_ids = collect_pair_ids(images_dir, instances_dir)
    cups_ids = [x for x in all_ids if x.startswith(args.cups_prefix)]
    non_cups_ids = [x for x in all_ids if not x.startswith(args.cups_prefix)]

    s1_train, s1_val = split_ids(non_cups_ids, args.val_split_stage1, args.seed)
    s2_train, s2_val = split_ids(cups_ids, args.val_split_stage2, args.seed)

    save_ids(out_dir / "stage1_pretrain_train_ids.txt", s1_train)
    save_ids(out_dir / "stage1_pretrain_val_ids.txt", s1_val)
    save_ids(out_dir / "stage2_finetune_train_ids.txt", s2_train)
    save_ids(out_dir / "stage2_finetune_val_ids.txt", s2_val)

    print(f"all_ids={len(all_ids)}")
    print(f"cups_ids={len(cups_ids)}")
    print(f"non_cups_ids={len(non_cups_ids)}")
    print(f"stage1 train={len(s1_train)} val={len(s1_val)}")
    print(f"stage2 train={len(s2_train)} val={len(s2_val)}")
    print(f"saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

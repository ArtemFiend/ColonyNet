from __future__ import annotations
import os
import glob
from .splits import split_ids
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .targets import make_targets_from_instances

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def _list_images(img_dir: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    files = sorted(files)
    return files

class ImageInstancesDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        instances_dir: str,
        transform,
        img_size: int,
        out_stride: int = 4,
        ids: list[str] | None = None,
        target_cfg: dict | None = None,
        repeat: int = 1,
    ):
        self.images_dir = images_dir
        self.instances_dir = instances_dir
        self.transform = transform
        self.img_size = img_size
        self.out_stride = out_stride
        self.target_cfg = target_cfg or {}
        self.repeat = max(1, int(repeat))

        if ids is None:
            self.images = _list_images(images_dir)
            self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.images]
        else:
            self.ids = ids
            self.images = [os.path.join(images_dir, f"{i}.png") for i in ids]

        # validate
        self.samples = []
        for _id, img_path in zip(self.ids, self.images):
            if not os.path.exists(img_path):
                # try other extension
                found = None
                for ext in IMG_EXTS:
                    cand = os.path.join(images_dir, _id + ext)
                    if os.path.exists(cand):
                        found = cand
                        break
                if found is None:
                    continue
                img_path = found
            inst_path = os.path.join(instances_dir, f"{_id}.png")
            if os.path.exists(inst_path):
                self.samples.append((_id, img_path, inst_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {images_dir} + {instances_dir}")

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx: int):
        if self.repeat > 1:
            idx = idx % len(self.samples)
        _id, img_path, inst_path = self.samples[idx]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        if inst is None:
            raise RuntimeError(f"Failed to read instances: {inst_path}")
        if inst.ndim == 3:
            inst = inst[:, :, 0]
        inst = inst.astype(np.int32)

        aug = self.transform(image=img_rgb, instances=inst)
        x = aug["image"]  # torch float32 C,H,W in [0,1]
        inst_t = aug["instances"].cpu().numpy().astype(np.int32)

        H, W = inst_t.shape[:2]
        out_h, out_w = H // self.out_stride, W // self.out_stride
        t = make_targets_from_instances(inst_t, out_h, out_w, **self.target_cfg)

        # to torch
        y_sem = torch.from_numpy(t["sem"]).unsqueeze(0)        # 1, h, w
        y_center = torch.from_numpy(t["center"]).unsqueeze(0)  # 1, h, w
        y_boundary = torch.from_numpy(t["boundary"]).unsqueeze(0)

        return {
            "id": _id,
            "image": x,
            "instances": torch.from_numpy(inst_t).long(),
            "y_sem": y_sem,
            "y_center": y_center,
            "y_boundary": y_boundary,
        }

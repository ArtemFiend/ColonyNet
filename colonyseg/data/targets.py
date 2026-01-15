from __future__ import annotations
import numpy as np
import cv2
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: float):
    h, w = heatmap.shape
    # Kernel radius ~ 3 sigma
    r = max(2, int(3 * sigma))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma + 1e-8)).astype(np.float32)

    patch = heatmap[y0:y1, x0:x1]
    np.maximum(patch, g, out=patch)
    heatmap[y0:y1, x0:x1] = patch

def make_targets_from_instances(
    instances_hw: np.ndarray,
    out_h: int,
    out_w: int,
    boundary_thickness: int = 2,
    sigma_min: float = 2.0,
    sigma_max: float = 6.0,
) -> dict[str, np.ndarray]:
    """
    instances_hw: HxW int32 mask with ids 0..N
    returns float32 maps at out_h x out_w: sem, center, boundary (0..1)
    """
    # Downscale instance ids to output resolution (nearest)
    inst = cv2.resize(
        instances_hw.astype(np.int32),
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST
    )

    sem = (inst > 0).astype(np.float32)

    # Boundary: thick boundaries on labeled mask catches both outer + inner boundaries between instances
    b = find_boundaries(inst, mode="thick").astype(np.uint8)
    if boundary_thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_thickness*2+1, boundary_thickness*2+1))
        b = cv2.dilate(b, k, iterations=1)
    boundary = b.astype(np.float32)

    # Centers: gaussian at each instance centroid; sigma proportional to instance size (clipped)
    center = np.zeros((out_h, out_w), dtype=np.float32)
    props = regionprops(inst)
    for p in props:
        if p.label == 0:
            continue
        cy, cx = p.centroid  # (row, col)
        area = float(p.area)
        # equiv diameter ~ sqrt(4A/pi); map to sigma
        eq_d = np.sqrt(4.0 * area / (np.pi + 1e-8))
        sigma = np.clip(eq_d * 0.10, sigma_min, sigma_max)
        _draw_gaussian(center, int(round(cx)), int(round(cy)), float(sigma))

    return {"sem": sem, "center": center, "boundary": boundary}

from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table

def postprocess_watershed(
    sem_prob: np.ndarray,
    center_prob: np.ndarray,
    boundary_prob: np.ndarray,
    t_sem: float = 0.5,
    t_center: float = 0.35,
    min_distance: int = 6,
    lambda_boundary: float = 3.0,
    area_min: int = 20,
    area_max: int = 200000,
):
    """Return instance labels (H,W) int32 where 0=background, 1..N instances.

    Inputs are expected at the same resolution.
    """
    sem = (sem_prob >= t_sem).astype(np.uint8)
    if sem.sum() == 0:
        return np.zeros_like(sem, dtype=np.int32)

    dist = ndi.distance_transform_edt(sem)
    # Seeds from center heatmap within sem mask
    cm = (center_prob * sem).astype(np.float32)
    coords = peak_local_max(cm, labels=sem, min_distance=min_distance, threshold_abs=t_center)

    markers = np.zeros_like(sem, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    if markers.max() == 0:
        # fallback: use distance peaks if center head is weak
        coords = peak_local_max(dist, labels=sem, min_distance=min_distance)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        if markers.max() == 0:
            return np.zeros_like(sem, dtype=np.int32)

    energy = -dist + lambda_boundary * boundary_prob.astype(np.float32)
    labels = watershed(energy, markers, mask=sem.astype(bool))

    # Filter by area (cheap)
    tbl = regionprops_table(labels, properties=("label", "area"))
    keep = []
    for lab, a in zip(tbl["label"], tbl["area"]):
        if area_min <= int(a) <= area_max:
            keep.append(int(lab))
    out = labels.astype(np.int32)
    out[~np.isin(out, keep)] = 0
    return out

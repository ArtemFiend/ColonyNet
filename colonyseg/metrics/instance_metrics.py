from __future__ import annotations
import numpy as np

def _densify_labels(x: np.ndarray):
    """Map arbitrary positive instance ids to contiguous 1..N, keep 0 as background."""
    x = x.astype(np.int64, copy=False)
    labels = np.unique(x)
    labels = labels[labels != 0]
    n = int(labels.size)
    if n == 0:
        return np.zeros_like(x, dtype=np.int32), 0
    out = np.zeros_like(x, dtype=np.int32)
    mask = x > 0
    out[mask] = np.searchsorted(labels, x[mask]).astype(np.int32) + 1
    return out, n

def _contingency(gt: np.ndarray, pr: np.ndarray):
    gt, gt_n = _densify_labels(gt)
    pr, pr_n = _densify_labels(pr)
    if gt_n == 0 or pr_n == 0:
        return None, gt_n, pr_n
    # pair id
    pair = gt * (pr_n + 1) + pr
    inter = np.bincount(pair.ravel(), minlength=(gt_n + 1) * (pr_n + 1)).reshape(gt_n + 1, pr_n + 1)
    return inter, gt_n, pr_n

def instance_scores(gt: np.ndarray, pr: np.ndarray, iou_thr: float = 0.5):
    """Returns dict with F1, precision, recall, merge_count, split_count, count_err."""
    gt = gt.astype(np.int32)
    pr = pr.astype(np.int32)
    inter, gt_n, pr_n = _contingency(gt, pr)
    if inter is None:
        tp = 0
        fp = pr_n
        fn = gt_n
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2*prec*rec / max(1e-8, prec+rec)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "merge": 0, "split": 0,
            "count_err": abs(pr_n - gt_n),
            "gt_n": gt_n, "pr_n": pr_n,
        }

    gt_area = inter.sum(axis=1)  # (gt_max+1,)
    pr_area = inter.sum(axis=0)  # (pr_max+1,)

    # IoU for all pairs (ignore 0)
    ious = []
    g_idx, p_idx = np.nonzero(inter[1:, 1:])
    for g0, p0 in zip(g_idx.tolist(), p_idx.tolist()):
        g = g0 + 1
        p = p0 + 1
        inter_gp = inter[g, p]
        union = gt_area[g] + pr_area[p] - inter_gp
        iou = inter_gp / max(1, union)
        ious.append((iou, g, p))
    ious.sort(reverse=True, key=lambda x: x[0])

    matched_g = set()
    matched_p = set()
    tp = 0
    for iou, g, p in ious:
        if iou < iou_thr:
            break
        if g in matched_g or p in matched_p:
            continue
        matched_g.add(g)
        matched_p.add(p)
        tp += 1

    fp = pr_n - tp
    fn = gt_n - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2*prec*rec / max(1e-8, prec+rec)

    # merge/split: how many overlaps above thr per object
    # compute overlaps counts
    overlaps_thr = {("g", g): 0 for g in range(1, gt_n + 1)}
    overlaps_thr.update({("p", p): 0 for p in range(1, pr_n + 1)})

    for iou, g, p in ious:
        if iou < iou_thr:
            break
        overlaps_thr[("g", g)] += 1
        overlaps_thr[("p", p)] += 1

    split = sum(1 for g in range(1, gt_n + 1) if overlaps_thr[("g", g)] > 1)
    merge = sum(1 for p in range(1, pr_n + 1) if overlaps_thr[("p", p)] > 1)

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "merge": merge, "split": split,
        "count_err": abs(pr_n - gt_n),
        "gt_n": gt_n, "pr_n": pr_n,
    }

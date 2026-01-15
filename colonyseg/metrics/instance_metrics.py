from __future__ import annotations
import numpy as np

def _contingency(gt: np.ndarray, pr: np.ndarray):
    gt = gt.astype(np.int64)
    pr = pr.astype(np.int64)
    gt_max = int(gt.max())
    pr_max = int(pr.max())
    if gt_max == 0 or pr_max == 0:
        return None, gt_max, pr_max
    # pair id
    pair = gt * (pr_max + 1) + pr
    inter = np.bincount(pair.ravel(), minlength=(gt_max+1)*(pr_max+1)).reshape(gt_max+1, pr_max+1)
    return inter, gt_max, pr_max

def instance_scores(gt: np.ndarray, pr: np.ndarray, iou_thr: float = 0.5):
    """Returns dict with F1, precision, recall, merge_count, split_count, count_err."""
    gt = gt.astype(np.int32)
    pr = pr.astype(np.int32)
    inter, gt_max, pr_max = _contingency(gt, pr)
    if inter is None:
        tp = 0
        fp = pr_max
        fn = gt_max
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2*prec*rec / max(1e-8, prec+rec)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "merge": 0, "split": 0,
            "count_err": abs(pr_max - gt_max),
            "gt_n": gt_max, "pr_n": pr_max,
        }

    gt_area = inter.sum(axis=1)  # (gt_max+1,)
    pr_area = inter.sum(axis=0)  # (pr_max+1,)

    # IoU for all pairs (ignore 0)
    ious = []
    for g in range(1, gt_max+1):
        for p in range(1, pr_max+1):
            inter_gp = inter[g, p]
            if inter_gp == 0:
                continue
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

    fp = pr_max - tp
    fn = gt_max - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2*prec*rec / max(1e-8, prec+rec)

    # merge/split: how many overlaps above thr per object
    # compute overlaps counts
    overlaps_thr = {("g", g): 0 for g in range(1, gt_max+1)}
    overlaps_thr.update({("p", p): 0 for p in range(1, pr_max+1)})

    for iou, g, p in ious:
        if iou < iou_thr:
            break
        overlaps_thr[("g", g)] += 1
        overlaps_thr[("p", p)] += 1

    split = sum(1 for g in range(1, gt_max+1) if overlaps_thr[("g", g)] > 1)
    merge = sum(1 for p in range(1, pr_max+1) if overlaps_thr[("p", p)] > 1)

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "merge": merge, "split": split,
        "count_err": abs(pr_max - gt_max),
        "gt_n": gt_max, "pr_n": pr_max,
    }

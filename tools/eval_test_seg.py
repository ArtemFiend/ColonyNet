from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from colonyseg.eval import evaluate_checkpoint_on_test_image


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate checkpoint on one test image (petri crop + watershed + optional GT compare)."
    )
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (*.pt)")
    ap.add_argument("--image_path", required=True, help="Path to test image")
    ap.add_argument(
        "--reference_path",
        default=None,
        help="Path to GT instance labels (.png/.tif) or folder with per-instance masks",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for artifacts and report")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--mask_outside", action="store_true")
    ap.add_argument("--min_r_frac", type=float, default=0.35)
    ap.add_argument("--max_r_frac", type=float, default=0.55)
    ap.add_argument("--center_tol", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cuda/cpu; default=auto")
    args = ap.parse_args()

    result = evaluate_checkpoint_on_test_image(
        ckpt_path=args.ckpt,
        image_path=args.image_path,
        out_dir=args.out_dir,
        reference_path=args.reference_path,
        iou_thr=args.iou_thr,
        petri_pad=args.pad,
        petri_mask_outside=bool(args.mask_outside),
        petri_min_r_frac=args.min_r_frac,
        petri_max_r_frac=args.max_r_frac,
        petri_center_tol=args.center_tol,
        device=args.device,
    )

    print("Done.")
    print(f"Report: {result['report_path']}")
    print("Metrics:")
    for k, v in sorted(result["metrics"].items()):
        print(f"  {k}: {v}")

    out_json = os.path.join(args.out_dir, "result_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()

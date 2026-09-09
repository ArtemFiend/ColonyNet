"""Build an explanatory visual example for colony anomaly detection outputs."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle


DEFAULT_RESULT_DIR = Path("outputs/colony_anomaly_detection/1127763_IMG_4363")


EVIDENCE_COLUMNS = [
    ("size_shape_score_rank", "форма/размер"),
    ("color_background_score_rank", "цвет/фон"),
    ("intensity_score_rank", "яркость"),
    ("texture_score_rank", "текстура"),
    ("histogram_score_rank", "гистограмма"),
    ("neighbor_difference_score_rank", "соседи"),
    ("morphotype_score_rank", "морфотип"),
]


def read_image_rgb(path: Path) -> np.ndarray:
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def choose_candidate(selected_df: pd.DataFrame, scores_df: pd.DataFrame, candidate_id: int | None) -> pd.Series:
    source = selected_df if not selected_df.empty else scores_df
    if source.empty:
        raise ValueError("No anomaly rows found in selected_anomalies.csv or colony_anomaly_scores.csv")

    source = source.copy()
    source["colony_id"] = pd.to_numeric(source["colony_id"], errors="coerce")

    if candidate_id is not None:
        match = source[source["colony_id"] == int(candidate_id)]
        if match.empty:
            raise ValueError(f"Candidate colony_id={candidate_id} not found")
        return match.iloc[0]

    sort_col = "anomaly_rank" if "anomaly_rank" in source.columns else "final_anomaly_score"
    ascending = sort_col == "anomaly_rank"
    return source.sort_values(sort_col, ascending=ascending, na_position="last").iloc[0]


def label_contours(label_map: np.ndarray, label_id: int) -> list[np.ndarray]:
    mask = (label_map == int(label_id)).astype(np.uint8)
    if mask.max() == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] for c in contours if len(c) >= 3]


def draw_label_contours(
    ax,
    label_map: np.ndarray,
    label_ids: list[int],
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
):
    for label_id in label_ids:
        for contour in label_contours(label_map, label_id):
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                solid_joinstyle="round",
                solid_capstyle="round",
            )


def centroid_from_mask(label_map: np.ndarray, label_id: int) -> tuple[float, float]:
    ys, xs = np.where(label_map == int(label_id))
    if len(xs) == 0:
        raise ValueError(f"No pixels found for colony_id={label_id}")
    return float(xs.mean()), float(ys.mean())


def bbox_from_mask(label_map: np.ndarray, label_id: int, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(label_map == int(label_id))
    if len(xs) == 0:
        raise ValueError(f"No pixels found for colony_id={label_id}")
    h, w = shape
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def expand_to_square(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    h, w = shape
    side = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return x0, y0, x1, y1


def clean_reasons(text: object, max_items: int = 4) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    reasons = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        # Keep the human reason and drop the numeric evidence tail.
        reason = part.split(":", 1)[0].strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= max_items:
            break
    return reasons


def format_wrapped_lines(lines: list[str], width: int = 48) -> str:
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(f"- {line}", width=width, subsequent_indent="  "))
    return "\n".join(wrapped)


def build_visual(result_dir: Path, out_path: Path, candidate_id: int | None = None) -> Path:
    image_path = result_dir / "preprocessed_736.png"
    mask_path = result_dir / "masks_labeled.npy"
    scores_path = result_dir / "colony_anomaly_scores.csv"
    selected_path = result_dir / "selected_anomalies.csv"
    review_path = result_dir / "review_candidates.csv"
    technical_path = result_dir / "technical_warnings.csv"

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = read_image_rgb(image_path)
    label_map = np.load(mask_path)
    scores_df = load_csv(scores_path)
    selected_df = load_csv(selected_path)
    review_df = load_csv(review_path)
    technical_df = load_csv(technical_path)

    candidate = choose_candidate(selected_df, scores_df, candidate_id)
    cid = int(candidate["colony_id"])
    candidate_score = float(candidate.get("final_anomaly_score", np.nan))

    selected_ids = (
        selected_df["colony_id"].dropna().astype(int).tolist()
        if not selected_df.empty and "colony_id" in selected_df.columns
        else [cid]
    )
    review_ids = (
        review_df["colony_id"].dropna().astype(int).tolist()
        if not review_df.empty and "colony_id" in review_df.columns
        else []
    )
    technical_ids = (
        technical_df["colony_id"].dropna().astype(int).tolist()
        if not technical_df.empty and "colony_id" in technical_df.columns
        else []
    )
    all_ids = [int(i) for i in np.unique(label_map) if int(i) > 0]

    h, w = image.shape[:2]
    zoom_box = expand_to_square(bbox_from_mask(label_map, cid, pad=70, shape=(h, w)), shape=(h, w))
    x0, y0, x1, y1 = zoom_box
    crop_img = image[y0:y1, x0:x1]
    crop_map = label_map[y0:y1, x0:x1]

    fig = plt.figure(figsize=(15, 8.5), facecolor="white", constrained_layout=True)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 0.48, 0.62],
    )
    ax_full = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_reason = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[2, 1])

    fig.suptitle("Визуальное объяснение анализа аномальности колоний", fontsize=18, fontweight="bold")

    ax_full.imshow(image)
    ax_full.set_title("Чашка: все колонии и кандидаты", fontsize=13)
    ax_full.axis("off")
    normal_ids = [i for i in all_ids if i not in selected_ids and i not in review_ids and i not in technical_ids]
    draw_label_contours(ax_full, label_map, normal_ids, color="#31c7d8", linewidth=0.65, alpha=0.55)
    draw_label_contours(ax_full, label_map, technical_ids, color="#8a8a8a", linewidth=1.0, linestyle=":", alpha=0.75)
    draw_label_contours(ax_full, label_map, review_ids, color="#f59f00", linewidth=1.5, linestyle="--", alpha=0.9)
    draw_label_contours(ax_full, label_map, selected_ids, color="#ffd60a", linewidth=2.4, alpha=1.0)
    draw_label_contours(ax_full, label_map, [cid], color="#ff2d75", linewidth=3.8, alpha=1.0)
    ax_full.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff2d75", linewidth=2.5)
    )
    cx, cy = centroid_from_mask(label_map, cid)
    ax_full.text(
        cx,
        cy - 22,
        f"ID {cid}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="#ffd60a", edgecolor="black", boxstyle="round,pad=0.22"),
    )
    ax_full.legend(
        handles=[
            Patch(facecolor="#ff2d75", edgecolor="#ff2d75", label="выбранный пример"),
            Patch(facecolor="#ffd60a", edgecolor="#ffd60a", label="selected_anomalies"),
            Patch(facecolor="#f59f00", edgecolor="#f59f00", label="review_candidates"),
            Patch(facecolor="#31c7d8", edgecolor="#31c7d8", label="обычные колонии"),
        ],
        loc="lower left",
        framealpha=0.92,
        fontsize=9,
    )

    ax_zoom.imshow(crop_img)
    ax_zoom.set_title(f"Увеличение: колония ID {cid}", fontsize=13)
    ax_zoom.axis("off")
    crop_ids = [int(i) for i in np.unique(crop_map) if int(i) > 0 and int(i) != cid]
    draw_label_contours(ax_zoom, crop_map, crop_ids, color="#31c7d8", linewidth=1.1, alpha=0.7)
    draw_label_contours(ax_zoom, crop_map, [cid], color="black", linewidth=5.0, alpha=0.95)
    draw_label_contours(ax_zoom, crop_map, [cid], color="#ff2d75", linewidth=3.0, alpha=1.0)

    ax_reason.axis("off")
    ax_reason.set_title("Почему система выделила эту колонию", fontsize=13, loc="left")
    reasons = clean_reasons(candidate.get("explanations_text", ""))
    if not reasons:
        reasons = ["совокупность признаков отличается от основной массы колоний"]
    reason_text = format_wrapped_lines(reasons)
    ax_reason.text(0.0, 0.98, reason_text, ha="left", va="top", fontsize=10, linespacing=1.25)

    labels: list[str] = []
    values: list[float] = []
    for col, label in EVIDENCE_COLUMNS:
        if col in candidate.index and pd.notna(candidate[col]):
            labels.append(label)
            values.append(float(candidate[col]))
    if labels:
        y = np.arange(len(labels))
        colors = ["#ff2d75" if v >= 0.9 else "#f59f00" if v >= 0.75 else "#80c7d8" for v in values]
        ax_bar.barh(y, values, color=colors, height=0.62)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(labels, fontsize=9)
        ax_bar.set_xlim(0, 1)
        ax_bar.invert_yaxis()
        ax_bar.set_xticks([])
        ax_bar.set_title("Профиль необычности по группам признаков", fontsize=10, loc="left")
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.65)
        ax_bar.text(
            0.0,
            -0.08,
            "длиннее полоса = сильнее отличие от типичных колоний",
            transform=ax_bar.transAxes,
            fontsize=8.5,
            color="#555555",
        )
    else:
        ax_bar.axis("off")

    status = str(candidate.get("recommendation_status", ""))
    score_text = "высокий" if np.isfinite(candidate_score) and candidate_score >= 0.85 else "средний"
    ax_bar.text(
        0.0,
        -0.24,
        f"Статус: {status or 'candidate'} | интегральная оценка: {score_text}",
        transform=ax_bar.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        color="#333333",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--candidate-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (args.result_dir / "anomaly_visual_explained.png")
    path = build_visual(args.result_dir, out, candidate_id=args.candidate_id)
    print(path)


if __name__ == "__main__":
    main()

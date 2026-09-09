from __future__ import annotations

import math
import os
import queue
import colorsys
import sys
import threading
import time
import traceback
try:
    from full_pipline.runtime import create_run_dir, validate_image_batch, safe_table
except ImportError:
    from runtime import create_run_dir, validate_image_batch, safe_table
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from skimage.measure import find_contours

try:
    from full_pipline.full_pipeline import (
        AnomalyDetectionConfig,
        load_pipeline_models,
        make_full_pipeline_config,
        normalize_masks_input,
        read_image_rgb,
        run_single_image_pipeline,
        select_visual_highlight_ids,
        visualize_anomalies,
    )
except ImportError:
    from full_pipeline import (
        AnomalyDetectionConfig,
        load_pipeline_models,
        make_full_pipeline_config,
        normalize_masks_input,
        read_image_rgb,
        run_single_image_pipeline,
        select_visual_highlight_ids,
        visualize_anomalies,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_ANOMALY_VISUALIZATION_FRACTION = 0.20
MAX_ANOMALY_VISUALIZATION_COUNT: int | None = 20
MIN_ANOMALIES_TO_SHOW = 0
MAX_VISUAL_HIGHLIGHTS_PER_NEIGHBORHOOD = 2
MAX_VISUAL_HIGHLIGHTS_PER_MORPHOTYPE = 2
VISUAL_HIGHLIGHT_EDGE_BAND_DIAMETERS = 2.0
MAX_VISUAL_HIGHLIGHTS_EDGE_FRACTION: float | None = 0.25
VISUAL_HIGHLIGHT_EDGE_SCORE_PENALTY = 0.20
MAX_VISUAL_REVIEW_SEGMENTATION_FRACTION: float | None = 0.20
MAX_VISUAL_REVIEW_SEGMENTATION_COUNT: int | None = 4


def app_base_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def default_output_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ColonyNet" / "outputs"
    return Path.cwd() / "outputs" / "colony_pipeline_app"


def collect_images_from_folder(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Не удалось сохранить изображение: {path}")
    encoded.tofile(path)


def figure_to_rgb_image(fig, crop_to_axes: bool = True) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    if crop_to_axes and fig.axes:
        bbox = fig.axes[0].get_window_extent()
        height, width = rgba.shape[:2]
        x0 = max(0, int(math.floor(bbox.x0)))
        x1 = min(width, int(math.ceil(bbox.x1)))
        y0 = max(0, int(math.floor(height - bbox.y1)))
        y1 = min(height, int(math.ceil(height - bbox.y0)))
        if x1 > x0 and y1 > y0:
            rgba = rgba[y0:y1, x0:x1]
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def draw_petri_bbox(image_rgb: np.ndarray, preprocess_info: dict[str, Any]) -> np.ndarray:
    view = image_rgb.copy()
    xyxy = preprocess_info.get("crop_xyxy") if isinstance(preprocess_info, dict) else None
    detected = bool(preprocess_info.get("petri_detected", False)) if isinstance(preprocess_info, dict) else False
    if detected and xyxy is not None and len(xyxy) == 4:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        thickness = max(3, int(round(min(view.shape[:2]) * 0.006)))
        cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), thickness=thickness)
    return view


def mask_color(index: int) -> np.ndarray:
    hue = (0.13 + 0.61803398875 * index) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return np.array(rgb, dtype=float)


def overlay_instance_masks(image_rgb: np.ndarray, masks, alpha: float = 0.34) -> np.ndarray:
    masks_list = normalize_masks_input(masks)
    base = image_rgb.astype(np.float32) / 255.0
    out = base.copy()
    h, w = image_rgb.shape[:2]
    smooth_sigma = 1.6
    contour_level = 0.35

    for idx, mask in enumerate(masks_list):
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = mask_color(idx)
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color

    out_u8 = np.clip(out * 255, 0, 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.imshow(out_u8)
    ax.axis("off")

    for idx, mask in enumerate(masks_list):
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        color = mask_color(idx)
        mask_prob = cv2.GaussianBlur(mask_bool.astype(np.float32), (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
        for contour in find_contours(mask_prob, level=contour_level):
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                color=color,
                linewidth=0.75,
                antialiased=True,
                solid_joinstyle="round",
                solid_capstyle="round",
            )

    return figure_to_rgb_image(fig)


def anomaly_visualization_limit(n_colonies: int) -> int:
    if n_colonies <= 0:
        return 0
    by_fraction = int(math.floor(n_colonies * float(MAX_ANOMALY_VISUALIZATION_FRACTION)))
    limit = max(int(MIN_ANOMALIES_TO_SHOW), by_fraction)
    if MAX_ANOMALY_VISUALIZATION_COUNT is not None:
        limit = min(limit, int(MAX_ANOMALY_VISUALIZATION_COUNT))
    return max(0, min(limit, int(n_colonies)))


def choose_anomaly_ids_for_visualization(scores_df: pd.DataFrame, n_colonies: int) -> list[int]:
    return select_visual_highlight_ids(
        scores_df,
        n_colonies=n_colonies,
        max_fraction=MAX_ANOMALY_VISUALIZATION_FRACTION,
        max_count=MAX_ANOMALY_VISUALIZATION_COUNT,
        min_count=MIN_ANOMALIES_TO_SHOW,
        max_per_neighborhood=MAX_VISUAL_HIGHLIGHTS_PER_NEIGHBORHOOD,
        max_per_morphotype=MAX_VISUAL_HIGHLIGHTS_PER_MORPHOTYPE,
        edge_band_diameters=VISUAL_HIGHLIGHT_EDGE_BAND_DIAMETERS,
        max_edge_fraction=MAX_VISUAL_HIGHLIGHTS_EDGE_FRACTION,
        edge_score_penalty=VISUAL_HIGHLIGHT_EDGE_SCORE_PENALTY,
        max_review_segmentation_fraction=MAX_VISUAL_REVIEW_SEGMENTATION_FRACTION,
        max_review_segmentation_count=MAX_VISUAL_REVIEW_SEGMENTATION_COUNT,
    )


def render_anomaly_overlay(result: dict[str, Any], show_scores: bool) -> np.ndarray:
    scores_df = result.get("scores", pd.DataFrame())
    masks = result.get("masks", [])
    n_colonies = len(normalize_masks_input(masks))
    highlight_ids = choose_anomaly_ids_for_visualization(scores_df, n_colonies)

    selected_df = result.get("selected", pd.DataFrame())
    selected_ids = (
        selected_df["colony_id"].dropna().astype(int).tolist()
        if isinstance(selected_df, pd.DataFrame) and not selected_df.empty and "colony_id" in selected_df.columns
        else []
    )

    fig = visualize_anomalies(
        image_rgb=result["image"],
        masks=masks,
        results_df=scores_df,
        selected_ids=selected_ids,
        save_path=None,
        title=None,
        show_scores=show_scores,
        show_review=True,
        show_technical_warnings=False,
        highlight_ids=highlight_ids,
    )
    return figure_to_rgb_image(fig)


def result_summary_row(image_path: Path, result: dict[str, Any], image_output_dir: Path) -> dict[str, Any]:
    scores_df = result.get("scores", pd.DataFrame())
    selected_df = result.get("selected", pd.DataFrame())
    visual_df = result.get("visual_highlights", pd.DataFrame())
    review_df = result.get("review_candidates", pd.DataFrame())
    technical_df = result.get("technical_warnings", pd.DataFrame())
    plate_quality_df = result.get("plate_quality", pd.DataFrame())

    n_detected = int(len(scores_df))
    n_valid = (
        int(scores_df["valid_for_anomaly"].fillna(False).astype(bool).sum())
        if n_detected and "valid_for_anomaly" in scores_df.columns
        else 0
    )
    max_score = (
        float(scores_df["final_anomaly_score_raw"].max())
        if n_detected and "final_anomaly_score_raw" in scores_df.columns
        else np.nan
    )
    quality = (
        str(plate_quality_df.loc[0, "plate_quality_status"])
        if isinstance(plate_quality_df, pd.DataFrame)
        and not plate_quality_df.empty
        and "plate_quality_status" in plate_quality_df.columns
        else "unknown"
    )

    return {
        "image_name": image_path.name,
        "n_detected_colonies": n_detected,
        "n_valid_colonies": n_valid,
        "n_selected_anomalies": int(len(selected_df)) if isinstance(selected_df, pd.DataFrame) else 0,
        "n_visual_highlights": int(len(visual_df)) if isinstance(visual_df, pd.DataFrame) else 0,
        "n_review_candidates": int(len(review_df)) if isinstance(review_df, pd.DataFrame) else 0,
        "n_technical_warnings": int(len(technical_df)) if isinstance(technical_df, pd.DataFrame) else 0,
        "max_anomaly_score_raw": max_score,
        "plate_quality_status": quality,
        "output_dir": str(image_output_dir),
    }


def build_runtime_config(output_dir: Path, save_xlsx: bool, save_colony_crops: bool) -> AnomalyDetectionConfig:
    config = make_full_pipeline_config(output_dir=output_dir)
    config.save_csv = True
    config.save_xlsx = bool(save_xlsx)
    config.save_visualizations = False
    config.save_colony_crops = bool(save_colony_crops)
    config.save_feature_space_plot = False
    config.save_feature_correlation_report = True
    config.visual_highlight_percent = MAX_ANOMALY_VISUALIZATION_FRACTION
    config.visual_highlight_max_count = MAX_ANOMALY_VISUALIZATION_COUNT
    config.visual_highlight_min_count = MIN_ANOMALIES_TO_SHOW
    config.visual_highlight_max_per_neighborhood = MAX_VISUAL_HIGHLIGHTS_PER_NEIGHBORHOOD
    config.visual_highlight_max_per_morphotype = MAX_VISUAL_HIGHLIGHTS_PER_MORPHOTYPE
    config.visual_highlight_edge_band_diameters = VISUAL_HIGHLIGHT_EDGE_BAND_DIAMETERS
    config.visual_highlight_max_edge_fraction = MAX_VISUAL_HIGHLIGHTS_EDGE_FRACTION
    config.visual_highlight_edge_score_penalty = VISUAL_HIGHLIGHT_EDGE_SCORE_PENALTY
    config.visual_highlight_max_review_segmentation_fraction = MAX_VISUAL_REVIEW_SEGMENTATION_FRACTION
    config.visual_highlight_max_review_segmentation_count = MAX_VISUAL_REVIEW_SEGMENTATION_COUNT
    return config


def save_stage_images(result: dict[str, Any], image_output_dir: Path) -> None:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    write_rgb_image(image_output_dir / "stage_01_original.png", result["image_original"])
    write_rgb_image(image_output_dir / "stage_02_petri_bbox.png", draw_petri_bbox(result["image_original"], result["preprocess"]))
    write_rgb_image(image_output_dir / "stage_03_petri_crop_736.png", result["image"])
    write_rgb_image(image_output_dir / "stage_04_colony_masks.png", overlay_instance_masks(result["image"], result["masks"]))
    write_rgb_image(image_output_dir / "stage_05_anomalies_scores.png", render_anomaly_overlay(result, show_scores=True))
    write_rgb_image(image_output_dir / "stage_05_anomalies_no_scores.png", render_anomaly_overlay(result, show_scores=False))


class ColonyPipelineApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ColonyNet • Анализ колоний")
        self.root.geometry(f"{min(1400, root.winfo_screenwidth() - 80)}x{min(900, root.winfo_screenheight() - 100)}+20+20")
        self.root.minsize(1000, 680)

        self.image_paths: list[Path] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.summary_rows: list[dict[str, Any]] = []
        self.render_cache: dict[tuple[str, str, bool], np.ndarray] = {}
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.running = False
        self.cancel_event = threading.Event()
        self.last_output_dir: Path | None = None

        self.output_dir_var = tk.StringVar(value=str(default_output_dir()))
        self.show_scores_var = tk.BooleanVar(value=True)
        self.save_xlsx_var = tk.BooleanVar(value=True)
        self.save_colony_crops_var = tk.BooleanVar(value=False)
        self.stage_var = tk.StringVar(value="Аномалии")
        self.status_var = tk.StringVar(value="Выберите изображения и запустите пайплайн.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.current_preview_key: str | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_rgb: np.ndarray | None = None

        self._configure_theme()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_worker_queue()

    def _configure_theme(self) -> None:
        self.root.configure(bg="#0b1220")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#111c2e", foreground="#e4edf7", borderwidth=0, font=("Segoe UI", 10))
        style.configure("TFrame", background="#111c2e")
        style.configure("TLabel", padding=(0, 3))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"), foreground="#5eead4")
        style.configure("Muted.TLabel", foreground="#94a9c2")
        style.configure("TButton", background="#253650", padding=(12, 10), font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#354b69"), ("disabled", "#192538")])
        style.configure("Accent.TButton", background="#0f766e", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#0d9488"), ("disabled", "#253650")])
        style.configure("TEntry", fieldbackground="#0b1220", foreground="#e4edf7", padding=8)
        style.configure("TCombobox", fieldbackground="#0b1220", arrowcolor="#5eead4", padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", "#0b1220")], foreground=[("readonly", "#e4edf7")])
        style.configure("TLabelframe", background="#111c2e", bordercolor="#253650")
        style.configure("TLabelframe.Label", foreground="#94a9c2")
        style.configure("Treeview", background="#111c2e", fieldbackground="#111c2e", rowheight=32)
        style.configure("Treeview.Heading", background="#253650", padding=8, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#115e59")])
        style.configure("TCheckbutton", padding=4)
        style.map("TCheckbutton", background=[("active", "#253650")])
        style.configure("Horizontal.TProgressbar", background="#2dd4bf", troughcolor="#253650", thickness=7)

    def cancel_pipeline(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Остановка после текущего изображения…")

    def _on_close(self) -> None:
        if self.running:
            self.cancel_pipeline()
            self.status_var.set("Дождитесь сохранения текущего изображения, затем закройте окно.")
            return
        self.root.destroy()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, padding=(24, 18))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="ColonyNet", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Лаборатория анализа колоний", style="Muted.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(header, text="●  ЛОКАЛЬНАЯ ОБРАБОТКА", foreground="#5eead4").grid(row=0, column=1, sticky="e")
        ttk.Label(header, text="Снимок → чашка → колонии → аномалии", style="Muted.TLabel").grid(row=1, column=1, sticky="e")

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=1, column=0, sticky="ns", padx=12, pady=(0, 12))
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Входные изображения").grid(row=0, column=0, sticky="w")
        list_frame = ttk.Frame(controls)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 8))
        controls.rowconfigure(1, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.file_list = tk.Listbox(list_frame, width=34, height=12, exportselection=False, bg="#0b1220", fg="#dbeafe", selectbackground="#115e59", selectforeground="#ffffff", relief="flat", highlightthickness=0, activestyle="none")
        self.file_list.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=list_scroll.set)

        button_grid = ttk.Frame(controls)
        button_grid.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        button_grid.columnconfigure((0, 1), weight=1)
        ttk.Button(button_grid, text="Добавить файлы", command=self.add_files).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_grid, text="Добавить папку", command=self.add_folder).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(button_grid, text="Удалить", command=self.remove_selected_files).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(6, 0))
        ttk.Button(button_grid, text="Очистить", command=self.clear_files).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(6, 0))

        ttk.Label(controls, text="Папка результата").grid(row=3, column=0, sticky="w", pady=(8, 0))
        output_frame = ttk.Frame(controls)
        output_frame.grid(row=4, column=0, sticky="ew", pady=(6, 8))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(output_frame, text="...", width=4, command=self.choose_output_dir).grid(row=0, column=1)

        options = ttk.LabelFrame(controls, text="Опции", padding=8)
        options.grid(row=5, column=0, sticky="ew", pady=(4, 8))
        ttk.Checkbutton(
            options,
            text="Показывать оценки аномалий",
            variable=self.show_scores_var,
            command=self.refresh_preview,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Сохранять Excel", variable=self.save_xlsx_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(options, text="Сохранять фрагменты колоний", variable=self.save_colony_crops_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        ttk.Label(controls, text="Вид предпросмотра").grid(row=6, column=0, sticky="w")
        stage_combo = ttk.Combobox(
            controls,
            textvariable=self.stage_var,
            values=["Аномалии", "Маски", "Crop 736", "bbox чашки", "Исходное"],
            state="readonly",
        )
        stage_combo.grid(row=7, column=0, sticky="ew", pady=(6, 8))
        stage_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_preview())

        self.run_button = ttk.Button(controls, text="Начать анализ  →", command=self.start_pipeline, style="Accent.TButton")
        self.run_button.grid(row=8, column=0, sticky="ew", pady=(8, 6))
        ttk.Button(controls, text="Открыть папку результата", command=self.open_output_dir).grid(row=9, column=0, sticky="ew")

        ttk.Progressbar(controls, variable=self.progress_var, maximum=100).grid(row=10, column=0, sticky="ew", pady=(14, 6))
        ttk.Label(controls, textvariable=self.status_var, wraplength=310, justify="left").grid(row=11, column=0, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Остановить после снимка", command=self.cancel_pipeline, state="disabled")
        self.stop_button.grid(row=12, column=0, sticky="ew", pady=(8, 0))

        main = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        main.grid(row=1, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        preview_frame = ttk.Frame(main)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(preview_frame, bg="#0b1220", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._draw_preview_image())

        table_frame = ttk.LabelFrame(main, text="Результаты", padding=8)
        table_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        table_frame.columnconfigure(0, weight=1)

        columns = ("image", "detected", "valid", "selected", "highlighted", "review", "technical", "score", "quality")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        headings = {
            "image": "Файл",
            "detected": "Колоний",
            "valid": "Валидных",
            "selected": "Аномалий",
            "highlighted": "Выделено",
            "review": "Review",
            "technical": "Тех.",
            "score": "Max score",
            "quality": "Качество",
        }
        widths = {
            "image": 220,
            "detected": 80,
            "valid": 80,
            "selected": 80,
            "highlighted": 80,
            "review": 70,
            "technical": 60,
            "score": 80,
            "quality": 150,
        }
        for col in columns:
            self.results_tree.heading(col, text=headings[col])
            self.results_tree.column(col, width=widths[col], anchor="w" if col in {"image", "quality"} else "center")
        self.results_tree.grid(row=0, column=0, sticky="ew")
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        tree_horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_tree.xview)
        tree_horizontal.grid(row=1, column=0, sticky="ew")
        self.results_tree.configure(yscrollcommand=tree_scroll.set, xscrollcommand=tree_horizontal.set)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_selected)

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        self._add_image_paths(Path(p) for p in paths)

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self._add_image_paths(collect_images_from_folder(folder))

    def remove_selected_files(self) -> None:
        selected = list(self.file_list.curselection())
        if not selected:
            return
        selected_paths = {self.image_paths[i] for i in selected}
        self.image_paths = [p for p in self.image_paths if p not in selected_paths]
        self._sync_file_list()

    def clear_files(self) -> None:
        self.image_paths.clear()
        self._sync_file_list()

    def choose_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку результата", initialdir=self.output_dir_var.get())
        if folder:
            self.output_dir_var.set(folder)

    def open_output_dir(self) -> None:
        output_dir = self.last_output_dir or Path(self.output_dir_var.get())
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(output_dir))
        except OSError as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку:\n{exc}")

    def _add_image_paths(self, paths) -> None:
        existing = {p.resolve() for p in self.image_paths if p.exists()}
        added = 0
        for path in paths:
            p = Path(path)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                resolved = p.resolve()
                if resolved not in existing:
                    self.image_paths.append(p)
                    existing.add(resolved)
                    added += 1
        self._sync_file_list()
        self.status_var.set(f"Добавлено изображений: {added}. Всего: {len(self.image_paths)}.")

    def _sync_file_list(self) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.image_paths:
            self.file_list.insert(tk.END, path.name)

    def start_pipeline(self) -> None:
        if self.running:
            return
        if not self.image_paths:
            messagebox.showwarning("Нет изображений", "Добавьте изображения или папку перед запуском.")
            return

        try:
            validate_image_batch(self.image_paths)
            output_dir = create_run_dir(Path(self.output_dir_var.get()))
        except (ValueError, OSError) as exc:
            messagebox.showerror("Не удалось начать анализ", str(exc))
            return
        self.last_output_dir = output_dir
        self.cancel_event.clear()
        self.stop_button.configure(state="normal")
        self.running = True
        self.run_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self.status_var.set("Загрузка моделей...")
        self.results.clear()
        self.summary_rows.clear()
        self.render_cache.clear()
        self.current_preview_key = None
        self.preview_rgb = None
        self.preview_canvas.delete("all")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        args = (
            list(self.image_paths),
            output_dir,
            bool(self.save_xlsx_var.get()),
            bool(self.save_colony_crops_var.get()),
        )
        self.worker_thread = threading.Thread(target=self._run_pipeline_worker, args=args, daemon=True)
        self.worker_thread.start()

    def _run_pipeline_worker(
        self,
        image_paths: list[Path],
        output_dir: Path,
        save_xlsx: bool,
        save_colony_crops: bool,
    ) -> None:
        started = time.time()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            config = build_runtime_config(output_dir, save_xlsx=save_xlsx, save_colony_crops=save_colony_crops)
            self.worker_queue.put(("status", "Загрузка YOLO моделей..."))
            segmentation_model, petri_detector_model = load_pipeline_models(config)

            summary_rows: list[dict[str, Any]] = []
            selected_tables: list[pd.DataFrame] = []
            visual_tables: list[pd.DataFrame] = []
            review_tables: list[pd.DataFrame] = []
            technical_tables: list[pd.DataFrame] = []

            total = len(image_paths)
            for index, image_path in enumerate(image_paths, start=1):
                if self.cancel_event.is_set():
                    break
                self.worker_queue.put(("status", f"Обработка {index}/{total}: {image_path.name}"))
                result = run_single_image_pipeline(
                    image_path=image_path,
                    model=segmentation_model,
                    config=config,
                    petri_detector_model=petri_detector_model,
                )
                fig = result.get("figure")
                if fig is not None:
                    plt.close(fig)

                image_output_dir = output_dir / image_path.stem
                save_stage_images(result, image_output_dir)
                row = result_summary_row(image_path, result, image_output_dir)
                summary_rows.append(row)
                self._collect_combined_table(result, image_path.name, "selected", selected_tables)
                self._collect_combined_table(result, image_path.name, "visual_highlights", visual_tables)
                self._collect_combined_table(result, image_path.name, "review_candidates", review_tables)
                self._collect_combined_table(result, image_path.name, "technical_warnings", technical_tables)
                self.worker_queue.put(("result", (str(image_path), result, row)))
                self.worker_queue.put(("progress", 100.0 * index / total))

            summary_df = pd.DataFrame(summary_rows)
            safe_table(summary_df).to_csv(output_dir / "pipeline_summary.csv", index=False, encoding="utf-8-sig")
            self._save_combined_table(selected_tables, output_dir / "all_selected_anomalies.csv")
            self._save_combined_table(visual_tables, output_dir / "all_visual_highlighted_objects.csv")
            self._save_combined_table(review_tables, output_dir / "all_review_candidates.csv")
            self._save_combined_table(technical_tables, output_dir / "all_technical_warnings.csv")
            if save_xlsx:
                self._save_summary_excel(output_dir, summary_df, selected_tables, visual_tables, review_tables, technical_tables)

            elapsed = time.time() - started
            state = "Остановлено" if self.cancel_event.is_set() else "Готово"
            self.worker_queue.put(("done", f"{state}: {len(summary_rows)} изображений за {elapsed:.1f} сек."))
        except Exception as exc:
            details = traceback.format_exc()
            self.worker_queue.put(("error", f"{exc}\n\n{details}"))

    @staticmethod
    def _collect_combined_table(result: dict[str, Any], image_name: str, key: str, target: list[pd.DataFrame]) -> None:
        df = result.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            tmp = df.copy()
            tmp.insert(0, "image_name", image_name)
            target.append(tmp)

    @staticmethod
    def _save_combined_table(tables: list[pd.DataFrame], path: Path) -> None:
        df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        safe_table(df).to_csv(path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _save_summary_excel(
        output_dir: Path,
        summary_df: pd.DataFrame,
        selected_tables: list[pd.DataFrame],
        visual_tables: list[pd.DataFrame],
        review_tables: list[pd.DataFrame],
        technical_tables: list[pd.DataFrame],
    ) -> None:
        selected_df = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
        visual_df = pd.concat(visual_tables, ignore_index=True) if visual_tables else pd.DataFrame()
        review_df = pd.concat(review_tables, ignore_index=True) if review_tables else pd.DataFrame()
        technical_df = pd.concat(technical_tables, ignore_index=True) if technical_tables else pd.DataFrame()
        with pd.ExcelWriter(output_dir / "pipeline_report.xlsx") as writer:
            safe_table(summary_df).to_excel(writer, sheet_name="summary", index=False)
            safe_table(selected_df).to_excel(writer, sheet_name="selected_anomalies", index=False)
            safe_table(visual_df).to_excel(writer, sheet_name="visual_highlights", index=False)
            safe_table(review_df).to_excel(writer, sheet_name="review_candidates", index=False)
            safe_table(technical_df).to_excel(writer, sheet_name="technical_warnings", index=False)

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "status":
                    self.status_var.set(str(payload))
                elif event == "progress":
                    self.progress_var.set(float(payload))
                elif event == "result":
                    image_key, result, row = payload
                    self.results[image_key] = result
                    self.summary_rows.append(row)
                    self._insert_result_row(image_key, row)
                    if self.current_preview_key is None:
                        self.current_preview_key = image_key
                        self.refresh_preview()
                elif event == "done":
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.progress_var.set(100.0)
                    self.status_var.set(str(payload))
                elif event == "error":
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_var.set("Ошибка выполнения.")
                    messagebox.showerror("Ошибка пайплайна", str(payload))
        except queue.Empty:
            pass
        self.root.after(150, self._poll_worker_queue)

    def _insert_result_row(self, image_key: str, row: dict[str, Any]) -> None:
        max_score = row.get("max_anomaly_score_raw", np.nan)
        score_text = f"{max_score:.3f}" if np.isfinite(max_score) else ""
        values = (
            row.get("image_name", Path(image_key).name),
            row.get("n_detected_colonies", 0),
            row.get("n_valid_colonies", 0),
            row.get("n_selected_anomalies", 0),
            row.get("n_visual_highlights", 0),
            row.get("n_review_candidates", 0),
            row.get("n_technical_warnings", 0),
            score_text,
            row.get("plate_quality_status", ""),
        )
        self.results_tree.insert("", tk.END, iid=image_key, values=values)

    def _on_result_selected(self, _event) -> None:
        selection = self.results_tree.selection()
        if selection:
            self.current_preview_key = selection[0]
            self.refresh_preview()

    def refresh_preview(self) -> None:
        if not self.current_preview_key or self.current_preview_key not in self.results:
            return
        stage = self.stage_var.get()
        show_scores = bool(self.show_scores_var.get())
        cache_key = (self.current_preview_key, stage, show_scores)
        if cache_key in self.render_cache:
            self.preview_rgb = self.render_cache[cache_key]
            self._draw_preview_image()
            return

        result = self.results[self.current_preview_key]
        try:
            if stage == "Исходное":
                image = result["image_original"]
            elif stage == "bbox чашки":
                image = draw_petri_bbox(result["image_original"], result["preprocess"])
            elif stage == "Crop 736":
                image = result["image"]
            elif stage == "Маски":
                image = overlay_instance_masks(result["image"], result["masks"])
            else:
                image = render_anomaly_overlay(result, show_scores=show_scores)
            self.render_cache[cache_key] = image
            self.preview_rgb = image
            self._draw_preview_image()
        except Exception as exc:
            messagebox.showerror("Ошибка визуализации", str(exc))

    def _draw_preview_image(self) -> None:
        self.preview_canvas.delete("all")
        if self.preview_rgb is None:
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text="Ваш следующий результат начинается здесь\n\nДобавьте снимки чашек Петри и нажмите «Начать анализ»",
                fill="#94a9c2", font=("Segoe UI", 14), justify="center", width=max(200, self.preview_canvas.winfo_width() - 64),
            )
            return

        canvas_w = max(1, self.preview_canvas.winfo_width())
        canvas_h = max(1, self.preview_canvas.winfo_height())
        image = Image.fromarray(self.preview_rgb)
        img_w, img_h = image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.preview_canvas.create_image(x, y, anchor="nw", image=self.preview_photo)


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    ColonyPipelineApp(root)
    root.mainloop()


def self_test() -> int:
    config = build_runtime_config(default_output_dir() / "_self_test", save_xlsx=False, save_colony_crops=False)
    missing = [
        path
        for path in [Path(config.petri_detector_weights_path), Path(config.model_weights_path or "")]
        if not path.exists()
    ]
    if missing:
        print("Missing model files:")
        for path in missing:
            print(path)
        return 2
    print("ColonyNetPipeline self-test OK")
    print("Petri detector:", config.petri_detector_weights_path)
    print("Colony segmenter:", config.model_weights_path)
    print("Output dir:", config.output_dir)
    return 0


def run_once_cli(argv: list[str]) -> int:
    output_dir = default_output_dir() / "run_once"
    if "--output" in argv:
        output_index = argv.index("--output")
        if output_index + 1 >= len(argv):
            print("Missing value after --output")
            return 2
        output_dir = Path(argv[output_index + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_once.log"

    def log(message: str) -> None:
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(text)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    try:
        image_index = argv.index("--run-once")
        if image_index + 1 >= len(argv):
            log("Usage: ColonyNetPipeline.exe --run-once IMAGE_PATH [--output OUTPUT_DIR]")
            return 2

        image_path = Path(argv[image_index + 1])
        if not image_path.exists():
            log(f"Image not found: {image_path}")
            return 2

        log(f"Start run-once for {image_path}")
        config = build_runtime_config(
            output_dir=output_dir,
            save_xlsx="--no-xlsx" not in argv,
            save_colony_crops="--save-crops" in argv,
        )
        log("Loading models")
        segmentation_model, petri_detector_model = load_pipeline_models(config)
        log("Running image pipeline")
        result = run_single_image_pipeline(
            image_path=image_path,
            model=segmentation_model,
            config=config,
            petri_detector_model=petri_detector_model,
        )
        fig = result.get("figure")
        if fig is not None:
            plt.close(fig)

        log("Saving stage images")
        image_output_dir = output_dir / image_path.stem
        save_stage_images(result, image_output_dir)
        summary = result_summary_row(image_path, result, image_output_dir)
        pd.DataFrame([summary]).to_csv(output_dir / "pipeline_summary.csv", index=False, encoding="utf-8-sig")
        log(f"Run-once OK: {summary}")
        return 0
    except Exception:
        log("Run-once failed")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    if "--run-once" in sys.argv:
        raise SystemExit(run_once_cli(sys.argv))
    main()

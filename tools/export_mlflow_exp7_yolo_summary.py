"""Export YOLO n/s/m/l/x metrics from MLflow experiment 7."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


MODEL_ORDER = ["n", "s", "m", "l", "x"]

TRAIN_METRICS = {
    "metrics/precisionB": "train_val_box_precision",
    "metrics/recallB": "train_val_box_recall",
    "metrics/mAP50B": "train_val_box_mAP50",
    "metrics/mAP50-95B": "train_val_box_mAP50_95",
    "metrics/precisionM": "train_val_mask_precision",
    "metrics/recallM": "train_val_mask_recall",
    "metrics/mAP50M": "train_val_mask_mAP50",
    "metrics/mAP50-95M": "train_val_mask_mAP50_95",
}

TEST_METRICS = {
    "precision_B": "test_box_precision",
    "recall_B": "test_box_recall",
    "mAP50_B": "test_box_mAP50",
    "mAP50_95_B": "test_box_mAP50_95",
    "precision_M": "test_mask_precision",
    "recall_M": "test_mask_recall",
    "mAP50_M": "test_mask_mAP50",
    "mAP50_95_M": "test_mask_mAP50_95",
    "fitness": "test_fitness",
    "dice_M_mean": "test_dice_mask_mean",
    "dice_M_median": "test_dice_mask_median",
    "mae_count": "test_mae_count",
    "rmse_count": "test_rmse_count",
    "mape_count_nonzero": "test_mape_count_nonzero",
    "test_images_eval": "test_images_eval",
    "test_images_missing_labels": "test_images_missing_labels",
}


def latest_metric(metrics_df: pd.DataFrame, run_id: str, key: str) -> float | None:
    rows = metrics_df[(metrics_df["run_uuid"] == run_id) & (metrics_df["key"] == key)]
    if rows.empty:
        return None
    rows = rows.sort_values(["step", "timestamp"], ascending=[False, False])
    return float(rows.iloc[0]["value"])


def first_param(params_df: pd.DataFrame, run_id: str, key: str) -> str | None:
    rows = params_df[(params_df["run_uuid"] == run_id) & (params_df["key"] == key)]
    if rows.empty:
        return None
    return str(rows.iloc[0]["value"])


def build_summary(db_path: Path, experiment_id: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        runs_df = pd.read_sql_query(
            """
            select
                run_uuid,
                name,
                status,
                start_time,
                end_time,
                artifact_uri
            from runs
            where experiment_id = ?
            """,
            con,
            params=(experiment_id,),
        )
        metrics_df = pd.read_sql_query(
            """
            select m.run_uuid, m.key, m.value, m.step, m.timestamp
            from metrics m
            join runs r on r.run_uuid = m.run_uuid
            where r.experiment_id = ?
            """,
            con,
            params=(experiment_id,),
        )
        params_df = pd.read_sql_query(
            """
            select p.run_uuid, p.key, p.value
            from params p
            join runs r on r.run_uuid = p.run_uuid
            where r.experiment_id = ?
            """,
            con,
            params=(experiment_id,),
        )

    rows = []
    for short in MODEL_ORDER:
        model_name = f"yolo26{short}-seg"
        train_name = f"{model_name}_cropped736_offline_aug"
        train_runs = runs_df[
            (runs_df["name"] == train_name)
            & (runs_df["status"] == "FINISHED")
        ].sort_values("start_time", ascending=False)

        train_run = train_runs.iloc[0] if not train_runs.empty else None

        eval_candidates = []
        for _, run in runs_df[runs_df["status"] == "FINISHED"].iterrows():
            best_weights = first_param(params_df, str(run["run_uuid"]), "best_weights")
            if best_weights and train_name in best_weights:
                eval_candidates.append(run)
        eval_run = None
        if eval_candidates:
            eval_df = pd.DataFrame(eval_candidates).sort_values("start_time", ascending=False)
            eval_run = eval_df.iloc[0]

        row: dict[str, object] = {
            "model": f"{model_name}.pt",
            "model_short": f"yolo26{short}",
            "mlflow_experiment_id": experiment_id,
            "train_run_name": train_name,
            "train_run_id": str(train_run["run_uuid"]) if train_run is not None else None,
            "train_run_status": str(train_run["status"]) if train_run is not None else None,
            "test_run_name": str(eval_run["name"]) if eval_run is not None else None,
            "test_run_id": str(eval_run["run_uuid"]) if eval_run is not None else None,
        }

        if train_run is not None:
            train_id = str(train_run["run_uuid"])
            for key, out_col in TRAIN_METRICS.items():
                row[out_col] = latest_metric(metrics_df, train_id, key)

        if eval_run is not None:
            eval_id = str(eval_run["run_uuid"])
            row["best_weights"] = first_param(params_df, eval_id, "best_weights")
            row["ultralytics_train_dir"] = first_param(params_df, eval_id, "ultralytics_train_dir")
            row["ultralytics_test_dir"] = first_param(params_df, eval_id, "ultralytics_test_dir")
            for key, out_col in TEST_METRICS.items():
                row[out_col] = latest_metric(metrics_df, eval_id, key)

        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = [
        "model",
        "model_short",
        "mlflow_experiment_id",
        "train_run_name",
        "train_run_id",
        "train_run_status",
        "test_run_name",
        "test_run_id",
        "train_val_mask_mAP50_95",
        "train_val_mask_mAP50",
        "train_val_mask_precision",
        "train_val_mask_recall",
        "train_val_box_mAP50_95",
        "train_val_box_mAP50",
        "train_val_box_precision",
        "train_val_box_recall",
        "test_mask_mAP50_95",
        "test_mask_mAP50",
        "test_mask_precision",
        "test_mask_recall",
        "test_box_mAP50_95",
        "test_box_mAP50",
        "test_box_precision",
        "test_box_recall",
        "test_fitness",
        "test_dice_mask_mean",
        "test_dice_mask_median",
        "test_mae_count",
        "test_rmse_count",
        "test_mape_count_nonzero",
        "test_images_eval",
        "test_images_missing_labels",
        "best_weights",
        "ultralytics_train_dir",
        "ultralytics_test_dir",
    ]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--experiment-id", type=int, default=7)
    parser.add_argument("--out-csv", type=Path, default=Path("mlflow_experiment_7_yolo_n_to_x_summary.csv"))
    parser.add_argument("--out-xlsx", type=Path, default=Path("mlflow_experiment_7_yolo_n_to_x_summary.xlsx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_summary(args.db, args.experiment_id)
    df.to_csv(args.out_csv, index=False)
    try:
        df.to_excel(args.out_xlsx, index=False)
    except Exception as exc:
        print(f"Skipped xlsx export: {exc}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved XLSX: {args.out_xlsx}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

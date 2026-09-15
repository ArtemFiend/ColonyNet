"""Local inference defaults and safe filesystem helpers (no ML imports)."""
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


def configure_offline() -> None:
    # Must run before importing Ultralytics: even its import probes connectivity.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    # Keep third-party settings out of the user's global profile. This directory
    # contains library preferences only, never input images or analysis results.
    config_dir = Path(os.environ.get("COLONYNET_CONFIG_DIR", Path(tempfile.gettempdir()) / "ColonyNet" / "config"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def create_run_dir(parent: Path) -> Path:
    path = parent / f"analysis_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_image_batch(paths: list[Path]) -> None:
    # The pipeline names output subdirectories after stems. Reject ambiguity
    # before inference rather than silently overwrite another sample's results.
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Изображение не найдено: {path.name}")
        key = path.stem.casefold()
        if key in seen:
            raise ValueError(f"Повторяющееся имя {path.stem}: переименуйте снимки перед анализом.")
        seen.add(key)


def safe_table(df):
    """Neutralize spreadsheet formulas in untrusted text, preserving numbers."""
    def escape(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
            return "'" + value
        return value
    result = df.copy()
    for column in result.columns:
        if result[column].dtype == object or str(result[column].dtype).startswith("string"):
            result[column] = result[column].map(escape)
    return result

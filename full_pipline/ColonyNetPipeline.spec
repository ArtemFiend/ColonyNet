# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


SPEC_DIR = Path(SPECPATH).resolve()
if SPEC_DIR.is_file():
    SPEC_DIR = SPEC_DIR.parent
ROOT = SPEC_DIR if (SPEC_DIR / "full_pipline").exists() else SPEC_DIR.parent
APP = ROOT / "full_pipline" / "colony_pipeline_app.py"
MODELS = ROOT / "full_pipline" / "models"


datas = []
binaries = []
hiddenimports = [
    "PIL._tkinter_finder",
    "matplotlib.backends.backend_agg",
    "openpyxl",
    "_tkinter",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
    "full_pipline",
    "full_pipline.full_pipeline",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree._utils",
    "torchvision",
    "ultralytics",
    "ultralytics.cfg",
    "ultralytics.engine.model",
    "ultralytics.engine.predictor",
    "ultralytics.engine.results",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.models.yolo.segment",
    "ultralytics.models.yolo.segment.predict",
    "ultralytics.models.yolo.segment.val",
    "ultralytics.nn.tasks",
    "ultralytics.utils",
    "ultralytics.utils.ops",
]

for model_path in MODELS.glob("*.pt"):
    datas.append((str(model_path), "full_pipline/models"))

PYTHON_PREFIX = Path(sys.base_prefix)
TCL_DIR = PYTHON_PREFIX / "tcl" / "tcl8.6"
TK_DIR = PYTHON_PREFIX / "tcl" / "tk8.6"
TKINTER_DIR = PYTHON_PREFIX / "Lib" / "tkinter"
if TCL_DIR.exists():
    datas.append((str(TCL_DIR), "_tcl_data"))
if TK_DIR.exists():
    datas.append((str(TK_DIR), "_tk_data"))
if TKINTER_DIR.exists():
    datas.append((str(TKINTER_DIR), "tkinter"))

datas += collect_data_files("ultralytics")

for package_name in ["sklearn", "skimage"]:
    hiddenimports += collect_submodules(package_name)

for distribution_name in [
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "scipy",
    "matplotlib",
    "openpyxl",
]:
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        pass


a = Analysis(
    [str(APP)],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "notebook",
        "IPython",
        "jupyter",
        "jupyterlab",
        "tensorboard",
        "tensorboard_data_server",
        "tensorflow",
        "tensorflow_intel",
        "tensorflow_estimator",
        "keras",
        "ml_dtypes",
        "jax",
        "jaxlib",
        "pyarrow",
        "polars",
        "numba",
        "llvmlite",
        "onnx",
        "onnxruntime",
        "openvino",
        "coremltools",
        "tflite_runtime",
        "wandb",
        "mlflow",
        "ultralytics.trackers",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ColonyNetPipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ColonyNetPipeline",
)

# MLflow Model Configs

This folder contains ready-to-run configs grouped by model:
- `mit_b2` (1-stage + 2-stage)
- `mit_b3` (1-stage + 2-stage)
- `unetpp` (1-stage + 2-stage notebooks)

## Structure
- `mit_b2/one_stage.yaml` - train config for `train.py` (MLflow enabled)
- `mit_b2/stage1_pretrain.yaml` - stage 1 config for 2-stage
- `mit_b2/stage2_finetune.yaml` - stage 2 config for 2-stage
- `mit_b2/one_stage_pipeline.yaml` - pipeline to run 1-stage via `tools/run_pipeline.py`
- `mit_b2/two_stage_pipeline.yaml` - pipeline to run stage1 -> stage2

Same structure is provided for `mit_b3`.

For `unetpp`:
- `unetpp/one_stage_pipeline.yaml` - executes `train_unified_unetpp.ipynb`
- `unetpp/two_stage_pipeline.yaml` - executes `train_two_stage_unetpp.ipynb`

## Quick Start
Start MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

Run one model:
```bash
python tools/run_pipeline.py --pipeline configs/mlflow_models/mit_b3/one_stage_pipeline.yaml
python tools/run_pipeline.py --pipeline configs/mlflow_models/mit_b3/two_stage_pipeline.yaml
```

Run all models from this folder:
```bash
python tools/run_pipeline.py --pipeline configs/mlflow_models/run_all.yaml --continue_on_error
```

Recommended on Windows/Conda:
```bash
python tools/run_pipeline.py --pipeline run_all --python C:\\ColonyNet\\.venv\\Scripts\\python.exe --mlflow --mlflow_tracking_uri http://127.0.0.1:5000 --mlflow_experiment colony_models --continue_on_error
```

Notes:
- `mit_*` configs log to MLflow through `train.py`.
- `unetpp` notebooks include MLflow logging cells (using `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `MLFLOW_RUN_NAME` env vars).
- `unetpp` notebooks also auto-install missing kernel deps (`opencv-python`, `scikit-image`, `segmentation-models-pytorch`, `timm`) before training.
- Pipeline runner creates a temporary Jupyter kernel bound to `--python` and prepends its directory to `PATH` for notebook stages, preventing accidental Anaconda kernel imports.

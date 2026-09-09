from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_pipeline_path(pipeline_arg: str, repo_root: Path) -> Path:
    p = Path(pipeline_arg)
    if not p.is_absolute():
        p = repo_root / p
    if p.exists():
        return p

    by_name = repo_root / "configs" / "pipelines" / f"{pipeline_arg}.yaml"
    if by_name.exists():
        return by_name

    by_mlflow_name = repo_root / "configs" / "mlflow_models" / f"{pipeline_arg}.yaml"
    if by_mlflow_name.exists():
        return by_mlflow_name

    raise FileNotFoundError(
        f"Pipeline not found: '{pipeline_arg}'. "
        f"Tried '{p}', '{by_name}', and '{by_mlflow_name}'."
    )


def resolve_path(value: str, base_dir: Path, repo_root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    p1 = (base_dir / p).resolve()
    if p1.exists():
        return p1
    return (repo_root / p).resolve()


def _sanitize_kernel_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", name.strip())
    s = s.strip("_").lower()
    return s or "pipeline_kernel"


def ensure_temp_kernel(args_python: str, pipeline_out_dir: Path, stage_name: str) -> tuple[str, Path]:
    """
    Create a temporary kernelspec bound to the selected interpreter.

    Returns:
      (kernel_name, jupyter_data_root)
      where jupyter_data_root contains `kernels/<kernel_name>/kernel.json`.
    """
    kernel_name = f"pipeline_{_sanitize_kernel_name(stage_name)}"
    py = str(Path(args_python).resolve())

    jupyter_data_root = (pipeline_out_dir / "jupyter_data").resolve()
    kernel_dir = jupyter_data_root / "kernels" / kernel_name
    ensure_dir(kernel_dir)

    kernel_json = {
        "argv": [py, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"Pipeline ({Path(py).name})",
        "language": "python",
    }
    with open(kernel_dir / "kernel.json", "w", encoding="utf-8") as f:
        json.dump(kernel_json, f, ensure_ascii=False, indent=2)

    return kernel_name, jupyter_data_root


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def run_command(
    cmd: list[str],
    cwd: Path,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> int:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"  $ {cmd_str}")
    if dry_run:
        return 0
    env = os.environ.copy()
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                continue
            env[str(k)] = str(v)
    completed = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    return int(completed.returncode)


def stage_train_config(
    stage: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
    pipeline_dir: Path,
    tmp_cfg_dir: Path,
    stage_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stage_name = str(stage["name"])
    cfg_src = resolve_path(str(stage["config"]), pipeline_dir, repo_root)
    cfg = load_yaml(cfg_src)

    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    if not isinstance(cfg.get("train"), dict):
        cfg["train"] = {}

    # Global model overrides from CLI
    if args.model_variant:
        cfg["model"]["model_variant"] = str(args.model_variant)
    if args.backbone_id:
        cfg["model"]["backbone_id"] = str(args.backbone_id)

    # Stage-level model overrides
    if "model_variant" in stage:
        cfg["model"]["model_variant"] = str(stage["model_variant"])
    if "backbone_id" in stage:
        cfg["model"]["backbone_id"] = str(stage["backbone_id"])

    # Optional ckpt chaining from previous stage
    init_from_stage = stage.get("init_from_stage")
    if init_from_stage:
        if init_from_stage not in stage_outputs:
            raise RuntimeError(
                f"Stage '{stage_name}' references unknown init_from_stage '{init_from_stage}'."
            )
        src = stage_outputs[init_from_stage]
        candidates = []
        if src.get("best_ckpt"):
            candidates.append(str(src["best_ckpt"]))
        if src.get("last_ckpt"):
            candidates.append(str(src["last_ckpt"]))
        if not candidates:
            raise RuntimeError(
                f"Stage '{stage_name}' could not resolve checkpoint from '{init_from_stage}'."
            )

        init_ckpt = candidates[0]
        if not args.dry_run:
            existing = [p for p in candidates if Path(p).exists()]
            if existing:
                init_ckpt = existing[0]
            elif src.get("status") == "ok":
                raise RuntimeError(
                    f"Stage '{stage_name}' expected checkpoint from '{init_from_stage}', "
                    f"but files do not exist: {candidates}"
                )

        cfg["train"]["init_ckpt"] = init_ckpt

    # Run name controls
    base_run_name = str(cfg.get("run_name", stage_name))
    if args.run_name_suffix:
        base_run_name = f"{base_run_name}_{args.run_name_suffix}"
    if stage.get("run_name_suffix"):
        base_run_name = f"{base_run_name}_{stage['run_name_suffix']}"
    if stage.get("run_name_prefix"):
        base_run_name = f"{stage['run_name_prefix']}_{base_run_name}"
    if stage.get("run_name"):
        base_run_name = str(stage["run_name"])
    cfg["run_name"] = base_run_name

    for dotted_key, val in (stage.get("set") or {}).items():
        set_nested(cfg, str(dotted_key), val)

    tmp_cfg = tmp_cfg_dir / f"{stage_name}.yaml"
    save_yaml(tmp_cfg, cfg)

    cmd = [
        args.python,
        "train.py",
        "--config",
        str(tmp_cfg),
        "--runs_dir",
        str(args.runs_dir),
    ]
    if args.mlflow:
        cmd.append("--mlflow")
    if args.no_mlflow:
        cmd.append("--no_mlflow")
    if args.mlflow_tracking_uri:
        cmd.extend(["--mlflow_tracking_uri", args.mlflow_tracking_uri])
    if args.mlflow_experiment:
        cmd.extend(["--mlflow_experiment", args.mlflow_experiment])
    if args.mlflow_run_name:
        cmd.extend(["--mlflow_run_name", args.mlflow_run_name])
    if args.model_variant:
        cmd.extend(["--model_variant", args.model_variant])
    if args.backbone_id:
        cmd.extend(["--backbone_id", args.backbone_id])
    for extra in (stage.get("extra_args") or []):
        cmd.append(str(extra))

    t0 = time.time()
    rc = run_command(cmd=cmd, cwd=repo_root, dry_run=args.dry_run)
    elapsed = time.time() - t0

    run_dir = (repo_root / args.runs_dir / cfg["run_name"]).resolve()
    best_ckpt = run_dir / "best.pt"
    last_ckpt = run_dir / "last.pt"

    return {
        "type": "train_config",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": elapsed,
        "source_config": str(cfg_src),
        "effective_config": str(tmp_cfg),
        "run_name": cfg["run_name"],
        "run_dir": str(run_dir),
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "best_ckpt_exists": bool(best_ckpt.exists()),
        "last_ckpt_exists": bool(last_ckpt.exists()),
    }


def stage_notebook(
    stage: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
    pipeline_dir: Path,
    pipeline_out_dir: Path,
) -> dict[str, Any]:
    stage_name = str(stage["name"])
    nb_path = resolve_path(str(stage["notebook"]), pipeline_dir, repo_root)
    out_dir = pipeline_out_dir / "notebooks"
    ensure_dir(out_dir)

    output_name = str(stage.get("output_notebook", f"{nb_path.stem}.executed.ipynb"))
    timeout = int(stage.get("timeout_sec", 0))
    requested_kernel_name = stage.get("kernel_name", None)

    cmd = [
        args.python,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb_path),
        "--output",
        output_name,
        "--output-dir",
        str(out_dir),
    ]
    if timeout > 0:
        cmd.append(f"--ExecutePreprocessor.timeout={timeout}")

    env_overrides: dict[str, str] = {}
    # Important on Windows/Conda setups:
    # many kernelspecs use "python" (without absolute path), so we ensure
    # PATH starts with the interpreter directory selected by --python.
    py_dir = str(Path(args.python).resolve().parent)
    path_sep = os.pathsep
    env_overrides["PATH"] = py_dir + path_sep + os.environ.get("PATH", "")
    env_overrides["PYTHONNOUSERSITE"] = "1"

    if requested_kernel_name:
        kernel_name = str(requested_kernel_name)
    else:
        # Force notebook execution inside selected interpreter even when global
        # `python3` kernelspec points to Anaconda/system Python.
        kernel_name, jupyter_data_root = ensure_temp_kernel(
            args_python=args.python,
            pipeline_out_dir=pipeline_out_dir,
            stage_name=stage_name,
        )
        env_overrides["JUPYTER_PATH"] = str(jupyter_data_root) + path_sep + os.environ.get(
            "JUPYTER_PATH", ""
        )

    cmd.append(f"--ExecutePreprocessor.kernel_name={kernel_name}")

    if args.mlflow_tracking_uri:
        env_overrides["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    if args.mlflow_experiment:
        env_overrides["MLFLOW_EXPERIMENT"] = args.mlflow_experiment
    if args.mlflow_run_name:
        env_overrides["MLFLOW_RUN_NAME"] = args.mlflow_run_name

    t0 = time.time()
    rc = run_command(
        cmd=cmd,
        cwd=repo_root,
        dry_run=args.dry_run,
        env_overrides=env_overrides,
    )
    elapsed = time.time() - t0

    out_nb = out_dir / output_name
    return {
        "type": "notebook",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": elapsed,
        "notebook": str(nb_path),
        "output_notebook": str(out_nb),
    }


def stage_command(
    stage: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    stage_name = str(stage["name"])
    raw_cmd = stage.get("command")
    if not isinstance(raw_cmd, list) or not raw_cmd:
        raise ValueError(f"Stage '{stage_name}': command must be a non-empty list.")
    cmd = [str(x) for x in raw_cmd]

    t0 = time.time()
    rc = run_command(cmd=cmd, cwd=repo_root, dry_run=args.dry_run)
    elapsed = time.time() - t0
    return {
        "type": "command",
        "status": "ok" if rc == 0 else "failed",
        "returncode": rc,
        "elapsed_sec": elapsed,
        "command": cmd,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run multi-stage training pipelines.")
    ap.add_argument("--pipeline", required=True, help="Pipeline name or path to YAML")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--run_name_suffix", default=None)
    ap.add_argument("--model_variant", default=None)
    ap.add_argument("--backbone_id", default=None)

    # Forwarded to train.py stages:
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--no_mlflow", action="store_true")
    ap.add_argument("--mlflow_tracking_uri", default=None)
    ap.add_argument("--mlflow_experiment", default=None)
    ap.add_argument("--mlflow_run_name", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pipeline_path = resolve_pipeline_path(args.pipeline, repo_root)
    spec = load_yaml(pipeline_path)
    stages = spec.get("stages", None)
    if not isinstance(stages, list) or not stages:
        raise ValueError("Pipeline YAML must contain non-empty list: stages")

    pipeline_name = str(spec.get("name", pipeline_path.stem))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    pipeline_out_dir = (repo_root / args.runs_dir / "pipelines" / f"{pipeline_name}_{run_id}").resolve()
    tmp_cfg_dir = pipeline_out_dir / "tmp_configs"
    ensure_dir(tmp_cfg_dir)

    pipeline_dir = pipeline_path.parent
    stage_outputs: dict[str, dict[str, Any]] = {}
    failed = False

    print(f"Pipeline: {pipeline_name}")
    print(f"Spec: {pipeline_path}")
    print(f"Out: {pipeline_out_dir}")

    total_t0 = time.time()
    for idx, stage_raw in enumerate(stages, start=1):
        if not isinstance(stage_raw, dict):
            raise ValueError(f"Stage #{idx} must be mapping.")
        stage = copy.deepcopy(stage_raw)
        stage_name = str(stage.get("name", f"stage_{idx:02d}"))
        stage["name"] = stage_name
        stage_type = str(stage.get("type", "train_config"))

        print(f"\n[{idx}/{len(stages)}] {stage_name} ({stage_type})")

        try:
            if stage_type == "train_config":
                out = stage_train_config(
                    stage=stage,
                    args=args,
                    repo_root=repo_root,
                    pipeline_dir=pipeline_dir,
                    tmp_cfg_dir=tmp_cfg_dir,
                    stage_outputs=stage_outputs,
                )
            elif stage_type == "notebook":
                out = stage_notebook(
                    stage=stage,
                    args=args,
                    repo_root=repo_root,
                    pipeline_dir=pipeline_dir,
                    pipeline_out_dir=pipeline_out_dir,
                )
            elif stage_type == "command":
                out = stage_command(
                    stage=stage,
                    args=args,
                    repo_root=repo_root,
                )
            else:
                raise ValueError(f"Unsupported stage type: {stage_type}")
        except Exception as exc:
            out = {
                "type": stage_type,
                "status": "failed",
                "returncode": 1,
                "error": str(exc),
            }

        stage_outputs[stage_name] = out
        print(f"  status: {out.get('status')}")
        if out.get("returncode", 0) != 0:
            failed = True
            if not args.continue_on_error:
                print("Stopping pipeline due to failure.")
                break

    total_elapsed = time.time() - total_t0
    summary = {
        "pipeline_name": pipeline_name,
        "pipeline_path": str(pipeline_path),
        "run_id": run_id,
        "started_at": run_id,
        "elapsed_sec": total_elapsed,
        "status": "failed" if failed else "ok",
        "stages": stage_outputs,
    }
    ensure_dir(pipeline_out_dir)
    summary_path = pipeline_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

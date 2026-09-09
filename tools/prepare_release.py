"""Export an allowlisted source tree with notebook outputs removed.

Never stages the user's working tree or copies datasets, checkpoints or logs.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".ps1", ".spec", ".ipynb", ".svg"}
SOURCE_DIRS = ("colonyseg", "configs", "full_pipline", "tools", "tests", "docs", ".github")
SECRET_PATTERNS = (
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{40,}",
    r"hf_[A-Za-z0-9]{30,}",
    r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}",
    r"xox[baprs]-[0-9A-Za-z-]{20,}",
    r"glpat-[A-Za-z0-9_-]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"https?://[^\s/:]+:[^\s/@]+@",
)


def sanitize_notebook(text: str) -> str:
    notebook = json.loads(text)
    notebook["metadata"] = {key: value for key, value in notebook.get("metadata", {}).items() if key in {"kernelspec", "language_info"}}
    for cell in notebook.get("cells", []):
        cell["metadata"] = {}
        cell.pop("attachments", None)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def source_files(root: Path):
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode("utf-8").split("\0")
    paths = {root / name for name in tracked if name}
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES)
    for directory in SOURCE_DIRS:
        paths.update((root / directory).rglob("*"))
    paths.update(root / name for name in (".gitignore", ".gitattributes"))
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in {"__pycache__", "models", "outputs", ".ipynb_checkpoints"} for part in relative.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", ".gitattributes"}:
            continue
        if "_executed" in path.stem or " copy" in path.stem or path.name in {"test_mlflow.py", "mlflow_metrics_report.md", "colony_anomaly_features_algorithm.md"}:
            continue
        yield path


def export(destination: Path) -> int:
    destination = destination.resolve()
    if destination == ROOT or ROOT not in destination.parents:
        raise ValueError("Destination must be a new subdirectory of the project.")
    if destination.exists():
        raise ValueError("Destination already exists; choose a new directory.")
    prepared = []
    findings = []
    for path in source_files(ROOT):
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".ipynb":
            text = sanitize_notebook(text)
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                findings.append(f"{path.relative_to(ROOT)}: possible credential (value withheld)")
        if re.search(r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]{100,}", text):
            findings.append(f"{path.relative_to(ROOT)}: embedded image")
        prepared.append((path.relative_to(ROOT), text))
    if findings:
        raise ValueError("Review before publication:\n" + "\n".join(findings))
    for relative, text in prepared:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    print(f"Exported {len(prepared)} source files. Notebook outputs and attachments excluded.")
    return len(prepared)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.destination)

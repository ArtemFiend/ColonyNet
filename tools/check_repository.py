"""Check tracked publication files, without printing potentially secret values."""
import json
import argparse
from pathlib import Path
import re
import subprocess
import sys

try:
    from tools.prepare_release import SECRET_PATTERNS
except ImportError:
    from prepare_release import SECRET_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".safetensors", ".db", ".sqlite", ".zip", ".exe", ".dll", ".log", ".xlsx", ".docx"}
FORBIDDEN_DIRS = {".venv", ".venv-app", "data", "all_datasets", "trainable_pool", "runs", "outputs", "build", "dist", "mlartifacts", "mlruns"}


def check_file(path: Path, relative: Path) -> list[str]:
    findings = []
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or relative.parts[0] in FORBIDDEN_DIRS:
        return ["private/generated artifact"]
    if path.name.startswith(".env") and path.name != ".env.example":
        return ["environment file"]
    if path.stat().st_size > 5 * 1024 * 1024:
        return ["file exceeds 5 MiB source limit"]
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [] if relative.parts[:2] == ("docs", "assets") else ["unexpected binary file"]
    if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
        findings.append("possible credential; value withheld")
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
            for cell in notebook.get("cells", []):
                if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments"):
                    findings.append("notebook contains outputs, execution counts or attachments")
                    break
        except (ValueError, TypeError):
            findings.append("invalid notebook JSON")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-files", action="store_true", help="Check an exported folder without a .git directory")
    args = parser.parse_args()
    if args.all_files:
        names = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and not any(part in {".git", "__pycache__"} for part in p.relative_to(ROOT).parts)]
    else:
        top = Path(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]).decode().strip()).resolve()
        if top != ROOT:
            print("This folder is not a Git repository root. Use --all-files for an exported source folder.")
            return 1
        names = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode("utf-8").split("\0")
    failures = []
    for name in names:
        if not name:
            continue
        path = ROOT / name
        if path.is_file():
            failures.extend(f"{name}: {issue}" for issue in check_file(path, Path(name)))
    if failures:
        print("\n".join(failures))
        return 1
    print("Source hygiene checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

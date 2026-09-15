# Contributing

Keep changes focused on the desktop product and its inference pipeline. Research notebooks, datasets, checkpoints, generated reports, and training runs belong outside this repository.

Before committing:

```powershell
python -m unittest discover -s tests -v
python scripts/check_repository.py
git diff --check
```

Changes to inference should also pass `--self-test`, the desktop smoke test, and an end-to-end run with trusted local weights. Never commit `.pt` files or sample laboratory images.

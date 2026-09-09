import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_repository import check_file


class RepositoryTests(unittest.TestCase):
    def test_models_are_blocked(self):
        self.assertEqual(check_file(Path("model.pt"), Path("model.pt")), ["private/generated artifact"])

    def test_notebook_outputs_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.ipynb"
            path.write_text(json.dumps({"cells": [{"outputs": [{"text": "private"}]}]}), encoding="utf-8")
            self.assertIn("notebook contains outputs, execution counts or attachments", check_file(path, Path(path.name)))

    def test_secret_value_is_not_echoed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.py"
            prefix = "gh" + "p_"
            secret = prefix + "abcdefghijklmnopqrstuvwxyz1234567890"
            path.write_text(f"TOKEN = '{secret}'", encoding="utf-8")
            findings = check_file(path, Path(path.name))
            self.assertEqual(findings, ["possible credential; value withheld"])
            self.assertNotIn(prefix, " ".join(findings))


if __name__ == "__main__":
    unittest.main()

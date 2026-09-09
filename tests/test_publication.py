import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_release import sanitize_notebook
from tools.check_repository import check_file


class PublicationTests(unittest.TestCase):
    def test_outputs_attachments_and_metadata_removed(self):
        original = {"metadata": {"widgets": {"private": "value"}, "kernelspec": {"name": "python3"}}, "cells": [
            {"cell_type": "code", "source": ["print(1)"], "metadata": {"secret": "value"}, "outputs": [{"text": "private"}], "execution_count": 7},
            {"cell_type": "markdown", "source": ["Text"], "attachments": {"image": "private"}},
        ]}
        cleaned = json.loads(sanitize_notebook(json.dumps(original)))
        self.assertEqual(cleaned["cells"][0]["source"], ["print(1)"])
        self.assertEqual(cleaned["cells"][0]["outputs"], [])
        self.assertIsNone(cleaned["cells"][0]["execution_count"])
        self.assertNotIn("attachments", cleaned["cells"][1])
        self.assertNotIn("widgets", cleaned["metadata"])

    def test_notebook_outputs_block_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ipynb"
            path.write_text(json.dumps({"cells": [{"outputs": ["private"]}]}), encoding="utf-8")
            self.assertTrue(check_file(path, Path(path.name)))

    def test_model_files_block_publication(self):
        self.assertEqual(check_file(Path("model.pt"), Path("model.pt")), ["private/generated artifact"])


if __name__ == "__main__":
    unittest.main()

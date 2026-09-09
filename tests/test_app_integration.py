"""Optional integration checks; lightweight source CI skips absent ML packages."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HAS_APP = all(importlib.util.find_spec(name) is not None for name in ("ultralytics", "pandas", "skimage"))


@unittest.skipUnless(HAS_APP, "Install requirements-app.txt for integration checks")
class AppIntegrationTests(unittest.TestCase):
    def test_spreadsheet_formulas_are_text(self):
        import pandas as pd
        from full_pipline.runtime import safe_table
        original = pd.DataFrame({"name": ["=1+1", " @SUM(A1)", "normal"], "score": [-0.1, 0.2, 0.3]})
        result = safe_table(original)
        self.assertEqual(result["name"].tolist(), ["'=1+1", "' @SUM(A1)", "normal"])
        self.assertEqual(result["score"].tolist(), original["score"].tolist())
        self.assertEqual(original.iloc[0]["name"], "=1+1")

    def test_import_does_not_attempt_network(self):
        script = '''
import socket
attempts = []
def blocked(*args, **kwargs):
    attempts.append(True)
    raise OSError("Network is blocked in this test")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.getaddrinfo = blocked
socket.gethostbyname = blocked
from full_pipline import full_pipeline
assert not attempts, "Inference import attempted network access"
assert full_pipeline.settings["sync"] is False
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_detection_produces_report(self):
        import numpy as np
        import pandas as pd
        from full_pipline import full_pipeline as pipeline
        with tempfile.TemporaryDirectory() as tmp:
            config = pipeline.AnomalyDetectionConfig(output_dir=tmp, save_xlsx=False, save_visualizations=False, save_feature_space_plot=False)
            with patch.object(pipeline, "read_image_rgb", return_value=np.zeros((736, 736, 3), dtype=np.uint8)), patch.object(pipeline, "predict_colony_masks", return_value=([], pd.DataFrame())):
                result = pipeline.run_single_image_pipeline(Path("empty.png"), None, config)
            self.assertEqual(len(result["masks"]), 0)
            self.assertEqual(len(result["selected"]), 0)
            self.assertTrue((Path(tmp) / "empty" / "preprocess_info.json").is_file())


if __name__ == "__main__":
    unittest.main()

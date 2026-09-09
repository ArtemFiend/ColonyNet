import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_ML_STACK = all(importlib.util.find_spec(name) is not None for name in ("ultralytics", "pandas", "skimage"))


@unittest.skipUnless(HAS_ML_STACK, "Install the project dependencies for integration checks")
class IntegrationTests(unittest.TestCase):
    def test_import_remains_offline(self):
        script = """
import socket
attempts = []
def blocked(*args, **kwargs):
    attempts.append(True)
    raise OSError('network disabled')
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.getaddrinfo = blocked
socket.gethostbyname = blocked
from colonynet import pipeline
assert not attempts
assert pipeline.settings['sync'] is False
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_spreadsheet_formulas_become_text(self):
        import pandas as pd
        from colonynet.runtime import safe_table

        original = pd.DataFrame({"name": ["=1+1", "normal"], "score": [-0.1, 0.2]})
        escaped = safe_table(original)
        self.assertEqual(escaped["name"].tolist(), ["'=1+1", "normal"])
        self.assertEqual(escaped["score"].tolist(), [-0.1, 0.2])

    def test_empty_detection_still_creates_report(self):
        import numpy as np
        import pandas as pd
        from colonynet import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            config = pipeline.AnomalyDetectionConfig(
                output_dir=tmp,
                save_xlsx=False,
                save_visualizations=False,
                save_feature_space_plot=False,
            )
            image = np.zeros((736, 736, 3), dtype=np.uint8)
            with patch.object(pipeline, "read_image_rgb", return_value=image), patch.object(
                pipeline, "predict_colony_masks", return_value=([], pd.DataFrame())
            ):
                result = pipeline.run_single_image_pipeline(Path("empty.png"), None, config)
            self.assertEqual(len(result["masks"]), 0)
            self.assertEqual(len(result["selected"]), 0)
            self.assertTrue((Path(tmp) / "empty" / "preprocess_info.json").is_file())


if __name__ == "__main__":
    unittest.main()

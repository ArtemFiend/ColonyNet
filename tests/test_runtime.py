import os
import tempfile
import unittest
from pathlib import Path

from colonynet.runtime import configure_offline, create_run_dir, validate_image_batch


class RuntimeTests(unittest.TestCase):
    def test_unique_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = create_run_dir(Path(tmp))
            second = create_run_dir(Path(tmp))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_duplicate_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "Plate.jpg", Path(tmp) / "plate.png"]
            for path in paths:
                path.touch()
            with self.assertRaisesRegex(ValueError, "Повторяющееся"):
                validate_image_batch(paths)

    def test_offline_environment(self):
        configure_offline()
        self.assertEqual(os.environ["YOLO_OFFLINE"], "true")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertEqual(os.environ["DO_NOT_TRACK"], "1")


if __name__ == "__main__":
    unittest.main()

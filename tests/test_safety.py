import os
import tempfile
import unittest
from pathlib import Path

from colonyseg.data.splits import source_id, split_ids, validate_split
from full_pipline.runtime import configure_offline, create_run_dir, validate_image_batch


class DatasetSplitTests(unittest.TestCase):
    def test_augmentations_stay_with_source(self):
        ids = [f"plate{i}{suffix}" for i in range(10) for suffix in ("", "__soft01", "__soft02")]
        train, val = split_ids(ids, 0.2)
        self.assertEqual(len(val), 6)
        self.assertFalse({source_id(i) for i in train} & {source_id(i) for i in val})
        self.assertEqual(set(train + val), set(ids))
        self.assertEqual((train, val), split_ids(ids, 0.2))

    def test_explicit_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(["plate__soft01__soft02"], ["plate"])

    def test_invalid_partitions_rejected(self):
        for fraction in (0, 1, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                split_ids(["a", "b"], fraction)
        with self.assertRaises(ValueError):
            split_ids(["a", "a__soft01"], 0.2)


class RuntimeTests(unittest.TestCase):
    def test_run_directories_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = create_run_dir(Path(tmp)), create_run_dir(Path(tmp))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_case_insensitive_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "Plate.jpg", Path(tmp) / "plate.png"]
            for path in paths:
                path.touch()
            with self.assertRaises(ValueError):
                validate_image_batch(paths)

    def test_offline_defaults(self):
        configure_offline()
        self.assertEqual(os.environ["YOLO_OFFLINE"], "true")
        self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertIn("ColonyNet", os.environ["YOLO_CONFIG_DIR"])


if __name__ == "__main__":
    unittest.main()

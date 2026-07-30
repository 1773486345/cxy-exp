import subprocess
import unittest
from pathlib import Path


class FrozenDirectoryTests(unittest.TestCase):
    def test_original_and_v1_directories_are_not_modified(self):
        root = Path(__file__).resolve().parents[4]
        for path in ("ts_benchmark/baselines/catch", "ts_benchmark/baselines/typefusion_catch"):
            result = subprocess.run(["git", "diff", "--", path], cwd=root, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, "", path)

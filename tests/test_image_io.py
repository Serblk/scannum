from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from plate_guard.image_io import save_jpeg_atomic


class ImageIoTests(unittest.TestCase):
    def test_saves_jpeg_to_cyrillic_path(self) -> None:
        frame = np.zeros((60, 120, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "русская папка" / "Р888АХ63.jpg"
            saved = save_jpeg_atomic(target, frame)
            self.assertTrue(saved.is_file())
            decoded = cv2.imdecode(np.fromfile(saved, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], (60, 120))


if __name__ == "__main__":
    unittest.main()

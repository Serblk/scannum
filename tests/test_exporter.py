from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from plate_guard.exporter import export_to_xlsx


class ExcelExporterTests(unittest.TestCase):
    def test_creates_readable_xlsx(self) -> None:
        rows = [
            {
                "id": 7,
                "observed_at": "2026-07-10T09:00:00+00:00",
                "camera_name": "Камера 1",
                "raw_text": "A030BC77",
                "normalized_plate": "А030ВС77",
                "ocr_confidence": 0.91,
                "detection_confidence": 0.95,
                "decision": "ALLOWED",
                "reason": "Ограничения не обнаружены",
                "image_path": "captures/camera-1/test.jpg",
                "confirmed_at": None,
                "operator_note": None,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = export_to_xlsx(rows, Path(temporary_directory) / "report")
            self.assertEqual(target.suffix, ".xlsx")
            workbook = load_workbook(target, read_only=True)
            try:
                sheet = workbook["События"]
                self.assertEqual(sheet["A2"].value, 7)
                self.assertEqual(sheet["E2"].value, "А030ВС77")
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()

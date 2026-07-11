from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plate_guard.runtime import prepare_user_data_directory


class RuntimeTests(unittest.TestCase):
    def test_prepares_production_files_without_overwriting_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resources = root / "resources"
            local = root / "local"
            (resources / "models" / "cache").mkdir(parents=True)
            (resources / "config.toml").write_text("original", encoding="utf-8")
            (resources / "models" / "cache" / "model.onnx").write_bytes(b"model")

            with (
                patch.dict("os.environ", {"LOCALAPPDATA": str(local)}),
                patch("plate_guard.runtime.sys._MEIPASS", str(resources), create=True),
            ):
                target = prepare_user_data_directory()
                (target / "config.toml").write_text("changed", encoding="utf-8")
                prepare_user_data_directory()

            self.assertEqual((target / "config.toml").read_text(encoding="utf-8"), "changed")
            self.assertTrue((target / "models" / "cache" / "model.onnx").is_file())

    def test_migrates_old_data_to_operator_friendly_folders_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resources = root / "resources"
            local = root / "local"
            target = local / "ScanNum"
            (resources / "models" / "cache").mkdir(parents=True)
            (resources / "models" / "cache" / "model.onnx").write_bytes(b"model")
            (resources / "config.toml").write_text(
                '[app]\n'
                'database_path = "data/system.db"\n'
                'captures_directory = "captures"\n'
                'reports_directory = "reports"\n',
                encoding="utf-8",
            )
            (target / "data").mkdir(parents=True)
            (target / "captures" / "camera-1").mkdir(parents=True)
            (target / "reports").mkdir(parents=True)
            (target / "data" / "system.db").write_bytes(b"database")
            (target / "captures" / "camera-1" / "plate.jpg").write_bytes(b"photo")
            (target / "reports" / "events.xlsx").write_bytes(b"excel")

            with (
                patch.dict("os.environ", {"LOCALAPPDATA": str(local)}),
                patch("plate_guard.runtime.sys._MEIPASS", str(resources), create=True),
            ):
                prepare_user_data_directory()

            self.assertTrue((target / "База" / "system.db").is_file())
            self.assertTrue((target / "Фотографии" / "camera-1" / "plate.jpg").is_file())
            self.assertTrue((target / "Excel" / "events.xlsx").is_file())
            self.assertTrue((target / "data" / "system.db").is_file())
            config = (target / "config.toml").read_text(encoding="utf-8")
            self.assertIn('database_path = "База/system.db"', config)
            self.assertIn('captures_directory = "Фотографии"', config)
            self.assertIn('reports_directory = "Excel"', config)


if __name__ == "__main__":
    unittest.main()

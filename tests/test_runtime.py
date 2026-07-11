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


if __name__ == "__main__":
    unittest.main()

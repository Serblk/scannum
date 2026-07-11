from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plate_guard.camera_settings import (
    CameraSettingsError,
    load_selected_cameras,
    save_selected_cameras,
)
from plate_guard.config import CameraConfig


class CameraSettingsTests(unittest.TestCase):
    def test_round_trip_preserves_multiple_local_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "camera_sources.json"
            cameras = [
                CameraConfig("local-0", "Камера 0", 0),
                CameraConfig("local-2", "Камера 2", 2),
            ]

            save_selected_cameras(path, cameras)

            self.assertEqual(load_selected_cameras(path), tuple(cameras))

    def test_string_source_is_reserved_for_future_rtsp_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "camera_sources.json"
            camera = CameraConfig("rtsp-1", "Въезд", "rtsp://camera/stream")

            save_selected_cameras(path, [camera])

            self.assertEqual(load_selected_cameras(path), (camera,))

    def test_rejects_duplicate_camera_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "camera_sources.json"
            path.write_text(
                '{"cameras":[{"id":"same","name":"A","source":0},'
                '{"id":"same","name":"B","source":1}]}',
                encoding="utf-8",
            )
            with self.assertRaises(CameraSettingsError):
                load_selected_cameras(path)


if __name__ == "__main__":
    unittest.main()

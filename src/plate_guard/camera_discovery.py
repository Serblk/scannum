from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .camera import _import_cv2, _open_video_capture


@dataclass(frozen=True, slots=True)
class LocalCamera:
    source: int
    name: str


def discover_local_cameras(max_sources: int = 8) -> tuple[LocalCamera, ...]:
    if max_sources < 1:
        raise ValueError("max_sources должен быть положительным")
    cv2 = _import_cv2()
    found: list[LocalCamera] = []
    for source in range(max_sources):
        capture = _open_video_capture(cv2, source)
        try:
            if capture.isOpened() and _can_read_frame(capture):
                found.append(LocalCamera(source, f"Камера {source} (встроенная/USB)"))
        finally:
            capture.release()
    return tuple(found)


def _can_read_frame(capture: Any) -> bool:
    for _ in range(10):
        success, frame = capture.read()
        if success and frame is not None:
            return True
    return False

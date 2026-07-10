from __future__ import annotations

from pathlib import Path
from typing import Any


class ImageWriteError(OSError):
    pass


def save_jpeg_atomic(output_path: str | Path, frame: Any) -> Path:
    """Сохраняет JPEG через Python Path, включая пути с кириллицей на Windows."""
    try:
        import cv2
    except ImportError as exc:
        raise ImageWriteError("OpenCV не установлен") from exc

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.jpg")
    try:
        encoded, image_buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            raise ImageWriteError("OpenCV не смог закодировать изображение")
        temporary.write_bytes(image_buffer.tobytes())
        temporary.replace(target)
    except OSError as exc:
        raise ImageWriteError(f"Не удалось сохранить изображение: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target

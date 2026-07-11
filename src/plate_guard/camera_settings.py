from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import CameraConfig


class CameraSettingsError(RuntimeError):
    pass


def load_selected_cameras(path: str | Path) -> tuple[CameraConfig, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        raw_cameras = data.get("cameras", [])
        if not isinstance(raw_cameras, list):
            raise ValueError("cameras должен быть списком")
        cameras = tuple(_parse_camera(item) for item in raw_cameras)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CameraSettingsError(f"Не удалось прочитать настройки камер: {exc}") from exc
    if len({camera.id for camera in cameras}) != len(cameras):
        raise CameraSettingsError("В настройках повторяются идентификаторы камер")
    return cameras


def save_selected_cameras(
    path: str | Path, cameras: tuple[CameraConfig, ...] | list[CameraConfig]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "cameras": [
            {
                "id": camera.id,
                "name": camera.name,
                "source": camera.source,
                "enabled": camera.enabled,
                "width": camera.width,
                "height": camera.height,
            }
            for camera in cameras
        ],
    }
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
        ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        Path(temporary_name).replace(target)
    except OSError as exc:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise CameraSettingsError(f"Не удалось сохранить настройки камер: {exc}") from exc


def _parse_camera(data: object) -> CameraConfig:
    if not isinstance(data, dict):
        raise ValueError("описание камеры должно быть объектом")
    source = data.get("source")
    if isinstance(source, bool) or not isinstance(source, (int, str)):
        raise ValueError("источник камеры должен быть числом или RTSP-строкой")
    camera_id = data.get("id")
    name = data.get("name")
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError("у камеры отсутствует id")
    if not isinstance(name, str) or not name:
        raise ValueError("у камеры отсутствует название")
    return CameraConfig(
        id=camera_id,
        name=name,
        source=source,
        enabled=bool(data.get("enabled", True)),
        width=int(data.get("width", 1280)),
        height=int(data.get("height", 720)),
    )

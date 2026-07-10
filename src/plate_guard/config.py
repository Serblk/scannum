from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CameraConfig:
    id: str
    name: str
    source: int | str
    enabled: bool = True
    width: int = 1280
    height: int = 720


@dataclass(frozen=True, slots=True)
class AppConfig:
    database_path: Path
    captures_directory: Path
    reports_directory: Path
    timezone: str
    process_every_n_frames: int
    frame_queue_size: int
    camera_retry_seconds: float
    confirmations_required: int
    confirmation_window_seconds: float
    duplicate_cooldown_seconds: float
    minimum_ocr_confidence: float
    fueling_interval_hours: int
    manual_approval_enabled: bool


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    detector_model: str
    ocr_model: str
    device: str
    model_cache_directory: Path


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    app: AppConfig
    recognition: RecognitionConfig
    cameras: tuple[CameraConfig, ...]

    @property
    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Файл настроек не найден: {config_path}")

    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Ошибка в TOML-файле настроек: {exc}") from exc

    base_directory = config_path.parent
    app_data = _required_table(data, "app")
    recognition_data = _required_table(data, "recognition")
    raw_cameras = data.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ConfigError("В config.toml должна быть хотя бы одна секция [[cameras]]")

    app = AppConfig(
        database_path=_resolve_path(base_directory, app_data, "database_path"),
        captures_directory=_resolve_path(base_directory, app_data, "captures_directory"),
        reports_directory=_resolve_path(base_directory, app_data, "reports_directory"),
        timezone=_text(app_data, "timezone"),
        process_every_n_frames=_positive_int(app_data, "process_every_n_frames"),
        frame_queue_size=_positive_int(app_data, "frame_queue_size"),
        camera_retry_seconds=_positive_number(app_data, "camera_retry_seconds"),
        confirmations_required=_positive_int(app_data, "confirmations_required"),
        confirmation_window_seconds=_positive_number(
            app_data, "confirmation_window_seconds"
        ),
        duplicate_cooldown_seconds=_positive_number(
            app_data, "duplicate_cooldown_seconds"
        ),
        minimum_ocr_confidence=_confidence(app_data, "minimum_ocr_confidence"),
        fueling_interval_hours=_positive_int(app_data, "fueling_interval_hours"),
        manual_approval_enabled=_boolean(app_data, "manual_approval_enabled"),
    )
    recognition = RecognitionConfig(
        detector_model=_text(recognition_data, "detector_model"),
        ocr_model=_text(recognition_data, "ocr_model"),
        device=_device(recognition_data),
        model_cache_directory=_resolve_path(
            base_directory, recognition_data, "model_cache_directory"
        ),
    )
    cameras = tuple(_parse_camera(item, index) for index, item in enumerate(raw_cameras, 1))
    camera_ids = [camera.id for camera in cameras]
    if len(camera_ids) != len(set(camera_ids)):
        raise ConfigError("Идентификаторы камер должны быть уникальными")
    if not any(camera.enabled for camera in cameras):
        raise ConfigError("Должна быть включена хотя бы одна камера")

    return ProjectConfig(app=app, recognition=recognition, cameras=cameras)


def _parse_camera(data: Any, index: int) -> CameraConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"Секция камеры №{index} должна быть таблицей TOML")
    source = data.get("source")
    if isinstance(source, bool) or not isinstance(source, (int, str)):
        raise ConfigError(f"У камеры №{index} source должен быть числом или строкой")
    camera_id = _text(data, "id")
    if not all(character.isalnum() or character in "-_" for character in camera_id):
        raise ConfigError(
            f"Некорректный id камеры {camera_id!r}: разрешены буквы, цифры, - и _"
        )
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError(f"У камеры {camera_id} enabled должен быть true или false")
    return CameraConfig(
        id=camera_id,
        name=_text(data, "name"),
        source=source,
        enabled=enabled,
        width=_positive_int(data, "width"),
        height=_positive_int(data, "height"),
    )


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Отсутствует секция [{key}]")
    return value


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Параметр {key} должен быть непустой строкой")
    return value.strip()


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Параметр {key} должен быть положительным целым числом")
    return value


def _positive_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"Параметр {key} должен быть положительным числом")
    return float(value)


def _confidence(data: dict[str, Any], key: str) -> float:
    value = _positive_number(data, key)
    if value > 1:
        raise ConfigError(f"Параметр {key} должен находиться в диапазоне (0, 1]")
    return value


def _boolean(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Параметр {key} должен быть true или false")
    return value


def _device(data: dict[str, Any]) -> str:
    value = _text(data, "device").lower()
    if value not in {"cpu", "cuda", "auto"}:
        raise ConfigError("recognition.device должен быть cpu, cuda или auto")
    return value


def _resolve_path(base: Path, data: dict[str, Any], key: str) -> Path:
    value = Path(_text(data, key)).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()

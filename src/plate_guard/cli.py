from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .camera import CameraDependencyError, probe_camera
from .config import ConfigError, ProjectConfig, load_config
from .exporter import ExportError, export_to_xlsx
from .plates import PlateValidationError, normalize_plate
from .recognizer import FastAlprRecognizer, RecognitionDependencyError
from .runtime import RuntimeSetupError, default_config_path
from .service import PlateGuardService, ServiceConfigurationError
from .storage import SQLiteRepository, StorageError


DEFAULT_CONFIG_PATH = default_config_path()


class GuiDependencyError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console()
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    command = arguments.command or "gui"

    try:
        config = load_config(arguments.config)
        repository = _prepare_repository(config)
        if command == "gui":
            return _gui(config, repository)
        if command == "run":
            return _run(config, repository)
        if command == "init-db":
            print(f"База готова: {repository.database_path}")
            return 0
        if command == "confirm":
            return _confirm(repository, arguments.plate, arguments.note)
        if command == "export":
            return _export(config, repository, arguments.output)
        if command == "recent":
            return _recent(repository, arguments.limit)
        if command == "check":
            return _check_recognition(config)
        if command == "check-cameras":
            return _check_cameras(config)
        parser.error(f"Неизвестная команда: {command}")
    except (
        ConfigError,
        StorageError,
        RecognitionDependencyError,
        CameraDependencyError,
        GuiDependencyError,
        ServiceConfigurationError,
        ExportError,
        PlateValidationError,
        RuntimeSetupError,
        LookupError,
    ) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Автоматическая фиксация и проверка автомобильных номеров"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="путь к config.toml (по умолчанию: config.toml)",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("gui", help="запустить графический интерфейс")
    subparsers.add_parser("run", help="запустить камеры и распознавание")
    subparsers.add_parser("init-db", help="создать или обновить структуру SQLite")

    confirm = subparsers.add_parser("confirm", help="подтвердить состоявшуюся заправку")
    confirm.add_argument("plate", help="распознанный номер")
    confirm.add_argument("--note", help="необязательное примечание")

    export = subparsers.add_parser("export", help="выгрузить историю в Excel")
    export.add_argument("--output", help="путь к итоговому .xlsx")

    recent = subparsers.add_parser("recent", help="показать последние события")
    recent.add_argument("--limit", type=int, default=20, help="число событий")
    subparsers.add_parser(
        "check",
        help="загрузить модели и проверить движок распознавания без открытия камеры",
    )
    subparsers.add_parser(
        "check-cameras",
        help="проверить получение одного кадра от каждой включённой камеры",
    )
    return parser


def _prepare_repository(config: ProjectConfig) -> SQLiteRepository:
    repository = SQLiteRepository(config.app.database_path)
    repository.initialize()
    repository.upsert_cameras(config.cameras)
    return repository


def _run(config: ProjectConfig, repository: SQLiteRepository) -> int:
    print("Загрузка локальных моделей распознавания...")
    recognizer = FastAlprRecognizer(
        detector_model=config.recognition.detector_model,
        ocr_model=config.recognition.ocr_model,
        device=config.recognition.device,
        model_cache_directory=config.recognition.model_cache_directory,
    )
    service = PlateGuardService(config, repository, recognizer)
    service.run()
    return 0


def _gui(config: ProjectConfig, repository: SQLiteRepository) -> int:
    try:
        from .gui import run_gui
    except ImportError as exc:
        raise GuiDependencyError(
            "PySide6 не установлен. Выполните: python -m pip install -r requirements.txt"
        ) from exc
    print("Загрузка локальных моделей распознавания...")
    recognizer = FastAlprRecognizer(
        detector_model=config.recognition.detector_model,
        ocr_model=config.recognition.ocr_model,
        device=config.recognition.device,
        model_cache_directory=config.recognition.model_cache_directory,
    )
    service = PlateGuardService(config, repository, recognizer)
    return run_gui(config, repository, service)


def _confirm(repository: SQLiteRepository, raw_plate: str, note: str | None) -> int:
    plate = normalize_plate(raw_plate)
    fueling_id = repository.confirm_latest_allowed_fueling(
        plate=plate,
        confirmed_at=datetime.now(UTC),
        note=note,
    )
    print(f"Заправка {plate} подтверждена, запись #{fueling_id}")
    return 0


def _export(
    config: ProjectConfig,
    repository: SQLiteRepository,
    raw_output: str | None,
) -> int:
    default_name = f"events_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    output = Path(raw_output) if raw_output else config.app.reports_directory / default_name
    target = export_to_xlsx(repository.export_rows(), output)
    print(f"Excel-отчёт создан: {target}")
    return 0


def _recent(repository: SQLiteRepository, limit: int) -> int:
    rows = repository.recent_recognitions(limit)
    if not rows:
        print("Событий пока нет")
        return 0
    for row in rows:
        plate = row["normalized_plate"] or "нераспознанный номер"
        print(
            f"#{row['id']} {row['observed_at']} {row['camera_id']} "
            f"{plate} {row['decision']} — {row['reason']}"
        )
    return 0


def _check_recognition(config: ProjectConfig) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RecognitionDependencyError(
            "NumPy не установлен. Выполните: python -m pip install -r requirements.txt"
        ) from exc

    print("Проверка моделей распознавания...")
    recognizer = FastAlprRecognizer(
        detector_model=config.recognition.detector_model,
        ocr_model=config.recognition.ocr_model,
        device=config.recognition.device,
        model_cache_directory=config.recognition.model_cache_directory,
    )
    recognizer.recognize(np.zeros((384, 640, 3), dtype=np.uint8))
    print("Модели загружены, движок распознавания готов")
    return 0


def _check_cameras(config: ProjectConfig) -> int:
    failures = 0
    for camera in config.enabled_cameras:
        frame_size = probe_camera(camera)
        if frame_size is None:
            failures += 1
            print(
                f"{camera.id} ({camera.name}): камера недоступна; "
                f"проверьте подключение и source = {camera.source!r}"
            )
        else:
            print(f"{camera.id} ({camera.name}): кадр {frame_size[0]}x{frame_size[1]} получен")
    return 1 if failures else 0


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

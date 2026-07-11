from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "ScanNum"


class RuntimeSetupError(RuntimeError):
    pass


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def default_config_path() -> Path:
    if not is_frozen():
        return Path(__file__).resolve().parents[2] / "config.toml"
    return prepare_user_data_directory() / "config.toml"


def prepare_user_data_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeSetupError("Не найдена системная папка LOCALAPPDATA")

    target = Path(local_app_data) / APP_DIRECTORY_NAME
    target.mkdir(parents=True, exist_ok=True)
    resources = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    _copy_once(resources / "config.toml", target / "config.toml")
    _copy_tree_once(resources / "models" / "cache", target / "models" / "cache")
    _migrate_operator_data_layout(target)
    return target


def _migrate_operator_data_layout(target: Path) -> None:
    marker = target / ".operator-layout-v1"
    if marker.exists():
        return

    mappings = (
        (target / "data", target / "База"),
        (target / "captures", target / "Фотографии"),
        (target / "reports", target / "Excel"),
    )
    try:
        for source, destination in mappings:
            if source.is_dir() and not destination.exists():
                temporary = destination.with_name(f".{destination.name}.migration")
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary)
                temporary.replace(destination)
            else:
                destination.mkdir(parents=True, exist_ok=True)

        config_path = target / "config.toml"
        text = config_path.read_text(encoding="utf-8")
        replacements = {
            'database_path = "data/system.db"': 'database_path = "База/system.db"',
            'captures_directory = "captures"': 'captures_directory = "Фотографии"',
            'reports_directory = "reports"': 'reports_directory = "Excel"',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        temporary_config = config_path.with_suffix(".toml.migration")
        temporary_config.write_text(text, encoding="utf-8")
        temporary_config.replace(config_path)
        marker.write_text("1", encoding="ascii")
    except OSError as exc:
        raise RuntimeSetupError(f"Не удалось подготовить папки данных: {exc}") from exc


def _copy_once(source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.is_file():
        raise RuntimeSetupError(f"В сборке отсутствует файл: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree_once(source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.is_dir():
        raise RuntimeSetupError("В сборке отсутствуют модели распознавания")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

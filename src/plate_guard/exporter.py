from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


class ExportError(RuntimeError):
    pass


_COLUMNS = (
    ("id", "ID"),
    ("observed_at", "Дата и время UTC"),
    ("camera_name", "Камера"),
    ("raw_text", "Исходный OCR"),
    ("normalized_plate", "Номер"),
    ("ocr_confidence", "Уверенность OCR"),
    ("detection_confidence", "Уверенность детектора"),
    ("decision", "Решение"),
    ("reason", "Причина"),
    ("confirmed_at", "Заправка подтверждена UTC"),
    ("fueling_outcome", "Результат заправки"),
    ("decision_mode", "Режим решения"),
    ("decided_at", "Решение принято UTC"),
    ("operator_note", "Примечание"),
    ("image_path", "Фотография"),
)


def export_to_xlsx(rows: Sequence[dict[str, Any]], output_path: str | Path) -> Path:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise ExportError(
            "openpyxl не установлен. Выполните: python -m pip install -r requirements.txt"
        ) from exc

    target = Path(output_path).resolve()
    if target.suffix.lower() != ".xlsx":
        target = target.with_suffix(".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.xlsx")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "События"
    sheet.append([title for _, title in _COLUMNS])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([row.get(key) for key, _ in _COLUMNS])

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column_cells in sheet.columns:
        values = ("" if cell.value is None else str(cell.value) for cell in column_cells)
        max_length = max((len(value) for value in values), default=10)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 60)

    try:
        workbook.save(temporary)
        temporary.replace(target)
    except PermissionError as exc:
        raise ExportError(
            f"Не удалось обновить {target.name}. Закройте этот файл в Excel и повторите."
        ) from exc
    except OSError as exc:
        raise ExportError(f"Не удалось создать Excel-отчёт: {exc}") from exc
    finally:
        workbook.close()
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target

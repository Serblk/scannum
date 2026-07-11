from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import CameraConfig
from .models import DecisionMode, FuelingOutcome, RecognitionEvent


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recognitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL REFERENCES cameras(id),
    observed_at TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_plate TEXT,
    ocr_confidence REAL NOT NULL CHECK (ocr_confidence BETWEEN 0 AND 1),
    detection_confidence REAL NOT NULL CHECK (detection_confidence BETWEEN 0 AND 1),
    decision TEXT NOT NULL CHECK (decision IN ('ALLOWED', 'DENIED', 'REVIEW')),
    reason TEXT NOT NULL,
    image_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fuelings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recognition_id INTEGER NOT NULL UNIQUE REFERENCES recognitions(id),
    plate TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    operator_note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fueling_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recognition_id INTEGER NOT NULL UNIQUE REFERENCES recognitions(id),
    outcome TEXT NOT NULL CHECK (outcome IN ('FUELED', 'NOT_FUELED')),
    mode TEXT NOT NULL CHECK (mode IN ('MANUAL', 'AUTO')),
    decided_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_credentials (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT,
    occurred_at TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recognitions_plate_time
    ON recognitions(normalized_plate, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_fuelings_plate_time
    ON fuelings(plate, confirmed_at DESC);
CREATE INDEX IF NOT EXISTS idx_fueling_decisions_recognition
    ON fueling_decisions(recognition_id);
"""


class StorageError(RuntimeError):
    pass


class SQLiteRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connection() as connection:
                previous_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                connection.executescript(_SCHEMA)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fueling_decisions (
                        recognition_id, outcome, mode, decided_at, created_at
                    )
                    SELECT recognition_id, 'FUELED', 'MANUAL', confirmed_at, created_at
                    FROM fuelings
                    """
                )
                if previous_version in {1, 2}:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO fueling_decisions (
                            recognition_id, outcome, mode, decided_at, created_at
                        )
                        SELECT id, 'NOT_FUELED', 'MANUAL', created_at, created_at
                        FROM recognitions
                        WHERE decision = 'ALLOWED'
                        """
                    )
                connection.execute("PRAGMA user_version = 4")
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось создать базу {self.database_path}: {exc}") from exc

    def upsert_cameras(self, cameras: Sequence[CameraConfig]) -> None:
        now = _to_db_datetime(datetime.now(UTC))
        values = [
            (camera.id, camera.name, str(camera.source), int(camera.enabled), now, now)
            for camera in cameras
        ]
        try:
            with self._connection() as connection:
                connection.executemany(
                    """
                    INSERT INTO cameras (id, name, source, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        source = excluded.source,
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    values,
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось сохранить настройки камер: {exc}") from exc

    def record_recognition(self, event: RecognitionEvent, auto_confirm: bool = False) -> int:
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO recognitions (
                        camera_id, observed_at, raw_text, normalized_plate,
                        ocr_confidence, detection_confidence, decision,
                        reason, image_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.camera_id,
                        _to_db_datetime(event.observed_at),
                        event.raw_text,
                        event.normalized_plate,
                        event.ocr_confidence,
                        event.detection_confidence,
                        event.decision.value,
                        event.reason,
                        event.image_path,
                        now,
                    ),
                )
                recognition_id = int(cursor.lastrowid)
                if auto_confirm:
                    self._resolve_in_connection(
                        connection=connection,
                        recognition_id=recognition_id,
                        outcome=FuelingOutcome.FUELED,
                        mode=DecisionMode.AUTO,
                        decided_value=_to_db_datetime(event.observed_at),
                        note=None,
                        created_value=now,
                    )
                return recognition_id
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось записать распознавание: {exc}") from exc

    def last_confirmed_fueling(self, plate: str) -> datetime | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT confirmed_at
                    FROM fuelings
                    WHERE plate = ?
                    ORDER BY confirmed_at DESC
                    LIMIT 1
                    """,
                    (plate,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось проверить историю заправок: {exc}") from exc
        return _from_db_datetime(row["confirmed_at"]) if row else None

    def confirm_latest_allowed_fueling(
        self,
        plate: str,
        confirmed_at: datetime,
        note: str | None = None,
    ) -> int:
        confirmed_value = _to_db_datetime(confirmed_at)
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT r.id
                    FROM recognitions AS r
                    LEFT JOIN fueling_decisions AS d ON d.recognition_id = r.id
                    WHERE r.normalized_plate = ?
                      AND r.decision = 'ALLOWED'
                      AND r.observed_at <= ?
                      AND d.id IS NULL
                    ORDER BY r.observed_at DESC
                    LIMIT 1
                    """,
                    (plate, confirmed_value),
                ).fetchone()
                if row is None:
                    raise LookupError(
                        "Нет неподтверждённого разрешённого события для этого номера"
                    )
                return self._resolve_in_connection(
                    connection=connection,
                    recognition_id=int(row["id"]),
                    outcome=FuelingOutcome.FUELED,
                    mode=DecisionMode.MANUAL,
                    decided_value=confirmed_value,
                    note=note,
                    created_value=now,
                )
        except LookupError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось подтвердить заправку: {exc}") from exc

    def resolve_fueling_decision(
        self,
        recognition_id: int,
        outcome: FuelingOutcome,
        mode: DecisionMode,
        decided_at: datetime,
        note: str | None = None,
    ) -> int:
        decided_value = _to_db_datetime(decided_at)
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                return self._resolve_in_connection(
                    connection=connection,
                    recognition_id=recognition_id,
                    outcome=outcome,
                    mode=mode,
                    decided_value=decided_value,
                    note=note,
                    created_value=now,
                )
        except LookupError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось сохранить решение оператора: {exc}") from exc

    def pending_allowed_recognition(self, plate: str) -> int | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT r.id
                    FROM recognitions AS r
                    LEFT JOIN fueling_decisions AS d ON d.recognition_id = r.id
                    WHERE r.normalized_plate = ?
                      AND r.decision = 'ALLOWED'
                      AND d.id IS NULL
                    ORDER BY r.observed_at DESC
                    LIMIT 1
                    """,
                    (plate,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось проверить ожидающие решения: {exc}") from exc
        return int(row["id"]) if row else None

    def latest_pending_recognition(self) -> dict[str, Any] | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT
                        r.id, r.observed_at, r.camera_id, r.normalized_plate,
                        r.ocr_confidence, r.reason
                    FROM recognitions AS r
                    LEFT JOIN fueling_decisions AS d ON d.recognition_id = r.id
                    WHERE r.decision = 'ALLOWED'
                      AND r.normalized_plate IS NOT NULL
                      AND d.id IS NULL
                    ORDER BY r.observed_at DESC
                    LIMIT 1
                    """
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать ожидающее решение: {exc}") from exc
        return dict(row) if row else None

    def set_manual_approval_enabled(self, enabled: bool) -> None:
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES ('manual_approval_enabled', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    ("1" if enabled else "0", now),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось сохранить режим работы: {exc}") from exc

    def manual_approval_enabled(self, default: bool) -> bool:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT value FROM settings WHERE key = 'manual_approval_enabled'"
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать режим работы: {exc}") from exc
        if row is None:
            return default
        return row["value"] == "1"

    def set_display_timeout_seconds(self, seconds: int) -> None:
        if isinstance(seconds, bool) or not 3 <= seconds <= 120:
            raise ValueError("Время отображения должно быть от 3 до 120 секунд")
        self._set_setting("display_timeout_seconds", str(seconds))

    def display_timeout_seconds(self, default: int = 10) -> int:
        if not 3 <= default <= 120:
            raise ValueError("Значение по умолчанию должно быть от 3 до 120 секунд")
        value = self._get_setting("display_timeout_seconds")
        if value is None:
            return default
        try:
            seconds = int(value)
        except ValueError:
            return default
        return seconds if 3 <= seconds <= 120 else default

    def _set_setting(self, key: str, value: str) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, _to_db_datetime(datetime.now(UTC))),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось сохранить настройку {key}: {exc}") from exc

    def _get_setting(self, key: str) -> str | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT value FROM settings WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать настройку {key}: {exc}") from exc
        return str(row["value"]) if row else None

    def admin_password_hash(self) -> str | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT password_hash FROM admin_credentials WHERE id = 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать пароль администратора: {exc}") from exc
        return str(row["password_hash"]) if row else None

    def create_admin_password(self, password_hash: str) -> None:
        if not password_hash:
            raise ValueError("Хеш пароля не может быть пустым")
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                existing = connection.execute(
                    "SELECT 1 FROM admin_credentials WHERE id = 1"
                ).fetchone()
                if existing is not None:
                    raise LookupError("Пароль администратора уже создан")
                connection.execute(
                    """
                    INSERT INTO admin_credentials (
                        id, password_hash, created_at, updated_at
                    ) VALUES (1, ?, ?, ?)
                    """,
                    (password_hash, now, now),
                )
        except LookupError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось создать пароль администратора: {exc}") from exc

    def update_admin_password_hash(self, password_hash: str) -> None:
        if not password_hash:
            raise ValueError("Хеш пароля не может быть пустым")
        now = _to_db_datetime(datetime.now(UTC))
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE admin_credentials
                    SET password_hash = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (password_hash, now),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Пароль администратора ещё не создан")
        except LookupError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось обновить пароль администратора: {exc}") from exc

    def history_summary(self) -> dict[str, int]:
        tables = ("recognitions", "fuelings", "fueling_decisions", "errors")
        try:
            with self._connection() as connection:
                return {
                    table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in tables
                }
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось подсчитать историю: {exc}") from exc

    def clear_history(self) -> dict[str, int]:
        try:
            with self._connection() as connection:
                summary = {
                    table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in ("recognitions", "fuelings", "fueling_decisions", "errors")
                }
                connection.execute("DELETE FROM fueling_decisions")
                connection.execute("DELETE FROM fuelings")
                connection.execute("DELETE FROM recognitions")
                connection.execute("DELETE FROM errors")
                connection.execute(
                    """
                    DELETE FROM sqlite_sequence
                    WHERE name IN ('recognitions', 'fuelings', 'fueling_decisions', 'errors')
                    """
                )
                return summary
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось очистить историю: {exc}") from exc

    def log_error(
        self,
        category: str,
        message: str,
        camera_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        timestamp = occurred_at or datetime.now(UTC)
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO errors (camera_id, occurred_at, category, message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (camera_id, _to_db_datetime(timestamp), category, message[:2000]),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось записать ошибку в журнал: {exc}") from exc

    def export_rows(self) -> list[dict[str, Any]]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        r.id,
                        r.observed_at,
                        c.name AS camera_name,
                        r.raw_text,
                        r.normalized_plate,
                        r.ocr_confidence,
                        r.detection_confidence,
                        r.decision,
                        r.reason,
                        r.image_path,
                        f.confirmed_at,
                        f.operator_note,
                        d.outcome AS fueling_outcome,
                        d.mode AS decision_mode,
                        d.decided_at
                    FROM recognitions AS r
                    JOIN cameras AS c ON c.id = r.camera_id
                    LEFT JOIN fuelings AS f ON f.recognition_id = r.id
                    LEFT JOIN fueling_decisions AS d ON d.recognition_id = r.id
                    ORDER BY r.observed_at DESC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать данные для отчёта: {exc}") from exc
        return [dict(row) for row in rows]

    def recent_recognitions(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("Лимит должен быть положительным")
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        r.id, r.observed_at, r.camera_id, r.normalized_plate,
                        r.decision, r.reason, d.outcome, d.mode, d.decided_at
                    FROM recognitions AS r
                    LEFT JOIN fueling_decisions AS d ON d.recognition_id = r.id
                    ORDER BY r.observed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Не удалось прочитать последние события: {exc}") from exc
        return [dict(row) for row in rows]

    def _resolve_in_connection(
        self,
        connection: sqlite3.Connection,
        recognition_id: int,
        outcome: FuelingOutcome,
        mode: DecisionMode,
        decided_value: str,
        note: str | None,
        created_value: str,
    ) -> int:
        recognition = connection.execute(
            """
            SELECT id, normalized_plate, decision
            FROM recognitions
            WHERE id = ?
            """,
            (recognition_id,),
        ).fetchone()
        if recognition is None:
            raise LookupError(f"Событие #{recognition_id} не найдено")
        if recognition["decision"] != "ALLOWED" or recognition["normalized_plate"] is None:
            raise LookupError("Решение можно принять только для разрешённого номера")
        existing = connection.execute(
            "SELECT id FROM fueling_decisions WHERE recognition_id = ?",
            (recognition_id,),
        ).fetchone()
        if existing is not None:
            raise LookupError("Для этого события решение уже принято")

        decision_cursor = connection.execute(
            """
            INSERT INTO fueling_decisions (
                recognition_id, outcome, mode, decided_at, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (recognition_id, outcome.value, mode.value, decided_value, created_value),
        )
        if outcome is FuelingOutcome.FUELED:
            connection.execute(
                """
                INSERT INTO fuelings (
                    recognition_id, plate, confirmed_at, operator_note, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    recognition_id,
                    recognition["normalized_plate"],
                    decided_value,
                    note,
                    created_value,
                ),
            )
        return int(decision_cursor.lastrowid)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _to_db_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время для базы должно содержать часовой пояс")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _from_db_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise StorageError("В базе найдено время без часового пояса")
    return parsed.astimezone(UTC)

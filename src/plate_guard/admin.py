from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from .storage import SQLiteRepository, StorageError


MINIMUM_PASSWORD_LENGTH = 10
MAXIMUM_ATTEMPTS = 5
LOCK_SECONDS = 30


class AdminError(RuntimeError):
    pass


class PasswordPolicyError(AdminError):
    pass


class AuthenticationError(AdminError):
    pass


class AdminLockedError(AuthenticationError):
    def __init__(self, remaining_seconds: int) -> None:
        self.remaining_seconds = remaining_seconds
        super().__init__(f"Вход временно заблокирован. Повторите через {remaining_seconds} сек.")


class HistoryClearError(AdminError):
    pass


@dataclass(frozen=True, slots=True)
class HistoryPreview:
    recognitions: int
    fuelings: int
    decisions: int
    errors: int
    photos: int


@dataclass(frozen=True, slots=True)
class HistoryClearResult:
    preview: HistoryPreview
    cleanup_warning: str | None = None


class AdminService:
    def __init__(
        self,
        repository: SQLiteRepository,
        captures_directory: str | Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repository = repository
        self._captures_directory = Path(captures_directory).resolve()
        self._clock = clock
        self._password_hasher = PasswordHasher()
        self._failed_attempts = 0
        self._locked_until = 0.0

    @property
    def password_is_configured(self) -> bool:
        return self._repository.admin_password_hash() is not None

    def create_password(self, password: str, confirmation: str) -> None:
        if self.password_is_configured:
            raise PasswordPolicyError("Пароль администратора уже создан")
        _validate_password(password, confirmation)
        self._repository.create_admin_password(self._password_hasher.hash(password))

    def verify_password(self, password: str) -> None:
        self._check_lock()
        password_hash = self._repository.admin_password_hash()
        if password_hash is None:
            raise AuthenticationError("Пароль администратора ещё не создан")
        try:
            self._password_hasher.verify(password_hash, password)
        except VerifyMismatchError as exc:
            self._register_failure()
            raise AuthenticationError("Неверный пароль администратора") from exc
        except (VerificationError, InvalidHashError) as exc:
            raise AuthenticationError("Хеш пароля повреждён или имеет неверный формат") from exc

        self._failed_attempts = 0
        self._locked_until = 0.0
        if self._password_hasher.check_needs_rehash(password_hash):
            self._repository.update_admin_password_hash(
                self._password_hasher.hash(password)
            )

    def preview_history_clear(self) -> HistoryPreview:
        summary = self._repository.history_summary()
        return HistoryPreview(
            recognitions=summary["recognitions"],
            fuelings=summary["fuelings"],
            decisions=summary["fueling_decisions"],
            errors=summary["errors"],
            photos=_count_photos(self._captures_directory),
        )

    def clear_history_and_photos(self, password: str) -> HistoryClearResult:
        self.verify_password(password)
        preview = self.preview_history_clear()
        trash_directory = self._captures_directory.parent / (
            f".{self._captures_directory.name}-clear-{uuid4().hex}"
        )
        moved_names: list[str] = []
        try:
            moved_names = _move_capture_contents_to_trash(
                self._captures_directory, trash_directory
            )
            self._repository.clear_history()
        except Exception as exc:
            try:
                _restore_capture_contents(
                    self._captures_directory, trash_directory, moved_names
                )
            except OSError as restore_exc:
                raise HistoryClearError(
                    "Очистка не выполнена, а автоматическое восстановление фотографий "
                    f"завершилось ошибкой: {restore_exc}"
                ) from exc
            if isinstance(exc, (StorageError, OSError)):
                raise HistoryClearError(f"Очистка истории отменена: {exc}") from exc
            raise

        cleanup_warning = None
        try:
            if trash_directory.exists():
                shutil.rmtree(trash_directory)
        except OSError as exc:
            cleanup_warning = (
                "История очищена, но временная папка фотографий не удалена: "
                f"{trash_directory} ({exc})"
            )
        return HistoryClearResult(preview=preview, cleanup_warning=cleanup_warning)

    def _check_lock(self) -> None:
        remaining = self._locked_until - self._clock()
        if remaining > 0:
            raise AdminLockedError(max(1, int(remaining + 0.999)))
        if self._locked_until:
            self._failed_attempts = 0
            self._locked_until = 0.0

    def _register_failure(self) -> None:
        self._failed_attempts += 1
        if self._failed_attempts >= MAXIMUM_ATTEMPTS:
            self._locked_until = self._clock() + LOCK_SECONDS


def _validate_password(password: str, confirmation: str) -> None:
    if password != confirmation:
        raise PasswordPolicyError("Введённые пароли не совпадают")
    if len(password) < MINIMUM_PASSWORD_LENGTH:
        raise PasswordPolicyError(
            f"Пароль должен содержать не менее {MINIMUM_PASSWORD_LENGTH} символов"
        )
    if not password.strip():
        raise PasswordPolicyError("Пароль не может состоять только из пробелов")


def _count_photos(captures_directory: Path) -> int:
    if not captures_directory.is_dir():
        return 0
    return sum(
        1
        for path in captures_directory.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )


def _move_capture_contents_to_trash(
    captures_directory: Path,
    trash_directory: Path,
) -> list[str]:
    if not captures_directory.exists():
        return []
    trash_directory.mkdir(parents=True, exist_ok=False)
    moved_names: list[str] = []
    try:
        for item in captures_directory.iterdir():
            if item.name == ".gitkeep":
                continue
            item.replace(trash_directory / item.name)
            moved_names.append(item.name)
    except OSError:
        _restore_capture_contents(captures_directory, trash_directory, moved_names)
        raise
    return moved_names


def _restore_capture_contents(
    captures_directory: Path,
    trash_directory: Path,
    moved_names: list[str],
) -> None:
    captures_directory.mkdir(parents=True, exist_ok=True)
    for name in reversed(moved_names):
        source = trash_directory / name
        if source.exists():
            source.replace(captures_directory / name)
    if trash_directory.exists():
        trash_directory.rmdir()

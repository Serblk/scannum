from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .camera import CameraManager, FramePacket
from .config import ProjectConfig
from .consensus import TemporalConsensus
from .image_io import ImageWriteError, save_jpeg_atomic
from .models import (
    AccessDecision,
    DecisionMode,
    DecisionStatus,
    FuelingOutcome,
    PlateCandidate,
    RecognitionEvent,
)
from .recognizer import FastAlprRecognizer
from .rules import evaluate_access
from .storage import SQLiteRepository, StorageError


LOGGER = logging.getLogger(__name__)
FrameHandler = Callable[[str, Any, Sequence[PlateCandidate]], None]
EventHandler = Callable[[int, RecognitionEvent, AccessDecision, bool], None]
ErrorHandler = Callable[[str], None]
_STATUS_LABELS = {
    DecisionStatus.ALLOWED: "РАЗРЕШЕНО",
    DecisionStatus.DENIED: "ЗАПРЕЩЕНО",
    DecisionStatus.REVIEW: "ТРЕБУЕТСЯ ПРОВЕРКА",
}


class ServiceConfigurationError(RuntimeError):
    pass


class PlateGuardService:
    def __init__(
        self,
        config: ProjectConfig,
        repository: SQLiteRepository,
        recognizer: FastAlprRecognizer,
        frame_handler: FrameHandler | None = None,
        event_handler: EventHandler | None = None,
        error_handler: ErrorHandler | None = None,
    ) -> None:
        self._config = config
        self._repository = repository
        self._recognizer = recognizer
        self._frame_handler = frame_handler
        self._event_handler = event_handler
        self._error_handler = error_handler
        self._stop_event = threading.Event()
        self._mode_lock = threading.Lock()
        self._processing_lock = threading.RLock()
        self._camera_lock = threading.RLock()
        self._running = False
        self._manual_approval_enabled = repository.manual_approval_enabled(
            config.app.manual_approval_enabled
        )
        try:
            self._timezone = ZoneInfo(config.app.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ServiceConfigurationError(
                f"Неизвестный часовой пояс: {config.app.timezone}"
            ) from exc
        self._frames: queue.Queue[FramePacket] = queue.Queue(
            maxsize=config.app.frame_queue_size
        )
        self._consensus = self._create_consensus()
        self._active_cameras = config.enabled_cameras
        self._cameras = CameraManager(
            cameras=self._active_cameras,
            output_queue=self._frames,
            process_every_n_frames=config.app.process_every_n_frames,
            retry_seconds=config.app.camera_retry_seconds,
            error_handler=self._handle_camera_error,
        )

    @property
    def manual_approval_enabled(self) -> bool:
        with self._mode_lock:
            return self._manual_approval_enabled

    def set_handlers(
        self,
        frame_handler: FrameHandler | None,
        event_handler: EventHandler | None,
        error_handler: ErrorHandler | None,
    ) -> None:
        self._frame_handler = frame_handler
        self._event_handler = event_handler
        self._error_handler = error_handler

    def set_manual_approval_enabled(self, enabled: bool) -> None:
        self._repository.set_manual_approval_enabled(enabled)
        with self._mode_lock:
            self._manual_approval_enabled = enabled

    @property
    def display_timeout_seconds(self) -> int:
        return self._repository.display_timeout_seconds(default=10)

    def set_display_timeout_seconds(self, seconds: int) -> None:
        self._repository.set_display_timeout_seconds(seconds)

    @property
    def history_visible_columns(self) -> tuple[str, ...]:
        return self._repository.history_visible_columns()

    def set_history_visible_columns(self, columns: Sequence[str]) -> None:
        self._repository.set_history_visible_columns(columns)

    def resolve_manual_decision(
        self,
        recognition_id: int,
        fueled: bool,
        decided_at: datetime | None = None,
    ) -> int:
        outcome = FuelingOutcome.FUELED if fueled else FuelingOutcome.NOT_FUELED
        return self._repository.resolve_fueling_decision(
            recognition_id=recognition_id,
            outcome=outcome,
            mode=DecisionMode.MANUAL,
            decided_at=decided_at or datetime.now(UTC),
        )

    @property
    def active_cameras(self) -> tuple[Any, ...]:
        with self._camera_lock:
            return tuple(self._active_cameras)

    def configure_cameras(self, cameras: Sequence[Any]) -> None:
        selected = tuple(camera for camera in cameras if camera.enabled)
        with self._camera_lock:
            if self._running:
                self._cameras.stop()
            self._active_cameras = selected
            self._cameras = self._create_camera_manager(selected)
            self._consensus = self._create_consensus()
            self._clear_frame_queue()
            if self._running:
                self._cameras.start()

    def _create_camera_manager(self, cameras: Sequence[Any]) -> CameraManager:
        return CameraManager(
            cameras=cameras,
            output_queue=self._frames,
            process_every_n_frames=self._config.app.process_every_n_frames,
            retry_seconds=self._config.app.camera_retry_seconds,
            error_handler=self._handle_camera_error,
        )

    def _clear_frame_queue(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def run(self) -> None:
        self._stop_event.clear()
        self._config.app.captures_directory.mkdir(parents=True, exist_ok=True)
        with self._camera_lock:
            self._running = True
            self._cameras.start()
            camera_count = len(self._active_cameras)
        LOGGER.info("Запущено камер: %d", camera_count)
        print("Система запущена. Для остановки нажмите Ctrl+C.")
        try:
            while not self._stop_event.is_set():
                try:
                    packet = self._frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._process_frame(packet)
        except KeyboardInterrupt:
            print("\nОстановка системы...")
        finally:
            with self._camera_lock:
                self._cameras.stop()
                self._running = False

    def stop(self) -> None:
        self._stop_event.set()

    def perform_maintenance(self, operation: Callable[[], Any]) -> Any:
        """Выполняет обслуживание без одновременной записи новых событий."""
        with self._processing_lock:
            result = operation()
            self._consensus = self._create_consensus()
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            return result

    def _create_consensus(self) -> TemporalConsensus:
        return TemporalConsensus(
            confirmations_required=max(4, self._config.app.confirmations_required),
            window=timedelta(seconds=self._config.app.confirmation_window_seconds),
            duplicate_cooldown=timedelta(
                seconds=self._config.app.duplicate_cooldown_seconds
            ),
            minimum_ocr_confidence=max(0.70, self._config.app.minimum_ocr_confidence),
        )

    def _process_frame(self, packet: FramePacket) -> None:
        with self._processing_lock:
            self._process_frame_unlocked(packet)

    def _process_frame_unlocked(self, packet: FramePacket) -> None:
        try:
            candidates = self._recognizer.recognize(packet.frame)
        except Exception as exc:
            LOGGER.exception("Ошибка распознавания кадра камеры %s", packet.camera_id)
            self._safe_log_error("recognition", str(exc), packet.camera_id)
            self._emit_error(f"Ошибка распознавания: {exc}")
            return

        if self._frame_handler is not None:
            self._frame_handler(packet.camera_id, packet.frame, candidates)

        confirmed_results = self._consensus.observe_many(
            camera_id=packet.camera_id,
            candidates=candidates,
            observed_at=packet.observed_at,
            frame=packet.frame,
        )
        for confirmed in confirmed_results:
            decision = (
                AccessDecision(DecisionStatus.REVIEW, confirmed.review_reason)
                if confirmed.review_reason is not None
                else self._decide(
                    confirmed.candidate.normalized_plate,
                    confirmed.observed_at,
                )
            )
            manual_mode = self.manual_approval_enabled
            if (
                manual_mode
                and decision.status is DecisionStatus.ALLOWED
                and confirmed.candidate.normalized_plate is not None
            ):
                try:
                    pending = self._repository.pending_allowed_recognition(
                        confirmed.candidate.normalized_plate
                    )
                except StorageError as exc:
                    LOGGER.error("Не удалось проверить ожидающие решения: %s", exc)
                    self._emit_error(f"Ошибка базы: {exc}")
                    continue
                if pending is not None:
                    LOGGER.info(
                        "Номер %s уже ожидает ручного решения в событии #%d",
                        confirmed.candidate.normalized_plate,
                        pending,
                    )
                    continue

            image_path = self._save_capture(
                camera_id=confirmed.camera_id,
                plate=confirmed.candidate.stable_key,
                observed_at=confirmed.observed_at,
                frame=confirmed.frame,
            )
            event = RecognitionEvent(
                camera_id=confirmed.camera_id,
                observed_at=confirmed.observed_at,
                raw_text=confirmed.candidate.raw_text,
                normalized_plate=(
                    None
                    if confirmed.review_reason is not None
                    else confirmed.candidate.normalized_plate
                ),
                ocr_confidence=confirmed.average_ocr_confidence,
                detection_confidence=confirmed.candidate.detection_confidence,
                decision=decision.status,
                reason=decision.reason,
                image_path=str(image_path) if image_path else None,
            )
            try:
                event_id = self._repository.record_recognition(
                    event,
                    auto_confirm=(
                        not manual_mode and decision.status is DecisionStatus.ALLOWED
                    ),
                )
                requires_approval = manual_mode and decision.status is DecisionStatus.ALLOWED
            except (StorageError, LookupError) as exc:
                LOGGER.error("Событие или результат не записаны: %s", exc)
                self._emit_error(f"Ошибка базы: {exc}")
                continue

            self._print_decision(event_id, event, decision, requires_approval)
            if self._event_handler is not None:
                self._event_handler(event_id, event, decision, requires_approval)

    def _decide(self, plate: str | None, observed_at: datetime) -> AccessDecision:
        if plate is None:
            return AccessDecision(
                DecisionStatus.REVIEW,
                "OCR стабильно видит текст, но он не соответствует формату российского номера",
            )
        try:
            last_fueling = self._repository.last_confirmed_fueling(plate)
        except StorageError as exc:
            LOGGER.error("Не удалось проверить историю номера %s: %s", plate, exc)
            return AccessDecision(
                DecisionStatus.REVIEW,
                "История заправок временно недоступна; автоматическое разрешение запрещено",
            )
        return evaluate_access(
            plate=plate,
            observed_at=observed_at.astimezone(self._timezone),
            last_confirmed_fueling_at=last_fueling,
            fueling_interval=timedelta(hours=self._config.app.fueling_interval_hours),
        )

    def _save_capture(
        self,
        camera_id: str,
        plate: str,
        observed_at: datetime,
        frame: Any,
    ) -> Path | None:
        camera_directory = self._config.app.captures_directory / camera_id
        camera_directory.mkdir(parents=True, exist_ok=True)
        safe_plate = "".join(character for character in plate if character.isalnum()) or "unknown"
        timestamp = observed_at.astimezone(self._timezone).strftime("%Y%m%d_%H%M%S_%f")
        target = (camera_directory / f"{timestamp}_{safe_plate}.jpg").resolve()
        try:
            return save_jpeg_atomic(target, frame)
        except ImageWriteError as exc:
            LOGGER.warning("Не удалось сохранить кадр: %s", exc)
            self._safe_log_error("capture", str(exc), camera_id)
            return None

    def _handle_camera_error(self, camera_id: str, message: str) -> None:
        print(f"ОШИБКА КАМЕРЫ {camera_id}: {message}")
        self._safe_log_error("camera", message, camera_id)
        self._emit_error(message)

    def _safe_log_error(self, category: str, message: str, camera_id: str | None) -> None:
        try:
            self._repository.log_error(category, message, camera_id)
        except StorageError:
            LOGGER.exception("Не удалось записать ошибку в SQLite")

    def _emit_error(self, message: str) -> None:
        if self._error_handler is not None:
            self._error_handler(message)

    def _print_decision(
        self,
        event_id: int,
        event: RecognitionEvent,
        decision: AccessDecision,
        requires_approval: bool,
    ) -> None:
        plate = event.normalized_plate or event.raw_text
        local_time = event.observed_at.astimezone(self._timezone).strftime("%d.%m.%Y %H:%M:%S")
        suffix = " — ОЖИДАЕТ РУЧНОГО РЕШЕНИЯ" if requires_approval else ""
        print(
            f"[{local_time}] #{event_id} {plate} — {_STATUS_LABELS[event.decision]}: "
            f"{event.reason}{suffix}"
        )
        if decision.next_allowed_at is not None:
            next_time = decision.next_allowed_at.astimezone(self._timezone)
            print(f"  Следующий допустимый момент: {next_time:%d.%m.%Y %H:%M:%S}")

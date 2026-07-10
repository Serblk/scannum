from __future__ import annotations

import logging
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Sequence

from .config import CameraConfig


LOGGER = logging.getLogger(__name__)
CameraErrorHandler = Callable[[str, str], None]


class CameraDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    observed_at: datetime
    frame: Any


class CameraWorker(threading.Thread):
    def __init__(
        self,
        camera: CameraConfig,
        output_queue: queue.Queue[FramePacket],
        stop_event: threading.Event,
        process_every_n_frames: int,
        retry_seconds: float,
        error_handler: CameraErrorHandler,
    ) -> None:
        super().__init__(name=f"camera-{camera.id}", daemon=True)
        self._camera = camera
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._process_every_n_frames = process_every_n_frames
        self._retry_seconds = retry_seconds
        self._error_handler = error_handler

    def run(self) -> None:
        try:
            cv2 = _import_cv2()
        except CameraDependencyError as exc:
            self._error_handler(self._camera.id, str(exc))
            return

        while not self._stop_event.is_set():
            try:
                self._capture_until_failure(cv2)
            except Exception as exc:  # Камера должна переподключаться после аппаратного сбоя.
                message = f"Камера «{self._camera.name}» недоступна: {exc}"
                LOGGER.warning(message)
                self._error_handler(self._camera.id, message)
            self._stop_event.wait(self._retry_seconds)

    def _capture_until_failure(self, cv2: Any) -> None:
        capture = self._open_capture(cv2)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"не удалось открыть источник {self._camera.source!r}")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera.height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            LOGGER.info("Камера «%s» подключена", self._camera.name)

            frame_number = 0
            consecutive_failures = 0
            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        raise RuntimeError("видеопоток перестал отдавать кадры")
                    continue

                consecutive_failures = 0
                frame_number += 1
                if frame_number % self._process_every_n_frames != 0:
                    continue
                self._publish(
                    FramePacket(
                        camera_id=self._camera.id,
                        observed_at=datetime.now(UTC),
                        frame=frame,
                    )
                )
        finally:
            capture.release()

    def _open_capture(self, cv2: Any) -> Any:
        return _open_video_capture(cv2, self._camera.source)

    def _publish(self, packet: FramePacket) -> None:
        try:
            self._output_queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        try:
            self._output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._output_queue.put_nowait(packet)
        except queue.Full:
            LOGGER.debug("Кадр камеры %s пропущен: очередь занята", self._camera.id)


class CameraManager:
    def __init__(
        self,
        cameras: Sequence[CameraConfig],
        output_queue: queue.Queue[FramePacket],
        process_every_n_frames: int,
        retry_seconds: float,
        error_handler: CameraErrorHandler,
    ) -> None:
        self._stop_event = threading.Event()
        self._workers = [
            CameraWorker(
                camera=camera,
                output_queue=output_queue,
                stop_event=self._stop_event,
                process_every_n_frames=process_every_n_frames,
                retry_seconds=retry_seconds,
                error_handler=error_handler,
            )
            for camera in cameras
        ]

    def start(self) -> None:
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=3.0)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CameraDependencyError(
            "OpenCV не установлен. Выполните: python -m pip install -r requirements.txt"
        ) from exc
    return cv2


def probe_camera(camera: CameraConfig) -> tuple[int, int] | None:
    """Пытается получить один кадр и сразу освобождает камеру."""
    cv2 = _import_cv2()
    capture = _open_video_capture(cv2, camera.source)
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        for _ in range(30):
            success, frame = capture.read()
            if success and frame is not None:
                return int(frame.shape[1]), int(frame.shape[0])
        return None
    finally:
        capture.release()


def _open_video_capture(cv2: Any, source: int | str) -> Any:
    if sys.platform == "win32" and isinstance(source, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                return capture
            capture.release()
    return cv2.VideoCapture(source)

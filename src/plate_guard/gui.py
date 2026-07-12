from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .admin import (
    AdminError,
    AdminLockedError,
    AdminService,
    HistoryClearResult,
)
from .camera_discovery import LocalCamera, discover_local_cameras
from .camera_settings import (
    CameraSettingsError,
    load_selected_cameras,
    save_selected_cameras,
)
from .config import CameraConfig, ProjectConfig
from .exporter import ExportError, export_to_xlsx
from .models import AccessDecision, DecisionStatus, PlateCandidate, RecognitionEvent
from .service import PlateGuardService
from .storage import (
    HISTORY_COLUMNS,
    MANDATORY_HISTORY_COLUMNS,
    SQLiteRepository,
    StorageError,
)


_HISTORY_COLUMN_LABELS = {
    "time": "Время",
    "camera": "Камера",
    "plate": "Номер",
    "decision": "Проверка",
    "fueling": "Заправка",
    "mode": "Режим",
}


def run_gui(
    config: ProjectConfig,
    repository: SQLiteRepository,
    service: PlateGuardService,
) -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Контроль заправки")
    window = MainWindow(config, repository, service)
    window.show()
    return int(application.exec())


from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QObject
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _Bridge(QObject):
    frame_ready = Signal(str, object, object)
    event_ready = Signal(int, object, object, bool)
    error = Signal(str)


class VideoPanel(QLabel):
    def __init__(self) -> None:
        super().__init__("Ожидание видеопотока...")
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 460)
        self.setStyleSheet(
            "background:#0b1220; border:1px solid #263449; border-radius:14px; color:#8290a3;"
        )

    def show_frame(self, frame: Any, candidates: list[PlateCandidate]) -> None:
        try:
            import cv2
        except ImportError:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Segoe UI", max(13, width // 60), QFont.Weight.Bold))
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bounding_box
            valid = candidate.normalized_plate is not None
            color = QColor("#24d17e" if valid else "#ffbd2e")
            painter.setPen(QPen(color, max(2, width // 400)))
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            label = candidate.normalized_plate or candidate.raw_text
            painter.fillRect(x1, max(0, y1 - 32), max(130, len(label) * 16), 30, QColor(5, 12, 22, 220))
            painter.setPen(color)
            painter.drawText(x1 + 6, max(22, y1 - 8), label)
        painter.end()
        self._image = image
        self._render()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: ProjectConfig,
        repository: SQLiteRepository,
        service: PlateGuardService,
    ) -> None:
        super().__init__()
        self._config = config
        self._repository = repository
        self._service = service
        self._camera_settings_path = (
            config.app.database_path.parent.parent / "camera_sources.json"
        )
        try:
            self._saved_cameras = load_selected_cameras(self._camera_settings_path)
        except CameraSettingsError:
            self._saved_cameras = ()
        self._service.configure_cameras(())
        self._admin_service = AdminService(repository, config.app.captures_directory)
        self._timezone = ZoneInfo(config.app.timezone)
        self._service_thread: threading.Thread | None = None
        self._pending_event_id: int | None = None
        self._displayed_plate: str | None = None
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.timeout.connect(self._clear_current_display)
        self._bridge = _Bridge()
        self._bridge.frame_ready.connect(self._on_frame)
        self._bridge.event_ready.connect(self._on_event)
        self._bridge.error.connect(self._on_error)

        service.set_handlers(
            frame_handler=self._bridge.frame_ready.emit,
            event_handler=self._bridge.event_ready.emit,
            error_handler=self._bridge.error.emit,
        )

        self.setWindowTitle("Контроль заправки автомобилей")
        self.resize(1420, 880)
        self.setMinimumSize(1100, 700)
        self._build_ui()
        self._apply_theme()
        self._load_history()
        self._restore_latest_pending()
        QTimer.singleShot(100, self._start_service)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Контроль заправки")
        title.setObjectName("title")
        subtitle = QLabel("Автоматическое распознавание автомобильных номеров")
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.admin_button = QPushButton("Администрирование")
        self.admin_button.clicked.connect(self._open_administration)
        header.addWidget(self.admin_button)

        self.open_data_button = QPushButton("Открыть папку данных")
        self.open_data_button.clicked.connect(self._open_data_directory)
        header.addWidget(self.open_data_button)

        self.camera_settings_button = QPushButton("Настройка камер")
        self.camera_settings_button.clicked.connect(self._open_camera_settings)
        header.addWidget(self.camera_settings_button)

        self.manual_checkbox = QCheckBox("Ручное одобрение")
        self.manual_checkbox.setChecked(self._service.manual_approval_enabled)
        self.manual_checkbox.setToolTip(
            "Включено: оператор подтверждает заправку. Выключено: разрешённое событие запускает таймер автоматически."
        )
        self.manual_checkbox.toggled.connect(self._change_mode)
        header.addWidget(self.manual_checkbox)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.video_container = QWidget()
        self.video_grid = QGridLayout(self.video_container)
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        self.video_grid.setSpacing(10)
        self.video_scroll.setWidget(self.video_container)
        self._video_panels: dict[str, VideoPanel] = {}
        self._rebuild_video_grid(())
        content.addWidget(self.video_scroll, 3)
        content.addWidget(self._build_status_panel(), 2)
        root.addLayout(content, 4)

        history_label = QLabel("Последние события")
        history_label.setObjectName("section")
        root.addWidget(history_label)
        self.history = QTableWidget(0, 6)
        self.history.setHorizontalHeaderLabels(
            ["Время", "Камера", "Номер", "Проверка", "Заправка", "Режим"]
        )
        self.history.setAlternatingRowColors(True)
        self.history.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history.verticalHeader().setVisible(False)
        history_header = self.history.horizontalHeader()
        history_header.setMinimumSectionSize(80)
        history_header.setMaximumSectionSize(320)
        for column in range(5):
            history_header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        history_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._apply_history_column_visibility()
        self.history.setMinimumHeight(180)
        self.history.itemSelectionChanged.connect(self._select_history_pending)
        root.addWidget(self.history, 2)
        self.setCentralWidget(central)

    def _build_status_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Текущая камера"))
        camera_row.addStretch()
        self.camera_label = QLabel("—")
        self.camera_label.setObjectName("muted")
        camera_row.addWidget(self.camera_label)
        layout.addLayout(camera_row)

        self.plate_label = QLabel("—")
        self.plate_label.setObjectName("plate")
        self.plate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.plate_label)

        self.status_label = QLabel("ОЖИДАНИЕ")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.reason_label = QLabel("Поднесите номер автомобиля к камере")
        self.reason_label.setWordWrap(True)
        self.reason_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reason_label.setObjectName("muted")
        layout.addWidget(self.reason_label)

        self.confidence_label = QLabel("Уверенность OCR: —")
        self.confidence_label.setObjectName("muted")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.confidence_label)
        layout.addStretch()

        self.approval_hint = QLabel("Решение оператора")
        self.approval_hint.setObjectName("section")
        self.approval_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.approval_hint)

        buttons = QGridLayout()
        self.fueled_button = QPushButton("ЗАПРАВИЛСЯ")
        self.fueled_button.setObjectName("successButton")
        self.not_fueled_button = QPushButton("НЕ ЗАПРАВИЛСЯ")
        self.not_fueled_button.setObjectName("secondaryButton")
        self.fueled_button.clicked.connect(lambda: self._resolve_pending(True))
        self.not_fueled_button.clicked.connect(lambda: self._resolve_pending(False))
        buttons.addWidget(self.fueled_button, 0, 0)
        buttons.addWidget(self.not_fueled_button, 0, 1)
        layout.addLayout(buttons)
        self._set_approval_enabled(False)

        self.export_button = QPushButton("Экспортировать историю в Excel")
        self.export_button.clicked.connect(self._export_history)
        layout.addWidget(self.export_button)
        self.connection_label = QLabel("Запуск камеры...")
        self.connection_label.setObjectName("muted")
        self.connection_label.setWordWrap(True)
        layout.addWidget(self.connection_label)
        return panel

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#101824; color:#e8edf5; font:14px 'Segoe UI'; }
            QLabel#title { font-size:28px; font-weight:700; color:#ffffff; }
            QLabel#section { font-size:16px; font-weight:600; color:#dce5f2; }
            QLabel#muted { color:#93a2b7; }
            QLabel#plate { font-size:42px; font-weight:800; color:#ffffff; padding:14px; }
            QLabel#status { font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:#263449; }
            QFrame#card { background:#172231; border:1px solid #263449; border-radius:14px; }
            QCheckBox { spacing:10px; font-weight:600; }
            QPushButton { background:#263449; border:none; border-radius:9px; padding:12px 16px; font-weight:600; }
            QPushButton:hover { background:#31445d; }
            QPushButton:disabled { background:#1d2938; color:#5d6b7d; }
            QPushButton#successButton { background:#168653; color:white; }
            QPushButton#successButton:hover { background:#1aa566; }
            QPushButton#secondaryButton { background:#ad3b46; color:white; }
            QPushButton#secondaryButton:hover { background:#c84a56; }
            QTableWidget { background:#172231; alternate-background-color:#131e2b; border:1px solid #263449; border-radius:10px; gridline-color:#263449; }
            QHeaderView::section { background:#202e40; color:#cbd6e5; border:none; padding:8px; font-weight:600; }
            """
        )

    def _start_service(self) -> None:
        self._service_thread = threading.Thread(
            target=self._service.run,
            name="plate-guard-service",
            daemon=True,
        )
        self._service_thread.start()

    def _open_camera_settings(self) -> None:
        previous = self._service.active_cameras
        self._service.configure_cameras(())
        self._rebuild_video_grid(())
        self.connection_label.setText("Поиск встроенных и USB-камер...")
        QApplication.processEvents()
        try:
            dialog = CameraSelectionDialog(self._saved_cameras, self)
        except Exception as exc:
            self._service.configure_cameras(previous)
            self._rebuild_video_grid(previous)
            QMessageBox.critical(self, "Ошибка поиска камер", str(exc))
            return
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._service.configure_cameras(previous)
            self._rebuild_video_grid(previous)
            self.connection_label.setText("Настройка камер отменена")
            return
        selected = dialog.selected_cameras
        if not selected:
            QMessageBox.warning(self, "Камеры не выбраны", "Выберите хотя бы одну камеру.")
            self.connection_label.setText("Камеры не подключены")
            return
        try:
            save_selected_cameras(self._camera_settings_path, list(selected))
            self._repository.upsert_cameras(selected)
        except (CameraSettingsError, StorageError) as exc:
            QMessageBox.critical(self, "Не удалось сохранить камеры", str(exc))
            self.connection_label.setText("Камеры не подключены")
            return
        self._saved_cameras = selected
        self._rebuild_video_grid(selected)
        self._service.configure_cameras(selected)
        self.connection_label.setText(f"Подключение камер: {len(selected)}")

    def _open_data_directory(self) -> None:
        data_directory = self._config.app.database_path.parent.parent
        try:
            data_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Не удалось открыть данные", str(exc))
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_directory))):
            QMessageBox.warning(
                self,
                "Не удалось открыть данные",
                f"Откройте папку вручную:\n{data_directory}",
            )

    def _rebuild_video_grid(self, cameras: Any) -> None:
        while self.video_grid.count():
            item = self.video_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._video_panels = {}
        selected = tuple(cameras)
        if not selected:
            placeholder = QLabel("Камеры не подключены\nНажмите «Настройка камер»")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setObjectName("muted")
            self.video_grid.addWidget(placeholder, 0, 0)
            return
        columns = 1 if len(selected) == 1 else 2
        for index, camera in enumerate(selected):
            card = QFrame()
            card.setObjectName("card")
            layout = QVBoxLayout(card)
            title = QLabel(camera.name)
            title.setObjectName("section")
            panel = VideoPanel()
            panel.setMinimumSize(320, 220)
            layout.addWidget(title)
            layout.addWidget(panel, 1)
            self.video_grid.addWidget(card, index // columns, index % columns)
            self._video_panels[camera.id] = panel

    def _on_frame(
        self,
        camera_id: str,
        frame: Any,
        candidates: list[PlateCandidate],
    ) -> None:
        self.connection_label.setText(
            f"Подключено камер: {len(self._video_panels)}; распознавание работает"
        )
        panel = self._video_panels.get(camera_id)
        if panel is not None:
            panel.show_frame(frame, candidates)
        if self._displayed_plate and any(
            _candidate_matches(candidate, self._displayed_plate)
            for candidate in candidates
        ):
            self._restart_display_timer()

    def _on_event(
        self,
        event_id: int,
        event: RecognitionEvent,
        decision: AccessDecision,
        requires_approval: bool,
    ) -> None:
        plate = event.normalized_plate or event.raw_text
        self._displayed_plate = plate
        self.plate_label.setText(plate)
        self.camera_label.setText(event.camera_id)
        self.reason_label.setText(_display_reason(event.reason, decision, self._timezone))
        self.confidence_label.setText(f"Уверенность OCR: {event.ocr_confidence:.0%}")
        self._set_status(decision.status)
        self._pending_event_id = event_id if requires_approval else None
        self._set_approval_enabled(requires_approval)
        if requires_approval:
            self._display_timer.stop()
        else:
            self._restart_display_timer()
        outcome = "Ожидает решения" if requires_approval else (
            "Автоматически подтверждена"
            if event.decision is DecisionStatus.ALLOWED
            else "—"
        )
        mode = "Ручной" if requires_approval else (
            "Автоматический" if event.decision is DecisionStatus.ALLOWED else "—"
        )
        self._prepend_history_row(event_id, event, outcome, mode)

    def _on_error(self, message: str) -> None:
        self.connection_label.setText(message)

    def _set_status(self, status: DecisionStatus) -> None:
        labels = {
            DecisionStatus.ALLOWED: ("РАЗРЕШЕНО", "#168653"),
            DecisionStatus.DENIED: ("ЗАПРЕЩЕНО", "#ad3b46"),
            DecisionStatus.REVIEW: ("ТРЕБУЕТСЯ ПРОВЕРКА", "#a97817"),
        }
        text, color = labels[status]
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:{color}; color:white;"
        )

    def _set_approval_enabled(self, enabled: bool) -> None:
        visible = self._service.manual_approval_enabled or self._pending_event_id is not None
        self.approval_hint.setVisible(visible)
        self.fueled_button.setVisible(visible)
        self.not_fueled_button.setVisible(visible)
        self.fueled_button.setEnabled(enabled and visible)
        self.not_fueled_button.setEnabled(enabled and visible)

    def _change_mode(self, enabled: bool) -> None:
        try:
            self._service.set_manual_approval_enabled(enabled)
        except StorageError as exc:
            self.manual_checkbox.blockSignals(True)
            self.manual_checkbox.setChecked(not enabled)
            self.manual_checkbox.blockSignals(False)
            QMessageBox.critical(self, "Ошибка SQLite", str(exc))
            return
        self._set_approval_enabled(self._pending_event_id is not None)
        mode = "включено" if enabled else "выключено"
        self.connection_label.setText(f"Ручное одобрение {mode}")

    def _resolve_pending(self, fueled: bool) -> None:
        if self._pending_event_id is None:
            return
        try:
            self._service.resolve_manual_decision(self._pending_event_id, fueled)
        except (StorageError, LookupError) as exc:
            QMessageBox.critical(self, "Не удалось сохранить решение", str(exc))
            return
        outcome = "Заправился" if fueled else "Не заправился"
        self.reason_label.setText(
            "Заправка подтверждена, таймер восемь часов запущен"
            if fueled
            else "Отказ от заправки сохранён, таймер не запущен"
        )
        row = self._find_history_row(self._pending_event_id)
        if row is not None:
            outcome_item = QTableWidgetItem(outcome)
            outcome_item.setToolTip(outcome)
            mode_item = QTableWidgetItem("Ручной")
            mode_item.setToolTip("Ручной")
            self.history.setItem(row, 4, outcome_item)
            self.history.setItem(row, 5, mode_item)
            self._set_history_row_pending(row, False)
        self._pending_event_id = None
        self._set_approval_enabled(False)
        self._restart_display_timer()
        self._restore_latest_pending()

    def _restart_display_timer(self) -> None:
        if self._pending_event_id is not None:
            self._display_timer.stop()
            return
        try:
            seconds = self._service.display_timeout_seconds
        except StorageError:
            seconds = 10
        self._display_timer.start(seconds * 1000)

    def _clear_current_display(self) -> None:
        if self._pending_event_id is not None:
            return
        self._displayed_plate = None
        self.camera_label.setText("—")
        self.plate_label.setText("—")
        self.status_label.setText("ОЖИДАНИЕ")
        self.status_label.setStyleSheet(
            "font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:#263449; color:white;"
        )
        self.reason_label.setText("Ожидание автомобиля")
        self.confidence_label.setText("Уверенность OCR: —")

    def _prepend_history_row(
        self,
        event_id: int,
        event: RecognitionEvent,
        outcome: str,
        mode: str,
    ) -> int:
        self.history.insertRow(0)
        local_time = event.observed_at.astimezone(self._timezone).strftime("%d.%m.%Y %H:%M:%S")
        values = [
            local_time,
            event.camera_id,
            event.normalized_plate or event.raw_text,
            _decision_label(event.decision),
            outcome,
            mode,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            item.setData(Qt.ItemDataRole.UserRole, event_id)
            self.history.setItem(0, column, item)
        self._set_history_row_pending(0, outcome == "Ожидает решения")
        return 0

    def _load_history(self) -> None:
        try:
            rows = self._repository.recent_recognitions(50)
        except StorageError as exc:
            self.connection_label.setText(str(exc))
            return
        for row in reversed(rows):
            self.history.insertRow(0)
            observed = datetime.fromisoformat(row["observed_at"]).astimezone(self._timezone)
            outcome = _outcome_label(row.get("outcome"), row["decision"])
            mode = _mode_label(row.get("mode"))
            values = [
                observed.strftime("%d.%m.%Y %H:%M:%S"),
                row["camera_id"],
                row["normalized_plate"] or "нераспознан",
                _decision_label(DecisionStatus(row["decision"])),
                outcome,
                mode,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                self.history.setItem(0, column, item)
            self._set_history_row_pending(0, outcome == "Ожидает решения")

    def _set_history_row_pending(self, row: int, pending: bool) -> None:
        background = QColor("#7a2632" if pending else "#172231")
        foreground = QColor("#ffffff" if pending else "#e8edf5")
        for column in range(self.history.columnCount()):
            item = self.history.item(row, column)
            if item is not None:
                item.setBackground(background)
                item.setForeground(foreground)
                font = item.font()
                font.setBold(pending)
                item.setFont(font)

    def _apply_history_column_visibility(self) -> None:
        try:
            visible = set(self._service.history_visible_columns)
        except StorageError:
            visible = set(HISTORY_COLUMNS)
        for index, column in enumerate(HISTORY_COLUMNS):
            self.history.setColumnHidden(index, column not in visible)

    def _restore_latest_pending(self) -> None:
        try:
            pending = self._repository.latest_pending_recognition()
        except StorageError as exc:
            self.connection_label.setText(str(exc))
            return
        if pending is None:
            return
        self._pending_event_id = int(pending["id"])
        self._displayed_plate = pending["normalized_plate"]
        self._display_timer.stop()
        self.plate_label.setText(pending["normalized_plate"])
        self.status_label.setText("ОЖИДАЕТ РЕШЕНИЯ")
        self.status_label.setStyleSheet(
            "font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:#a97817; color:white;"
        )
        self.reason_label.setText("Укажите, состоялась ли фактическая заправка")
        self.confidence_label.setText(f"Уверенность OCR: {pending['ocr_confidence']:.0%}")
        self._set_approval_enabled(True)

    def _select_history_pending(self) -> None:
        selected = self.history.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        outcome_item = self.history.item(row, 4)
        id_item = self.history.item(row, 0)
        if outcome_item is None or id_item is None or outcome_item.text() != "Ожидает решения":
            return
        event_id = id_item.data(Qt.ItemDataRole.UserRole)
        if event_id is None:
            return
        self._pending_event_id = int(event_id)
        self._display_timer.stop()
        plate_item = self.history.item(row, 2)
        if plate_item is not None:
            self._displayed_plate = plate_item.text()
            self.plate_label.setText(plate_item.text())
        self.status_label.setText("ОЖИДАЕТ РЕШЕНИЯ")
        self.status_label.setStyleSheet(
            "font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:#a97817; color:white;"
        )
        self.reason_label.setText("Укажите, состоялась ли фактическая заправка")
        self._set_approval_enabled(True)

    def _find_history_row(self, event_id: int) -> int | None:
        for row in range(self.history.rowCount()):
            item = self.history.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == event_id:
                return row
        return None

    def _export_history(self) -> None:
        output = self._config.app.reports_directory / f"events_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        try:
            target = export_to_xlsx(self._repository.export_rows(), output)
        except (StorageError, ExportError) as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return
        QMessageBox.information(self, "Экспорт завершён", f"Файл создан:\n{target}")

    def _open_administration(self) -> None:
        if not self._admin_service.password_is_configured:
            if not self._create_admin_password():
                return
        elif not self._authenticate_admin():
            return

        dialog = AdminPanelDialog(
            admin_service=self._admin_service,
            plate_service=self._service,
            history_cleared=self._reset_after_history_clear,
            columns_changed=self._apply_history_column_visibility,
            parent=self,
        )
        dialog.exec()

    def _create_admin_password(self) -> bool:
        while True:
            dialog = PasswordDialog(
                title="Создание пароля администратора",
                message="Первый вход: создайте пароль длиной не менее 10 символов.",
                ask_confirmation=True,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
            try:
                self._admin_service.create_password(
                    dialog.password, dialog.confirmation
                )
            except AdminError as exc:
                QMessageBox.warning(self, "Пароль не создан", str(exc))
                continue
            QMessageBox.information(
                self,
                "Пароль создан",
                "Пароль администратора сохранён в защищённом виде.",
            )
            return True

    def _authenticate_admin(self) -> bool:
        while True:
            dialog = PasswordDialog(
                title="Вход в администрирование",
                message="Введите пароль администратора.",
                ask_confirmation=False,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
            try:
                self._admin_service.verify_password(dialog.password)
            except AdminLockedError as exc:
                QMessageBox.critical(self, "Вход заблокирован", str(exc))
                return False
            except AdminError as exc:
                QMessageBox.warning(self, "Вход не выполнен", str(exc))
                continue
            return True

    def _reset_after_history_clear(self) -> None:
        self.history.setRowCount(0)
        self._pending_event_id = None
        self._displayed_plate = None
        self._display_timer.stop()
        self._set_approval_enabled(False)
        self.plate_label.setText("—")
        self.status_label.setText("ОЖИДАНИЕ")
        self.status_label.setStyleSheet(
            "font-size:25px; font-weight:800; padding:14px; border-radius:10px; background:#263449; color:white;"
        )
        self.reason_label.setText("История очищена. Ожидание автомобиля.")
        self.confidence_label.setText("Уверенность OCR: —")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._service.stop()
        if self._service_thread is not None:
            self._service_thread.join(timeout=4.0)
        event.accept()


def _decision_label(status: DecisionStatus) -> str:
    return {
        DecisionStatus.ALLOWED: "Разрешено",
        DecisionStatus.DENIED: "Запрещено",
        DecisionStatus.REVIEW: "Проверка",
    }[status]


def _outcome_label(outcome: str | None, decision: str) -> str:
    if outcome == "FUELED":
        return "Заправился"
    if outcome == "NOT_FUELED":
        return "Не заправился"
    return "Ожидает решения" if decision == "ALLOWED" else "—"


def _mode_label(mode: str | None) -> str:
    if mode == "MANUAL":
        return "Ручной"
    if mode == "AUTO":
        return "Автоматический"
    return "—"


def _candidate_matches(candidate: PlateCandidate, displayed_plate: str) -> bool:
    return displayed_plate in {
        candidate.normalized_plate,
        candidate.raw_text,
        candidate.canonical_text,
    }


def _display_reason(
    reason: str,
    decision: AccessDecision,
    timezone: ZoneInfo,
) -> str:
    if decision.next_allowed_at is None:
        return reason
    next_time = decision.next_allowed_at.astimezone(timezone).strftime("%d.%m.%Y %H:%M:%S")
    return f"{reason}\n\nСледующий допустимый момент:\n{next_time}"


class CameraSelectionDialog(QDialog):
    def __init__(
        self,
        saved_cameras: tuple[CameraConfig, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._saved_sources = {camera.source for camera in saved_cameras}
        self._checkboxes: dict[int, QCheckBox] = {}
        self.setWindowTitle("Настройка камер")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._layout = QVBoxLayout(self)
        title = QLabel("Встроенные и USB-камеры")
        title.setObjectName("section")
        self._layout.addWidget(title)
        explanation = QLabel(
            "Отметьте все камеры, которые должны одновременно распознавать номера. "
            "Во время поиска видеопотоки временно отключаются."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("muted")
        self._layout.addWidget(explanation)

        self._camera_box = QVBoxLayout()
        self._layout.addLayout(self._camera_box)
        self._refresh_button = QPushButton("Обновить список")
        self._refresh_button.clicked.connect(self._refresh)
        self._layout.addWidget(self._refresh_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Подключить выбранные")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        self._layout.addWidget(buttons)
        self._refresh()

    @property
    def selected_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(
            CameraConfig(
                id=f"local-{source}",
                name=f"Камера {source} (встроенная/USB)",
                source=source,
                enabled=True,
                width=1280,
                height=720,
            )
            for source, checkbox in sorted(self._checkboxes.items())
            if checkbox.isChecked()
        )

    def _refresh(self) -> None:
        checked = {
            source
            for source, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        } or set(self._saved_sources)
        while self._camera_box.count():
            item = self._camera_box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes = {}
        self._refresh_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cameras = discover_local_cameras()
        finally:
            QApplication.restoreOverrideCursor()
            self._refresh_button.setEnabled(True)
        if not cameras:
            label = QLabel("Доступные камеры не найдены. Проверьте подключение и разрешения Windows.")
            label.setWordWrap(True)
            label.setObjectName("muted")
            self._camera_box.addWidget(label)
            return
        for camera in cameras:
            checkbox = QCheckBox(camera.name)
            checkbox.setChecked(camera.source in checked)
            self._checkboxes[camera.source] = checkbox
            self._camera_box.addWidget(checkbox)

    def _accept_selection(self) -> None:
        if not self.selected_cameras:
            QMessageBox.warning(self, "Камеры не выбраны", "Выберите хотя бы одну камеру.")
            return
        self.accept()


class PasswordDialog(QDialog):
    def __init__(
        self,
        title: str,
        message: str,
        ask_confirmation: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        description = QLabel(message)
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("Пароль администратора")
        form.addRow("Пароль:", self.password_edit)
        self.confirmation_edit: QLineEdit | None = None
        if ask_confirmation:
            self.confirmation_edit = QLineEdit()
            self.confirmation_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.confirmation_edit.setPlaceholderText("Повторите пароль")
            form.addRow("Повтор:", self.confirmation_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.password_edit.setFocus()

    @property
    def password(self) -> str:
        return self.password_edit.text()

    @property
    def confirmation(self) -> str:
        return self.confirmation_edit.text() if self.confirmation_edit else ""


class AdminPanelDialog(QDialog):
    def __init__(
        self,
        admin_service: AdminService,
        plate_service: PlateGuardService,
        history_cleared: Any,
        columns_changed: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._admin_service = admin_service
        self._plate_service = plate_service
        self._history_cleared = history_cleared
        self._columns_changed = columns_changed
        self.setWindowTitle("Администрирование")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        title = QLabel("Администрирование")
        title.setObjectName("title")
        layout.addWidget(title)
        explanation = QLabel(
            "Очистка удаляет события, решения, подтверждённые заправки, журнал ошибок "
            "и фотографии. Камеры, настройки, модели и пароль сохраняются."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        timeout_form = QFormLayout()
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(3, 120)
        self.timeout_spin.setSuffix(" сек.")
        try:
            self.timeout_spin.setValue(self._plate_service.display_timeout_seconds)
        except StorageError:
            self.timeout_spin.setValue(10)
        timeout_form.addRow("Очищать текущий номер через:", self.timeout_spin)
        layout.addLayout(timeout_form)
        self.save_timeout_button = QPushButton("Сохранить время отображения")
        self.save_timeout_button.clicked.connect(self._save_display_timeout)
        layout.addWidget(self.save_timeout_button)

        columns_title = QLabel("Колонки таблицы «Последние события»")
        columns_title.setObjectName("section")
        layout.addWidget(columns_title)
        try:
            visible_columns = set(self._plate_service.history_visible_columns)
        except StorageError:
            visible_columns = set(HISTORY_COLUMNS)
        columns_grid = QGridLayout()
        self.column_checkboxes: dict[str, QCheckBox] = {}
        for index, column in enumerate(HISTORY_COLUMNS):
            checkbox = QCheckBox(_HISTORY_COLUMN_LABELS[column])
            checkbox.setChecked(column in visible_columns)
            if column in MANDATORY_HISTORY_COLUMNS:
                checkbox.setEnabled(False)
                checkbox.setToolTip("Обязательная колонка")
            self.column_checkboxes[column] = checkbox
            columns_grid.addWidget(checkbox, index // 3, index % 3)
        layout.addLayout(columns_grid)
        self.save_columns_button = QPushButton("Сохранить видимые колонки")
        self.save_columns_button.clicked.connect(self._save_visible_columns)
        layout.addWidget(self.save_columns_button)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("muted")
        layout.addWidget(self.summary_label)

        self.clear_button = QPushButton("Очистить историю и фотографии")
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self._clear_history)
        layout.addWidget(self.clear_button)

        close_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_button.rejected.connect(self.reject)
        layout.addWidget(close_button)
        self._refresh_summary()

    def _save_display_timeout(self) -> None:
        try:
            self._plate_service.set_display_timeout_seconds(self.timeout_spin.value())
        except (StorageError, ValueError) as exc:
            QMessageBox.critical(self, "Настройка не сохранена", str(exc))
            return
        QMessageBox.information(
            self,
            "Настройка сохранена",
            f"Последний номер будет отображаться {self.timeout_spin.value()} секунд.",
        )

    def _save_visible_columns(self) -> None:
        selected = [
            column
            for column, checkbox in self.column_checkboxes.items()
            if checkbox.isChecked()
        ]
        try:
            self._plate_service.set_history_visible_columns(selected)
        except (StorageError, ValueError) as exc:
            QMessageBox.critical(self, "Настройка не сохранена", str(exc))
            return
        self._columns_changed()
        QMessageBox.information(
            self,
            "Настройка сохранена",
            "Набор колонок таблицы обновлён.",
        )

    def _refresh_summary(self) -> None:
        try:
            preview = self._admin_service.preview_history_clear()
        except (AdminError, StorageError) as exc:
            self.summary_label.setText(f"Не удалось получить статистику: {exc}")
            self.clear_button.setEnabled(False)
            return
        self.summary_label.setText(
            f"Распознаваний: {preview.recognitions}\n"
            f"Подтверждённых заправок: {preview.fuelings}\n"
            f"Решений операторов/системы: {preview.decisions}\n"
            f"Ошибок: {preview.errors}\n"
            f"Фотографий: {preview.photos}"
        )

    def _clear_history(self) -> None:
        try:
            preview = self._admin_service.preview_history_clear()
        except (AdminError, StorageError) as exc:
            QMessageBox.critical(self, "Не удалось прочитать историю", str(exc))
            return
        confirmation = HistoryClearConfirmation(
            preview=preview,
            parent=self,
        )
        confirmation.exec()
        if not confirmation.confirmed:
            return

        password_dialog = PasswordDialog(
            title="Подтверждение очистки",
            message="Повторно введите пароль администратора.",
            ask_confirmation=False,
            parent=self,
        )
        if password_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result: HistoryClearResult = self._plate_service.perform_maintenance(
                lambda: self._admin_service.clear_history_and_photos(
                    password_dialog.password
                )
            )
        except AdminLockedError as exc:
            QMessageBox.critical(self, "Очистка заблокирована", str(exc))
            return
        except AdminError as exc:
            QMessageBox.critical(self, "Очистка не выполнена", str(exc))
            return

        self._history_cleared()
        self._refresh_summary()
        message = (
            f"Удалено событий: {result.preview.recognitions}\n"
            f"Удалено фотографий: {result.preview.photos}"
        )
        if result.cleanup_warning:
            QMessageBox.warning(self, "Очистка завершена с предупреждением", message + "\n\n" + result.cleanup_warning)
        else:
            QMessageBox.information(self, "Очистка завершена", message)


class HistoryClearConfirmation(QMessageBox):
    def __init__(self, preview: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(QMessageBox.Icon.Warning)
        self.setWindowTitle("Необратимая очистка")
        self.setText(
            "Будут безвозвратно удалены:\n"
            f"• {preview.recognitions} событий\n"
            f"• {preview.fuelings} заправок\n"
            f"• {preview.errors} ошибок\n"
            f"• {preview.photos} фотографий\n\n"
            "Продолжить?"
        )
        self.delete_button = self.addButton(
            "Да, удалить",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        self.cancel_button = self.addButton(
            "Отмена",
            QMessageBox.ButtonRole.RejectRole,
        )
        self.setDefaultButton(self.cancel_button)
        self.setEscapeButton(self.cancel_button)

    @property
    def confirmed(self) -> bool:
        return self.clickedButton() is self.delete_button

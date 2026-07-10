from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .models import PlateCandidate
from .plates import PlateValidationError, canonicalize_ocr_text, normalize_plate


class RecognitionDependencyError(RuntimeError):
    pass


class FastAlprRecognizer:
    def __init__(
        self,
        detector_model: str,
        ocr_model: str,
        device: str,
        model_cache_directory: str | Path,
    ) -> None:
        try:
            from fast_plate_ocr.inference import hub as ocr_hub
            from fast_alpr import ALPR
            from open_image_models.detection.core import hub as detector_hub
        except ImportError as exc:
            raise RecognitionDependencyError(
                "FastALPR не установлен. Выполните: python -m pip install -r requirements.txt"
            ) from exc

        try:
            cache_directory = Path(model_cache_directory).resolve()
            cache_directory.mkdir(parents=True, exist_ok=True)
            detector_hub.MODEL_CACHE_DIR = cache_directory / "detector"
            ocr_hub.MODEL_CACHE_DIR = cache_directory / "ocr"
            self._engine = ALPR(
                detector_model=detector_model,
                ocr_model=ocr_model,
                ocr_device=device,
            )
        except Exception as exc:
            raise RecognitionDependencyError(
                "Не удалось загрузить модели распознавания. "
                "При первом запуске требуется интернет: "
                f"{exc}"
            ) from exc

    def recognize(self, frame: Any) -> list[PlateCandidate]:
        results = self._engine.predict(frame)
        candidates: list[PlateCandidate] = []
        for result in results:
            ocr = result.ocr
            if ocr is None or not ocr.text:
                continue

            raw_text = str(ocr.text).strip()
            canonical_text = canonicalize_ocr_text(raw_text)
            if not canonical_text:
                continue
            try:
                normalized_plate = normalize_plate(canonical_text)
                validation_error = None
            except PlateValidationError as exc:
                normalized_plate = None
                validation_error = str(exc)

            bounding_box = result.detection.bounding_box
            candidates.append(
                PlateCandidate(
                    raw_text=raw_text,
                    canonical_text=canonical_text,
                    normalized_plate=normalized_plate,
                    ocr_confidence=_average_confidence(ocr.confidence),
                    detection_confidence=_clamp_confidence(result.detection.confidence),
                    bounding_box=(
                        int(bounding_box.x1),
                        int(bounding_box.y1),
                        int(bounding_box.x2),
                        int(bounding_box.y2),
                    ),
                    validation_error=validation_error,
                )
            )
        return candidates


def _average_confidence(value: float | Iterable[float]) -> float:
    if isinstance(value, (int, float)):
        return _clamp_confidence(float(value))
    values = [float(item) for item in value]
    return _clamp_confidence(mean(values)) if values else 0.0


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))

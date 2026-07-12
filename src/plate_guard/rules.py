from __future__ import annotations

from datetime import datetime, timedelta

from .models import AccessDecision, DecisionStatus
from .plates import PlateValidationError, parse_plate


def evaluate_access(
    plate: str,
    observed_at: datetime,
    last_confirmed_fueling_at: datetime | None,
    fueling_interval: timedelta = timedelta(hours=8),
) -> AccessDecision:
    _require_aware(observed_at, "Время распознавания")
    if fueling_interval <= timedelta(0):
        raise ValueError("Интервал между заправками должен быть положительным")

    try:
        parts = parse_plate(plate)
    except PlateValidationError as exc:
        return AccessDecision(DecisionStatus.REVIEW, str(exc))

    number_is_even = int(parts.number[0]) % 2 == 0
    day_is_even = observed_at.day % 2 == 0
    if number_is_even != day_is_even:
        expected = "чётной" if day_is_even else "нечётной"
        expected_digits = "0, 2, 4, 6, 8" if day_is_even else "1, 3, 5, 7, 9"
        return AccessDecision(
            DecisionStatus.DENIED,
            f"Сегодня допускаются номера с {expected} первой цифрой числовой части "
            f"({expected_digits})",
        )

    if last_confirmed_fueling_at is None:
        return AccessDecision(DecisionStatus.ALLOWED, "Ограничения не обнаружены")

    _require_aware(last_confirmed_fueling_at, "Время последней заправки")
    elapsed = observed_at - last_confirmed_fueling_at.astimezone(observed_at.tzinfo)
    if elapsed < timedelta(0):
        return AccessDecision(
            DecisionStatus.REVIEW,
            "Системное время раньше последней подтверждённой заправки",
        )

    next_allowed_at = last_confirmed_fueling_at + fueling_interval
    if elapsed < fueling_interval:
        return AccessDecision(
            DecisionStatus.DENIED,
            "После последней подтверждённой заправки ещё не прошло восемь часов",
            next_allowed_at=next_allowed_at,
        )

    return AccessDecision(DecisionStatus.ALLOWED, "Ограничения не обнаружены")


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} должно содержать часовой пояс")

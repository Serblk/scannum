from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta, timezone

from plate_guard.models import DecisionStatus
from plate_guard.rules import evaluate_access


MOSCOW = timezone(timedelta(hours=3))


class AccessRuleTests(unittest.TestCase):
    def test_even_day_allows_even_first_digit(self) -> None:
        result = evaluate_access("А030ВС77", datetime(2026, 7, 10, 12, tzinfo=MOSCOW), None)
        self.assertEqual(result.status, DecisionStatus.ALLOWED)

    def test_odd_day_allows_odd_first_digit(self) -> None:
        result = evaluate_access("А130ВС77", datetime(2026, 7, 11, 12, tzinfo=MOSCOW), None)
        self.assertEqual(result.status, DecisionStatus.ALLOWED)

    def test_parity_mismatch_is_denied(self) -> None:
        result = evaluate_access("А130ВС77", datetime(2026, 7, 10, 12, tzinfo=MOSCOW), None)
        self.assertEqual(result.status, DecisionStatus.DENIED)

    def test_attempt_before_eight_hours_is_denied(self) -> None:
        last = datetime(2026, 7, 10, 1, tzinfo=UTC)
        result = evaluate_access(
            "А030ВС77",
            last + timedelta(hours=7, minutes=59),
            last,
        )
        self.assertEqual(result.status, DecisionStatus.DENIED)
        self.assertEqual(result.next_allowed_at, last + timedelta(hours=8))

    def test_attempt_exactly_after_eight_hours_is_allowed(self) -> None:
        last = datetime(2026, 7, 10, 1, tzinfo=UTC)
        result = evaluate_access("А030ВС77", last + timedelta(hours=8), last)
        self.assertEqual(result.status, DecisionStatus.ALLOWED)

    def test_clock_rollback_requires_review(self) -> None:
        last = datetime(2026, 7, 10, 12, tzinfo=UTC)
        result = evaluate_access("А030ВС77", last - timedelta(minutes=1), last)
        self.assertEqual(result.status, DecisionStatus.REVIEW)

    def test_naive_datetime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_access("А030ВС77", datetime(2026, 7, 10, 12), None)


if __name__ == "__main__":
    unittest.main()

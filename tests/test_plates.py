from __future__ import annotations

import unittest

from plate_guard.plates import PlateValidationError, canonicalize_ocr_text, normalize_plate


class PlateNormalizationTests(unittest.TestCase):
    def test_normalizes_latin_lookalikes_and_separators(self) -> None:
        self.assertEqual(normalize_plate("a 030 bc-77"), "А030ВС77")

    def test_preserves_leading_zero(self) -> None:
        self.assertEqual(normalize_plate("А030ВС777"), "А030ВС777")

    def test_corrects_zero_to_letter_o_in_letter_positions(self) -> None:
        self.assertEqual(normalize_plate("0533CM46"), "О533СМ46")
        self.assertEqual(normalize_plate("A533C046"), "А533СО46")

    def test_corrects_letter_o_to_zero_in_digit_positions(self) -> None:
        self.assertEqual(normalize_plate("AО33BC77"), "А033ВС77")

    def test_rejects_non_plate_letters(self) -> None:
        with self.assertRaises(PlateValidationError):
            normalize_plate("Д123АА77")

    def test_rejects_wrong_number_length(self) -> None:
        with self.assertRaises(PlateValidationError):
            normalize_plate("А30ВС77")

    def test_canonicalization_does_not_silently_remove_unknown_characters(self) -> None:
        self.assertEqual(canonicalize_ocr_text("A!123BC77"), "А!123ВС77")
        with self.assertRaises(PlateValidationError):
            normalize_plate("A!123BC77")


if __name__ == "__main__":
    unittest.main()

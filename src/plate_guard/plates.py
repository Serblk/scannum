from __future__ import annotations

import re
from dataclasses import dataclass


RUSSIAN_PLATE_LETTERS = "АВЕКМНОРСТУХ"
_LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "E": "Е",
        "K": "К",
        "M": "М",
        "H": "Н",
        "O": "О",
        "P": "Р",
        "C": "С",
        "T": "Т",
        "Y": "У",
        "X": "Х",
    }
)
_SEPARATORS = re.compile(r"[\s\-_.]+")
_PLATE_PATTERN = re.compile(
    rf"^(?P<prefix>[{RUSSIAN_PLATE_LETTERS}])"
    rf"(?P<number>\d{{3}})"
    rf"(?P<suffix>[{RUSSIAN_PLATE_LETTERS}]{{2}})"
    rf"(?P<region>\d{{2,3}})$"
)


class PlateValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlateParts:
    prefix: str
    number: str
    suffix: str
    region: str

    @property
    def full_plate(self) -> str:
        return f"{self.prefix}{self.number}{self.suffix}{self.region}"


def canonicalize_ocr_text(value: str) -> str:
    """Приводит похожие латинские буквы к кириллице, не скрывая прочие ошибки OCR."""
    if not isinstance(value, str):
        raise TypeError("Номер должен быть строкой")
    return _SEPARATORS.sub("", value.strip().upper()).translate(_LATIN_TO_CYRILLIC)


def parse_plate(value: str) -> PlateParts:
    canonical = canonicalize_ocr_text(value)
    match = _PLATE_PATTERN.fullmatch(canonical)
    if match is None:
        raise PlateValidationError(
            "Ожидается стандартный номер вида А123ВС77 или А123ВС777"
        )
    return PlateParts(**match.groupdict())


def normalize_plate(value: str) -> str:
    return parse_plate(value).full_plate

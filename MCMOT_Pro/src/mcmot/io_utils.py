"""JSONL yazma yardimcilari ve kontrat alan adlari.

tracks_<cam>.jsonl kayit semasi PROJE_REHBERI.md Bolum 6'daki veri kontratina
baglidir; alan adlari TRACK_FIELDS uzerinden tek yerde tutulur.
"""

import json
from pathlib import Path
from typing import Iterable, Iterator

# tracks_<cam>.jsonl kaydindaki zorunlu ust duzey alanlar (kontrat).
TRACK_FIELDS = (
    "timestamp",
    "frame",
    "camera_id",
    "track_id",
    "class",
    "conf",
    "bbox_xyxy",
    "foot_point",
    "hints",
)

# hints alt alanlari (kontrat).
HINT_FIELDS = ("dominant_color", "size_class", "aspect_ratio")


class JsonlWriter:
    """Kayitlari satir satir, akis halinde JSONL dosyasina yazar.

    Dosya acilirken sifirlanir (append degil); with blogu ile kullanilir.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter with blogu icinde kullanilmali")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


def read_jsonl(path: Path) -> Iterator[dict]:
    """JSONL dosyasini satir satir okur (test/dogrulama icin)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_track_record(record: dict) -> None:
    """Kaydin kontrat alanlarini icerdigini dogrular; eksikse ValueError."""
    missing = [f for f in TRACK_FIELDS if f not in record]
    if missing:
        raise ValueError(f"kontrat alanlari eksik: {missing}")
    missing_hints = [f for f in HINT_FIELDS if f not in record["hints"]]
    if missing_hints:
        raise ValueError(f"hints alanlari eksik: {missing_hints}")

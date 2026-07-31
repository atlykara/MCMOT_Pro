"""Takip kalite metrikleri: track ozetleri ve supheli ID degisimi tespiti.

Bu modul hem `01_track_quality.py` (tek kosu raporu) hem de
`01_sweep_tracker.py` (parametre kiyasi) tarafindan paylasilir; supheli
switch kurali tek yerde tanimlidir. Kayitlar tracks_<cam>.jsonl kontratina
uygun sozluklerdir (dosyadan okunmus ya da bellekten gelmis olabilir).
"""

from pathlib import Path
from typing import Iterable

from mcmot.io_utils import read_jsonl

SHORT_TRACK_MAX_SEC = 1.5   # bundan kisa yasayan track'ler hayalet/bolunme adayi
SWITCH_MAX_GAP_SEC = 1.0    # eski track kaybolduktan sonra yeni dogum icin azami sure
SWITCH_MAX_DIST_PX = 75.0   # eski son foot_point ile yeni ilk foot_point arasi azami mesafe


class TrackSummary:
    """Bir track_id'nin dogum/olum bilgisi (akis okumada guncellenir)."""

    __slots__ = ("track_id", "first_ts", "last_ts", "first_frame", "last_frame",
                 "first_foot", "last_foot", "first_bbox", "last_bbox", "count")

    def __init__(self, track_id: int, record: dict):
        self.track_id = track_id
        self.first_ts = self.last_ts = record["timestamp"]
        self.first_frame = self.last_frame = record["frame"]
        self.first_foot = self.last_foot = record["foot_point"]
        self.first_bbox = self.last_bbox = record["bbox_xyxy"]
        self.count = 1

    def update(self, record: dict) -> None:
        self.count += 1
        ts = record["timestamp"]
        if ts >= self.last_ts:
            self.last_ts = ts
            self.last_frame = record["frame"]
            self.last_foot = record["foot_point"]
            self.last_bbox = record["bbox_xyxy"]

    @property
    def lifetime(self) -> float:
        return self.last_ts - self.first_ts


def add_record(summaries: dict, record: dict) -> None:
    """Tek kaydi ozet sozlugune ekler (yerinde); yeni track_id ise olusturur."""
    tid = record["track_id"]
    summary = summaries.get(tid)
    if summary is None:
        summaries[tid] = TrackSummary(tid, record)
    else:
        summary.update(record)


def summarize_records(records: Iterable[dict]) -> tuple:
    """Kayit akisindan (ozetler, kayit_sayisi, min_kare, max_kare) uretir."""
    summaries = {}
    record_count = 0
    min_frame = None
    max_frame = None
    for record in records:
        record_count += 1
        frame = record["frame"]
        min_frame = frame if min_frame is None else min(min_frame, frame)
        max_frame = frame if max_frame is None else max(max_frame, frame)
        add_record(summaries, record)
    return summaries, record_count, min_frame, max_frame


def collect_summaries(jsonl_path: Path) -> tuple:
    """JSONL'i akis halinde okuyup track ozetlerini ve kare sayaclarini toplar."""
    return summarize_records(read_jsonl(jsonl_path))


def find_suspect_switches(summaries: dict) -> list:
    """Kaybolan track'in yakininda kisa sure icinde dogan yeni track'leri bulur.

    Donen liste: (eski_id, yeni_id, timestamp, mesafe_px) — timestamp yeni
    track'in dogum ani. Her yeni track en yakin eski track ile eslestirilir.
    """
    tracks = sorted(summaries.values(), key=lambda t: t.first_ts)
    switches = []
    for new in tracks:
        best = None
        for old in tracks:
            if old.track_id == new.track_id:
                continue
            gap = new.first_ts - old.last_ts
            if not (0.0 < gap <= SWITCH_MAX_GAP_SEC):
                continue
            dx = new.first_foot[0] - old.last_foot[0]
            dy = new.first_foot[1] - old.last_foot[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= SWITCH_MAX_DIST_PX:
                continue
            if best is None or dist < best[1]:
                best = (old, dist)
        if best is not None:
            old, dist = best
            switches.append((old.track_id, new.track_id, new.first_ts, dist))
    switches.sort(key=lambda s: s[2])
    return switches

"""tracks_<cam>.jsonl kontrat testleri.

Amac: camA ve camB hatlarinin urettigi dosyalarin Faz 3'te birlesebilmesi
icin formatin birebir ayni oldugunu garanti etmek. outputs/tracks/ altindaki
tum tracks_*.jsonl dosyalarinin ilk MAX_LINES satiri dogrulanir.
"""

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"

MAX_LINES = 500
FOOT_POINT_TOL = 0.06  # bbox degerleri 1 haneye yuvarlandigi icin kucuk tolerans

EXPECTED_KEYS = {
    "timestamp", "frame", "camera_id", "track_id", "class",
    "conf", "bbox_xyxy", "foot_point", "hints",
}
EXPECTED_HINT_KEYS = {"dominant_color", "size_class", "aspect_ratio"}
ALLOWED_CLASSES = {"car", "bus", "truck"}
ALLOWED_SIZE_CLASSES = {"small", "medium", "large"}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _param_files():
    files = sorted(TRACKS_DIR.glob("tracks_*.jsonl"))
    if not files:
        reason = f"{TRACKS_DIR} altinda tracks_*.jsonl yok"
        return [pytest.param(None, id="tracks-yok", marks=pytest.mark.skip(reason=reason))]
    return [pytest.param(f, id=f.name) for f in files]


def _read_records(path: Path):
    """Ilk MAX_LINES satiri (satir_no, kayit) olarak dondurur."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line_no > MAX_LINES:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"{path.name}:{line_no} gecerli JSON degil: {exc}")
            records.append((line_no, record))
    assert records, f"{path.name} bos, en az bir kayit bekleniyordu"
    return records


@pytest.mark.parametrize("path", _param_files())
def test_track_record_schema(path: Path):
    for line_no, rec in _read_records(path):
        where = f"{path.name}:{line_no}"

        assert set(rec.keys()) == EXPECTED_KEYS, (
            f"{where} anahtarlar kontrata uymuyor: "
            f"eksik={EXPECTED_KEYS - set(rec)}, fazla={set(rec) - EXPECTED_KEYS}"
        )

        assert isinstance(rec["timestamp"], float), f"{where} timestamp float degil: {rec['timestamp']!r}"
        assert _is_int(rec["frame"]), f"{where} frame int degil: {rec['frame']!r}"
        assert isinstance(rec["camera_id"], str) and rec["camera_id"], f"{where} camera_id bos/gecersiz"
        assert _is_int(rec["track_id"]), f"{where} track_id int degil: {rec['track_id']!r}"
        assert rec["class"] in ALLOWED_CLASSES, f"{where} class gecersiz: {rec['class']!r}"
        assert isinstance(rec["conf"], float) and 0.0 <= rec["conf"] <= 1.0, (
            f"{where} conf 0-1 arasi float degil: {rec['conf']!r}"
        )

        bbox = rec["bbox_xyxy"]
        assert isinstance(bbox, list) and len(bbox) == 4 and all(_is_number(v) for v in bbox), (
            f"{where} bbox_xyxy 4 sayi degil: {bbox!r}"
        )
        x1, y1, x2, y2 = bbox
        assert x2 > x1 and y2 > y1, f"{where} bbox gecersiz (x2>x1, y2>y1 olmali): {bbox}"

        foot = rec["foot_point"]
        assert isinstance(foot, list) and len(foot) == 2 and all(_is_number(v) for v in foot), (
            f"{where} foot_point 2 sayi degil: {foot!r}"
        )
        assert math.isclose(foot[0], (x1 + x2) / 2, abs_tol=FOOT_POINT_TOL), (
            f"{where} foot_point[0] != (x1+x2)/2: {foot[0]} vs {(x1 + x2) / 2}"
        )
        assert math.isclose(foot[1], y2, abs_tol=FOOT_POINT_TOL), (
            f"{where} foot_point[1] != y2: {foot[1]} vs {y2}"
        )

        hints = rec["hints"]
        assert isinstance(hints, dict) and set(hints.keys()) == EXPECTED_HINT_KEYS, (
            f"{where} hints anahtarlari kontrata uymuyor: {hints!r}"
        )
        color = hints["dominant_color"]
        assert (
            isinstance(color, list) and len(color) == 3
            and all(_is_int(v) and 0 <= v <= 255 for v in color)
        ), f"{where} dominant_color 3 adet 0-255 int degil: {color!r}"
        assert hints["size_class"] in ALLOWED_SIZE_CLASSES, (
            f"{where} size_class gecersiz: {hints['size_class']!r}"
        )
        assert isinstance(hints["aspect_ratio"], float) and hints["aspect_ratio"] > 0, (
            f"{where} aspect_ratio pozitif float degil: {hints['aspect_ratio']!r}"
        )


@pytest.mark.parametrize("path", _param_files())
def test_frames_non_decreasing(path: Path):
    prev_frame = None
    for line_no, rec in _read_records(path):
        frame = rec["frame"]
        if prev_frame is not None:
            assert frame >= prev_frame, (
                f"{path.name}:{line_no} frame sirasi bozuk: {frame} < {prev_frame}"
            )
        prev_frame = frame

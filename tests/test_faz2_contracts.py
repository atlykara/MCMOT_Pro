"""Faz 2 cikti kontrat testleri: zone_events / zone_tracks / zone_tracks_mapped.

Amac: bolge hattinin urettigi CSV'lerin kolon seti, deger kumeleri ve ic
tutarliliginin sabit kalmasini garanti etmek. Beklenen kolon listeleri bu
dosyanin basinda sabittir; kontrat degisirse test KIRILIR, istenen davranis
budur. Testler hicbir cikti dosyasi uretmez veya degistirmez.
"""

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import load_zone_config, load_zone_params  # noqa: E402

ZONES_OUT_DIR = PROJECT_ROOT / "outputs" / "zones"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONES_MODULE = PROJECT_ROOT / "src" / "mcmot" / "zones.py"

# --- kontrat: kolon adlari ve SIRASI ---
EVENT_COLUMNS = [
    "camera_id", "zone_id", "track_id", "event", "frame", "timestamp",
    "foot_x", "foot_y", "lane", "class", "conf", "exit_reason",
]
TRACK_COLUMNS = [
    "camera_id", "zone_id", "track_id", "visit_index",
    "t_enter", "t_exit", "frame_enter", "frame_exit",
    "dwell_s", "n_frames_inside",
    "start_foot_x", "start_foot_y", "end_foot_x", "end_foot_y",
    "dx", "dy", "distance_px", "speed_px_s",
    "direction_label", "lane", "class_mode", "conf_mean", "exit_reason",
]
MAPPED_EXTRA_COLUMNS = ["movement_label", "mapping_rule"]
MAPPED_COLUMNS = TRACK_COLUMNS + MAPPED_EXTRA_COLUMNS

ALLOWED_EVENTS = {"enter", "exit"}
ALLOWED_EXIT_REASONS = {"left_zone", "track_end", "video_end"}
ALLOWED_LANES = {"upper", "lower", "unknown"}
ALLOWED_CLASSES = {"car", "bus", "truck"}
ALLOWED_DIRECTIONS = {"left", "right", "up", "down", "stationary"}
ALLOWED_MOVEMENTS = {"camA_to_camB", "camB_to_camA", "other"}

# Faz 2 geometridir; ogrenilmis model iceremez.
FORBIDDEN_IMPORTS = ("torch", "torchvision", "ultralytics", "onnxruntime")

DISTANCE_TOL = 1e-2

# Faz 3 gecis sarti (brief): her arac icin bulunmasi gereken alanlarin
# Faz 2 ciktisindaki karsiliklari.
FAZ3_REQUIRED_FIELDS = {
    "timestamp": ["t_enter", "t_exit"],
    "camera_id": ["camera_id"],
    "track_id": ["track_id"],
    "zone_id": ["zone_id"],
    "zemin/ayak noktasi": ["start_foot_x", "start_foot_y", "end_foot_x", "end_foot_y"],
}


def _param_files(pattern: str):
    files = sorted(ZONES_OUT_DIR.glob(pattern))
    if not files:
        reason = f"{ZONES_OUT_DIR} altinda {pattern} yok"
        return [pytest.param(None, id=f"{pattern}-yok",
                             marks=pytest.mark.skip(reason=reason))]
    return [pytest.param(f, id=f.name) for f in files]


def _read_csv(path: Path):
    """CSV'yi (satirlar, kolon adlari) olarak dondurur."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    assert rows, f"{path.name} bos, en az bir satir bekleniyordu"
    return rows, fieldnames


def _camera_of(path: Path) -> str:
    return path.stem.split("_")[-1]


def _zone_ids_of(camera_id: str):
    """configs/zones_<cam>.yaml icindeki zone_id kumesini dondurur; yoksa None."""
    config_path = CONFIG_DIR / f"zones_{camera_id}.yaml"
    if not config_path.is_file():
        return None
    return {zone.zone_id for zone in load_zone_config(config_path).zones}


def _min_dwell_s():
    """zone_params.yaml icindeki min_dwell_s degerini dondurur; yoksa None."""
    params_path = CONFIG_DIR / "zone_params.yaml"
    if not params_path.is_file():
        return None
    return float(load_zone_params(params_path)["min_dwell_s"])


# --------------------------------------------------------------------------
# zone_events_<cam>.csv
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", _param_files("zone_events_*.csv"))
def test_zone_events_kolonlari(path: Path):
    _, fieldnames = _read_csv(path)
    assert fieldnames == EVENT_COLUMNS, (
        f"{path.name} kolon seti/sirasi kontrata uymuyor:\n"
        f"  beklenen: {EVENT_COLUMNS}\n  gelen    : {fieldnames}"
    )


@pytest.mark.parametrize("path", _param_files("zone_events_*.csv"))
def test_zone_events_deger_kumeleri(path: Path):
    rows, _ = _read_csv(path)
    for line_no, row in enumerate(rows, start=2):
        where = f"{path.name}:{line_no}"
        assert row["event"] in ALLOWED_EVENTS, f"{where} event gecersiz: {row['event']!r}"
        assert row["lane"] in ALLOWED_LANES, f"{where} lane gecersiz: {row['lane']!r}"
        assert row["class"] in ALLOWED_CLASSES, f"{where} class gecersiz: {row['class']!r}"

        if row["event"] == "enter":
            assert row["exit_reason"] == "", (
                f"{where} enter satirinda exit_reason bos olmali: {row['exit_reason']!r}"
            )
        else:
            assert row["exit_reason"] in ALLOWED_EXIT_REASONS, (
                f"{where} exit_reason gecersiz: {row['exit_reason']!r}"
            )


@pytest.mark.parametrize("path", _param_files("zone_events_*.csv"))
def test_zone_events_track_icinde_monoton(path: Path):
    """Ayni track_id icinde frame ve timestamp azalmamali."""
    rows, _ = _read_csv(path)
    onceki = {}
    for line_no, row in enumerate(rows, start=2):
        track_id = int(row["track_id"])
        frame = int(row["frame"])
        timestamp = float(row["timestamp"])
        if track_id in onceki:
            prev_frame, prev_ts = onceki[track_id]
            assert frame >= prev_frame, (
                f"{path.name}:{line_no} track {track_id} frame geriye gitti: "
                f"{frame} < {prev_frame}"
            )
            assert timestamp >= prev_ts, (
                f"{path.name}:{line_no} track {track_id} timestamp geriye gitti: "
                f"{timestamp} < {prev_ts}"
            )
        onceki[track_id] = (frame, timestamp)


@pytest.mark.parametrize("path", _param_files("zone_events_*.csv"))
def test_zone_events_enter_exit_esitligi(path: Path):
    """Her (track_id, zone_id) icin enter sayisi exit sayisina esit olmali."""
    rows, _ = _read_csv(path)
    sayaclar: dict[tuple[str, str], Counter] = {}
    for row in rows:
        key = (row["track_id"], row["zone_id"])
        sayaclar.setdefault(key, Counter())[row["event"]] += 1

    hatali = {key: dict(counter) for key, counter in sayaclar.items()
              if counter["enter"] != counter["exit"]}
    assert not hatali, f"{path.name} enter/exit sayilari esit degil: {hatali}"


@pytest.mark.parametrize("path", _param_files("zone_events_*.csv"))
def test_zone_events_zone_id_tanimli(path: Path):
    camera_id = _camera_of(path)
    tanimli = _zone_ids_of(camera_id)
    if tanimli is None:
        pytest.skip(f"configs/zones_{camera_id}.yaml yok")
    rows, _ = _read_csv(path)
    gecersiz = {row["zone_id"] for row in rows} - tanimli
    assert not gecersiz, (
        f"{path.name} icinde zones_{camera_id}.yaml'da tanimli olmayan zone_id: {gecersiz}"
    )


# --------------------------------------------------------------------------
# zone_tracks_<cam>.csv
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", _param_files("zone_tracks_[!m]*.csv"))
def test_zone_tracks_kolonlari(path: Path):
    _, fieldnames = _read_csv(path)
    assert fieldnames == TRACK_COLUMNS, (
        f"{path.name} kolon seti/sirasi kontrata uymuyor:\n"
        f"  beklenen: {TRACK_COLUMNS}\n  gelen    : {fieldnames}"
    )


@pytest.mark.parametrize("path", _param_files("zone_tracks_[!m]*.csv"))
def test_zone_tracks_ic_tutarlilik(path: Path):
    rows, _ = _read_csv(path)
    min_dwell = _min_dwell_s()

    for line_no, row in enumerate(rows, start=2):
        where = f"{path.name}:{line_no}"

        t_enter, t_exit = float(row["t_enter"]), float(row["t_exit"])
        assert t_exit >= t_enter, f"{where} t_exit < t_enter: {t_exit} < {t_enter}"
        assert int(row["frame_exit"]) >= int(row["frame_enter"]), (
            f"{where} frame_exit < frame_enter"
        )

        if min_dwell is not None:
            assert float(row["dwell_s"]) >= min_dwell, (
                f"{where} dwell_s ({row['dwell_s']}) min_dwell_s ({min_dwell}) altinda"
            )

        dx, dy = float(row["dx"]), float(row["dy"])
        beklenen = math.hypot(dx, dy)
        assert math.isclose(float(row["distance_px"]), beklenen, abs_tol=DISTANCE_TOL), (
            f"{where} distance_px hypot(dx,dy) ile uyusmuyor: "
            f"{row['distance_px']} vs {beklenen:.3f}"
        )

        assert row["direction_label"] in ALLOWED_DIRECTIONS, (
            f"{where} direction_label gecersiz: {row['direction_label']!r}"
        )
        assert row["lane"] in ALLOWED_LANES, f"{where} lane gecersiz: {row['lane']!r}"
        assert row["class_mode"] in ALLOWED_CLASSES, (
            f"{where} class_mode gecersiz: {row['class_mode']!r}"
        )
        assert int(row["n_frames_inside"]) >= 1, (
            f"{where} n_frames_inside en az 1 olmali: {row['n_frames_inside']}"
        )


@pytest.mark.parametrize("path", _param_files("zone_tracks_[!m]*.csv"))
def test_zone_tracks_ziyaret_anahtari_benzersiz(path: Path):
    rows, _ = _read_csv(path)
    anahtarlar = [(row["track_id"], row["zone_id"], row["visit_index"]) for row in rows]
    tekrar = [key for key, count in Counter(anahtarlar).items() if count > 1]
    assert not tekrar, f"{path.name} (track_id, zone_id, visit_index) tekrar ediyor: {tekrar}"


# --------------------------------------------------------------------------
# zone_tracks_mapped_<cam>.csv
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", _param_files("zone_tracks_mapped_*.csv"))
def test_zone_tracks_mapped_kolonlari(path: Path):
    _, fieldnames = _read_csv(path)
    eksik = [column for column in TRACK_COLUMNS if column not in fieldnames]
    assert not eksik, f"{path.name} icinde zone_tracks kolonlari kaybolmus: {eksik}"
    assert fieldnames == MAPPED_COLUMNS, (
        f"{path.name} kolon seti/sirasi kontrata uymuyor:\n"
        f"  beklenen: {MAPPED_COLUMNS}\n  gelen    : {fieldnames}"
    )


@pytest.mark.parametrize("path", _param_files("zone_tracks_mapped_*.csv"))
def test_zone_tracks_mapped_degerleri(path: Path):
    rows, _ = _read_csv(path)
    for line_no, row in enumerate(rows, start=2):
        where = f"{path.name}:{line_no}"
        assert row["movement_label"] in ALLOWED_MOVEMENTS, (
            f"{where} movement_label gecersiz: {row['movement_label']!r}"
        )
        assert "TODO" not in row["movement_label"], (
            f"{where} doldurulmamis TODO degeri ciktiya sizmis"
        )
        assert row["mapping_rule"], f"{where} mapping_rule bos"


@pytest.mark.parametrize("path", _param_files("zone_tracks_mapped_*.csv"))
def test_zone_tracks_mapped_satir_sayisi_korunur(path: Path):
    camera_id = _camera_of(path)
    kaynak = ZONES_OUT_DIR / f"zone_tracks_{camera_id}.csv"
    if not kaynak.is_file():
        pytest.skip(f"{kaynak.name} yok")
    mapped_rows, _ = _read_csv(path)
    source_rows, _ = _read_csv(kaynak)
    assert len(mapped_rows) == len(source_rows), (
        f"{path.name} satir sayisi kaynaktan farkli: "
        f"{len(mapped_rows)} vs {len(source_rows)}"
    )


# --------------------------------------------------------------------------
# yasak kontrolu ve Faz 3 gecis sarti
# --------------------------------------------------------------------------

def test_reid_yasagi():
    """Faz 2 geometridir: zones modulu ogrenilmis model kutuphanesi icermemeli."""
    assert ZONES_MODULE.is_file(), f"{ZONES_MODULE} bulunamadi"
    metin = ZONES_MODULE.read_text(encoding="utf-8").lower()
    bulunan = [ad for ad in FORBIDDEN_IMPORTS if ad in metin]
    assert not bulunan, (
        f"{ZONES_MODULE.name} icinde yasak kutuphane gecti: {bulunan}. "
        "Faz 2'de ogrenilmis Re-ID/model kullanimi yasaktir."
    )


@pytest.mark.parametrize("path", _param_files("zone_tracks_mapped_*.csv"))
def test_faz3_gecis_sarti(path: Path):
    """Brief'in Faz 3 icin saydigi alanlarin karsiliklari eksiksiz ve dolu olmali."""
    rows, fieldnames = _read_csv(path)

    for brief_alan, kolonlar in FAZ3_REQUIRED_FIELDS.items():
        eksik = [column for column in kolonlar if column not in fieldnames]
        assert not eksik, (
            f"{path.name}: brief alani '{brief_alan}' icin kolon eksik: {eksik}"
        )

    gerekli = [column for kolonlar in FAZ3_REQUIRED_FIELDS.values() for column in kolonlar]
    for line_no, row in enumerate(rows, start=2):
        for column in gerekli:
            deger = row[column]
            assert deger not in (None, ""), (
                f"{path.name}:{line_no} '{column}' bos"
            )
            if column not in ("camera_id", "zone_id"):
                sayi = float(deger)
                assert not math.isnan(sayi), f"{path.name}:{line_no} '{column}' NaN"
                assert not math.isinf(sayi), f"{path.name}:{line_no} '{column}' sonsuz"

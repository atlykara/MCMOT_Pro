"""src/mcmot/zones.py birim testleri.

Amac: bolge geometrisi, serit isareti ve yon etiketi mantigini veri dosyasina
ihtiyac duymadan sabitlemek. Bu testler hicbir cikti dosyasi uretmez veya
degistirmez; gecici YAML'lar pytest'in tmp_path'ine yazilir.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import (  # noqa: E402
    ZoneDef,
    assign_lane,
    image_direction,
    load_zone_config,
    load_zone_params,
    point_in_zone,
)

# Birim testlerin refereans karesi: basit 100x100 kare poligon.
SQUARE = ZoneDef(
    zone_id="SQUARE",
    zone_type="transition",
    polygon=((0, 0), (100, 0), (100, 100), (0, 100)),
    lane_divider=None,
    notes="",
)

VALID_YAML = """camera_id: camTest
frame_size: [200, 200]
reference_frame: "data/reference_frames/ref_camTest.jpg"
zones:
  - zone_id: "Z-1"
    type: "transition"
    polygon:
      - [10, 10]
      - [100, 10]
      - [100, 100]
      - [10, 100]
    lane_divider:
      - [10, 50]
      - [100, 50]
    notes: "birim test"
"""

# (id, YAML metni, hata mesajinda gecmesi beklenen parca)
INVALID_CONFIGS = [
    (
        "iki-noktali-poligon",
        """camera_id: camTest
frame_size: [200, 200]
zones:
  - zone_id: "Z-1"
    polygon:
      - [10, 10]
      - [100, 100]
""",
        "en az 3 nokta",
    ),
    (
        "kendisiyle-kesisen-poligon",
        """camera_id: camTest
frame_size: [200, 200]
zones:
  - zone_id: "Z-1"
    polygon:
      - [0, 0]
      - [100, 100]
      - [100, 0]
      - [0, 100]
""",
        "kendisiyle kesisiyor",
    ),
    (
        "frame-size-disina-tasan-nokta",
        """camera_id: camTest
frame_size: [100, 100]
zones:
  - zone_id: "Z-1"
    polygon:
      - [10, 10]
      - [150, 10]
      - [80, 90]
""",
        "kare sinirlari",
    ),
    (
        "tekrar-eden-zone-id",
        """camera_id: camTest
frame_size: [200, 200]
zones:
  - zone_id: "Z-1"
    polygon:
      - [10, 10]
      - [100, 10]
      - [100, 100]
  - zone_id: "Z-1"
    polygon:
      - [110, 110]
      - [180, 110]
      - [180, 180]
""",
        "zone_id tekrar ediyor",
    ),
]

VALID_PARAMS = """min_conf: 0.4
allowed_classes: [car, bus, truck]
min_inside_frames: 3
min_outside_frames: 5
min_dwell_s: 0.5
direction_min_distance_px: 40
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# point_in_zone
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "point, expected, aciklama",
    [
        ((50, 50), True, "ic bolge"),
        ((150, 50), False, "poligon disi"),
        ((100, 50), True, "tam kenar uzeri - sinir ICERIDE sayilir"),
        ((0, 0), True, "tam kose - sinir ICERIDE sayilir"),
    ],
    ids=["ic-nokta", "dis-nokta", "kenar-uzeri", "kose"],
)
def test_point_in_zone(point, expected, aciklama):
    assert point_in_zone(point, SQUARE) is expected, (
        f"{aciklama}: {point} icin {expected} bekleniyordu"
    )


# --------------------------------------------------------------------------
# assign_lane
# --------------------------------------------------------------------------

def test_assign_lane_divider_yoksa_unknown():
    assert assign_lane((50, 50), SQUARE) == "unknown"


@pytest.mark.parametrize(
    "point, expected",
    [((50, 20), "upper"), ((50, 80), "lower")],
    ids=["cizginin-ustu", "cizginin-alti"],
)
def test_assign_lane_yatay_cizgi(point, expected):
    """Soldan saga cizilmis y=50 cizgisinde ust/alt etiketleri."""
    zone = ZoneDef("Z", "transition", SQUARE.polygon, ((0, 50), (100, 50)), "")
    assert assign_lane(point, zone) == expected


def test_assign_lane_cizim_yonu_etiketi_tersine_cevirir():
    """lane_divider'in iki noktasi ters sirada verilirse etiketler yer degistirir.

    Bu bilincli bir davranistir: capraz carpimin isareti cizim yonune baglidir.
    Test bu sozlesmeyi belgeliyor; ROI cizerken cizgi yonu onemlidir.
    """
    duz = ZoneDef("Z", "transition", SQUARE.polygon, ((0, 50), (100, 50)), "")
    ters = ZoneDef("Z", "transition", SQUARE.polygon, ((100, 50), (0, 50)), "")

    for point in ((50, 20), (50, 80)):
        assert assign_lane(point, duz) != assign_lane(point, ters), (
            f"{point}: cizim yonu ters cevrilince etiket de terslenmeli"
        )
    assert assign_lane((50, 20), ters) == "lower"
    assert assign_lane((50, 80), ters) == "upper"


# --------------------------------------------------------------------------
# image_direction
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dx, dy, min_distance, expected",
    [
        (100, 5, 40, "right"),
        (-100, 5, 40, "left"),
        (5, 100, 40, "down"),     # goruntu koordinatlarinda y asagi buyur
        (5, -100, 40, "up"),
        (10, 10, 40, "stationary"),
        (100, 100, 40, "right"),  # abs(dx) >= abs(dy) -> yatay eksen kazanir
    ],
    ids=["saga", "sola", "asagi", "yukari", "duragan", "esit-buyukluk"],
)
def test_image_direction(dx, dy, min_distance, expected):
    assert image_direction(dx, dy, min_distance) == expected


def test_image_direction_esik_hypot_ile_hesaplanir():
    """dx=30, dy=30 vakasi: esik karsilastirmasi hypot ile yapilir.

    hypot(30, 30) = 42.43 olup 40 esiginin USTUNDEDIR; bu nedenle sonuc
    "stationary" DEGIL, baskin eksene gore "right" olmalidir. Ezbere
    beklenti yazilmasin diye esik burada acikca hesaplanir.
    """
    mesafe = math.hypot(30, 30)
    assert mesafe > 40, f"on kabul hatali: hypot(30,30)={mesafe:.2f}"
    assert image_direction(30, 30, 40) == "right"

    # esigin gercekten altinda kalan bir vaka duragan olmali
    assert math.hypot(10, 10) < 40
    assert image_direction(10, 10, 40) == "stationary"


# --------------------------------------------------------------------------
# load_zone_config
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_text, beklenen_parca",
    [pytest.param(text, parca, id=test_id) for test_id, text, parca in INVALID_CONFIGS],
)
def test_load_zone_config_gecersiz_yapilandirmayi_reddeder(tmp_path, yaml_text, beklenen_parca):
    path = _write(tmp_path, "zones_camTest.yaml", yaml_text)
    with pytest.raises(ValueError) as hata:
        load_zone_config(path)
    assert beklenen_parca in str(hata.value), (
        f"hata mesajinda '{beklenen_parca}' bekleniyordu, gelen: {hata.value}"
    )


def test_load_zone_config_gecerli_yapilandirmayi_okur(tmp_path):
    config = load_zone_config(_write(tmp_path, "zones_camTest.yaml", VALID_YAML))

    assert config.camera_id == "camTest"
    assert config.frame_size == (200, 200)
    assert config.reference_frame == "data/reference_frames/ref_camTest.jpg"
    assert len(config.zones) == 1

    zone = config.zones[0]
    assert zone.zone_id == "Z-1"
    assert zone.zone_type == "transition"
    assert zone.polygon == ((10, 10), (100, 10), (100, 100), (10, 100))
    assert zone.lane_divider == ((10, 50), (100, 50))
    assert zone.notes == "birim test"


def test_load_zone_config_olmayan_dosya(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_zone_config(tmp_path / "yok.yaml")


# --------------------------------------------------------------------------
# load_zone_params
# --------------------------------------------------------------------------

def test_load_zone_params_gecerli_dosyayi_okur(tmp_path):
    params = load_zone_params(_write(tmp_path, "zone_params.yaml", VALID_PARAMS))
    assert params["min_conf"] == 0.4
    assert params["allowed_classes"] == ["car", "bus", "truck"]
    assert params["min_inside_frames"] == 3
    assert params["direction_min_distance_px"] == 40


@pytest.mark.parametrize(
    "eksik_anahtar",
    ["min_conf", "allowed_classes", "min_inside_frames",
     "min_outside_frames", "min_dwell_s", "direction_min_distance_px"],
)
def test_load_zone_params_eksik_anahtari_reddeder(tmp_path, eksik_anahtar):
    """Eksik esik icin sessizce varsayilan URETILMEZ; net hata verilir."""
    satirlar = [line for line in VALID_PARAMS.splitlines()
                if not line.startswith(f"{eksik_anahtar}:")]
    path = _write(tmp_path, "zone_params.yaml", "\n".join(satirlar) + "\n")

    with pytest.raises((KeyError, ValueError)) as hata:
        load_zone_params(path)
    assert eksik_anahtar in str(hata.value), (
        f"hata mesaji eksik anahtari ({eksik_anahtar}) belirtmeli, gelen: {hata.value}"
    )


def test_load_zone_params_olmayan_dosya(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_zone_params(tmp_path / "yok.yaml")

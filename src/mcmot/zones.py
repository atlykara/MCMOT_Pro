"""Tek kamera icin bolge (ROI) tanimlarini okuma ve nokta/bolge hesaplari.

Bu bir kutuphane modulu olup dogrudan calistirilmaz. Modul yalnizca okur ve
hesaplar; hicbir dosyaya yazmaz. Kapsam tek kameradir: kameralar arasi hicbir
mantik icermez.

Geometri shapely ile yapilir; esik degerleri koda gomulmez, cagiran taraf
configs/zone_params.yaml uzerinden parametre olarak gecirir.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import Point, Polygon

# configs/zone_params.yaml icinde bulunmasi ZORUNLU anahtarlar.
# Eksik olan icin sessizce varsayilan uretilmez; hata verilir.
REQUIRED_PARAM_KEYS: tuple[str, ...] = (
    "min_conf",
    "allowed_classes",
    "min_inside_frames",
    "min_outside_frames",
    "min_dwell_s",
    "direction_min_distance_px",
)

MIN_POLYGON_POINTS = 3
DEFAULT_ZONE_TYPE = "transition"

LANE_UPPER = "upper"
LANE_LOWER = "lower"
LANE_UNKNOWN = "unknown"

DIRECTION_STATIONARY = "stationary"
DIRECTION_RIGHT = "right"
DIRECTION_LEFT = "left"
DIRECTION_UP = "up"
DIRECTION_DOWN = "down"


@dataclass(frozen=True)
class ZoneDef:
    """Tek bir bolge tanimi.

    zone_id      : bolgenin benzersiz kimligi (kamera dosyasi icinde tekil)
    zone_type    : bolgenin rolu, ornegin "transition"
    polygon      : ((x, y), ...) seklinde int piksel kose noktalari
    lane_divider : ((x1, y1), (x2, y2)) serit ayirici cizgi veya None
    notes        : serbest not metni
    """

    zone_id: str
    zone_type: str
    polygon: tuple[tuple[int, int], ...]
    lane_divider: tuple[tuple[int, int], tuple[int, int]] | None
    notes: str


@dataclass(frozen=True)
class ZoneConfig:
    """Tek kameraya ait bolge yapilandirmasi.

    camera_id       : kamera kimligi (or. camA)
    frame_size      : (genislik, yukseklik) piksel
    reference_frame : bolgelerin cizildigi arka plan goruntusunun yolu
    zones           : (ZoneDef, ...) dosyadaki siralamayi korur
    """

    camera_id: str
    frame_size: tuple[int, int]
    reference_frame: str
    zones: tuple[ZoneDef, ...]


# --------------------------------------------------------------------------
# shapely onbellegi
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _polygon_of(zone: ZoneDef) -> Polygon:
    """ZoneDef icin shapely Polygon uretir ve onbellege alir.

    ZoneDef frozen ve tuple alanli oldugu icin hashlenebilir; onbellek anahtari
    zone_id degil ZoneDef'in kendisidir. Boylece ayni zone_id farkli poligonla
    yeniden yuklendiginde bayat geometri donmez.

    zone   : ZoneDef
    donus  : shapely Polygon
    """
    return Polygon(zone.polygon)


# --------------------------------------------------------------------------
# yukleme
# --------------------------------------------------------------------------

def _as_point(value: Any, context: str, errors: list[str]) -> tuple[int, int] | None:
    """YAML'dan gelen bir ogeyi (x, y) int ciftine cevirir; hatada None."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        errors.append(f"{context}: nokta [x, y] bicinde olmali, gelen: {value!r}")
        return None
    try:
        return (int(round(float(value[0]))), int(round(float(value[1]))))
    except (TypeError, ValueError):
        errors.append(f"{context}: nokta koordinatlari sayisal olmali, gelen: {value!r}")
        return None


def load_zone_config(path: Path) -> ZoneConfig:
    """configs/zones_<cam>.yaml dosyasini okur, dogrular ve ZoneConfig dondurur.

    Yapilan dogrulamalar:
      - camera_id, frame_size ve zones alanlari var mi
      - her poligon en az 3 nokta iceriyor mu
      - shapely Polygon gecerli mi (is_valid) yani kendisiyle kesismiyor mu
      - tum noktalar frame_size sinirlari icinde mi (0 <= x < genislik)
      - zone_id degerleri benzersiz mi
      - lane_divider varsa tam 2 nokta mi ve iki nokta birbirinden farkli mi
    Bulunan tum hatalar toplanip tek bir ValueError icinde bildirilir.
    'type' alani yoksa "transition" kabul edilir; 'reference_frame' opsiyoneldir.

    path   : zones_<cam>.yaml dosya yolu
    donus  : ZoneConfig
    hata   : FileNotFoundError (dosya yok), ValueError (icerik gecersiz)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bolge yapilandirmasi bulunamadi: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML cozumlenemedi: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: dosyanin en ust duzeyi sozluk olmali.")

    errors: list[str] = []

    camera_id = data.get("camera_id")
    if not isinstance(camera_id, str) or not camera_id.strip():
        errors.append("camera_id alani eksik veya bos.")
        camera_id = ""

    raw_size = data.get("frame_size")
    frame_size = (0, 0)
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        errors.append("frame_size alani eksik; [genislik, yukseklik] olmali.")
    else:
        try:
            frame_size = (int(raw_size[0]), int(raw_size[1]))
        except (TypeError, ValueError):
            errors.append(f"frame_size sayisal olmali, gelen: {raw_size!r}")
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            errors.append(f"frame_size pozitif olmali, gelen: {raw_size!r}")

    reference_frame = data.get("reference_frame") or ""
    if not isinstance(reference_frame, str):
        errors.append("reference_frame metin olmali.")
        reference_frame = ""

    raw_zones = data.get("zones")
    if not isinstance(raw_zones, list) or not raw_zones:
        errors.append("zones alani eksik veya bos liste.")
        raw_zones = []

    zones: list[ZoneDef] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_zones, start=1):
        if not isinstance(item, dict):
            errors.append(f"zones[{index}]: sozluk olmali.")
            continue

        zone_id = item.get("zone_id")
        label = zone_id if isinstance(zone_id, str) and zone_id.strip() else f"zones[{index}]"
        if not isinstance(zone_id, str) or not zone_id.strip():
            errors.append(f"{label}: zone_id eksik veya bos.")
            zone_id = ""
        elif zone_id in seen_ids:
            errors.append(f"{label}: zone_id tekrar ediyor.")
        else:
            seen_ids.add(zone_id)

        zone_type = item.get("type") or DEFAULT_ZONE_TYPE
        if not isinstance(zone_type, str):
            errors.append(f"{label}: type metin olmali.")
            zone_type = DEFAULT_ZONE_TYPE

        raw_polygon = item.get("polygon")
        points: list[tuple[int, int]] = []
        if not isinstance(raw_polygon, list):
            errors.append(f"{label}: polygon listesi eksik.")
        else:
            for point_index, raw_point in enumerate(raw_polygon, start=1):
                point = _as_point(raw_point, f"{label}: polygon[{point_index}]", errors)
                if point is not None:
                    points.append(point)

        if len(points) < MIN_POLYGON_POINTS:
            errors.append(f"{label}: poligon en az {MIN_POLYGON_POINTS} nokta icermeli "
                          f"(su an {len(points)}).")
        else:
            width, height = frame_size
            if width > 0 and height > 0:
                outside = [p for p in points
                           if not (0 <= p[0] < width and 0 <= p[1] < height)]
                if outside:
                    errors.append(f"{label}: {len(outside)} poligon noktasi kare sinirlari "
                                  f"disinda ({width}x{height}): {outside[:3]}")
            try:
                if not Polygon(points).is_valid:
                    errors.append(f"{label}: poligon gecersiz (kendisiyle kesisiyor).")
            except Exception as exc:  # shapely kurulum/geometri hatasi
                errors.append(f"{label}: poligon shapely ile dogrulanamadi: {exc}")

        raw_divider = item.get("lane_divider")
        divider: tuple[tuple[int, int], tuple[int, int]] | None = None
        if raw_divider is not None:
            if not isinstance(raw_divider, list) or len(raw_divider) != 2:
                errors.append(f"{label}: lane_divider tam 2 nokta icermeli.")
            else:
                divider_points = []
                for point_index, raw_point in enumerate(raw_divider, start=1):
                    point = _as_point(raw_point, f"{label}: lane_divider[{point_index}]", errors)
                    if point is not None:
                        divider_points.append(point)
                if len(divider_points) == 2:
                    if divider_points[0] == divider_points[1]:
                        errors.append(f"{label}: lane_divider'in iki noktasi ayni.")
                    else:
                        width, height = frame_size
                        if width > 0 and height > 0:
                            bad = [p for p in divider_points
                                   if not (0 <= p[0] < width and 0 <= p[1] < height)]
                            if bad:
                                errors.append(f"{label}: lane_divider noktasi kare disinda: {bad}")
                        divider = (divider_points[0], divider_points[1])

        zones.append(ZoneDef(
            zone_id=zone_id,
            zone_type=zone_type,
            polygon=tuple(points),
            lane_divider=divider,
            notes=str(item.get("notes") or ""),
        ))

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(f"{path}: bolge yapilandirmasi gecersiz:\n  - {detail}")

    return ZoneConfig(
        camera_id=camera_id,
        frame_size=frame_size,
        reference_frame=reference_frame,
        zones=tuple(zones),
    )


def load_zone_params(path: Path) -> dict[str, Any]:
    """configs/zone_params.yaml dosyasini okur ve sozluk olarak dondurur.

    REQUIRED_PARAM_KEYS icindeki anahtarlardan biri bile eksikse hata verilir;
    eksik esik icin sessizce varsayilan uretilmez. Dosyadaki ek anahtarlar
    oldugu gibi dondurulur.

    path   : zone_params.yaml dosya yolu
    donus  : parametre sozlugu
    hata   : FileNotFoundError (dosya yok), ValueError (icerik gecersiz/eksik)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Bolge parametre dosyasi bulunamadi: {path}. "
            f"Beklenen anahtarlar: {', '.join(REQUIRED_PARAM_KEYS)}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML cozumlenemedi: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: dosyanin en ust duzeyi sozluk olmali.")

    missing = [key for key in REQUIRED_PARAM_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path}: zorunlu parametreler eksik: {', '.join(missing)}")

    empty = [key for key in REQUIRED_PARAM_KEYS if data[key] is None]
    if empty:
        raise ValueError(f"{path}: su parametreler bos birakilmis: {', '.join(empty)}")

    return dict(data)


# --------------------------------------------------------------------------
# geometri
# --------------------------------------------------------------------------

def point_in_zone(point: tuple[float, float], zone: ZoneDef) -> bool:
    """Noktanin bolge poligonu icinde olup olmadigini dondurur.

    Sinir uzerindeki nokta ICERIDE sayilir. Bu nedenle Polygon.contains yerine
    Polygon.covers kullanilir: contains sinir noktalarini disarida birakir,
    covers ise "ic bolge + sinir" anlamina gelir (intersects ise yalnizca kesisme
    sorar, poligon icin covers ile ayni sonucu verse de anlami daha genistir).

    point  : (x, y) piksel, tipik olarak foot_point
    zone   : ZoneDef
    donus  : bool
    """
    return _polygon_of(zone).covers(Point(float(point[0]), float(point[1])))


def assign_lane(point: tuple[float, float], zone: ZoneDef) -> str:
    """Noktanin lane_divider cizgisinin hangi tarafinda kaldigini dondurur.

    lane_divider tanimli degilse "unknown" doner. Tanimliysa isaretli capraz
    carpim kullanilir:
        d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    d < 0  -> "upper", d > 0 -> "lower", d == 0 (cizgi uzerinde) -> "upper".

    ONEMLI: goruntu koordinatlarinda y asagi dogru buyur. Cizgi soldan saga
    ciziliyse (x2 > x1) d < 0 olan taraf ekranda cizginin USTUNDE kalir, bu
    yuzden "upper" adlandirmasi sezgiseldir. Cizgi sagdan sola cizilmisse isaret
    ters doner ve "upper"/"lower" ekranda yer degistirir; etiketlerin anlami
    lane_divider'in CIZIM YONUNE baglidir.

    point  : (x, y) piksel
    zone   : ZoneDef
    donus  : "upper" | "lower" | "unknown"
    """
    divider = zone.lane_divider
    if divider is None:
        return LANE_UNKNOWN

    (x1, y1), (x2, y2) = divider
    px, py = float(point[0]), float(point[1])
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return LANE_LOWER
    return LANE_UPPER


def find_zone(point: tuple[float, float], config: ZoneConfig) -> str | None:
    """Noktanin icinde bulundugu ilk bolgenin zone_id degerini dondurur.

    Bolgeler yapilandirma dosyasindaki sirayla taranir. Poligonlar cakisiyorsa
    ILK eslesen bolge kazanir; nokta birden fazla bolgede olsa bile tek bir
    zone_id doner. Hicbir bolgeye dusmuyorsa None doner.

    point  : (x, y) piksel, tipik olarak foot_point
    config : ZoneConfig
    donus  : zone_id (str) veya None
    """
    for zone in config.zones:
        if point_in_zone(point, zone):
            return zone.zone_id
    return None


def image_direction(dx: float, dy: float, min_distance_px: float) -> str:
    """Goruntu duzlemindeki yer degistirmeyi kaba yone cevirir (saf fonksiyon).

    Toplam yer degistirme min_distance_px altindaysa "stationary" doner. Aksi
    halde baskin eksene bakilir: abs(dx) >= abs(dy) ise yatay eksen secilir ve
    dx > 0 icin "right", dx < 0 icin "left"; dikey eksende dy > 0 icin "down",
    dy < 0 icin "up" doner (goruntu koordinatlarinda y asagi dogru buyur).

    dx              : x eksenindeki yer degistirme (piksel)
    dy              : y eksenindeki yer degistirme (piksel)
    min_distance_px : bu esigin altindaki hareket duragan sayilir
    donus           : "stationary" | "right" | "left" | "up" | "down"
    """
    if math.hypot(dx, dy) < min_distance_px:
        return DIRECTION_STATIONARY

    if abs(dx) >= abs(dy):
        if dx > 0:
            return DIRECTION_RIGHT
        if dx < 0:
            return DIRECTION_LEFT
        return DIRECTION_STATIONARY  # dx == dy == 0 ve esik 0 verilmis
    return DIRECTION_DOWN if dy > 0 else DIRECTION_UP

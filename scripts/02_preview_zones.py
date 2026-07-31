"""Faz 2 gorsel kontrol: ROI poligonlarini ve serit ayrimini referans kareye bindirir.

Script SALT OKUR: zones_<cam>.yaml, referans kare ve (istege bagli)
zone_tracks_<cam>.csv okunur; hicbir yapilandirma dosyasi degistirilmez ve
poligonlar otomatik duzeltilmez.

upper/lower etiketleri elle yazilmaz; her iki taraf icin bir nokta secilip
mcmot.zones.assign_lane cagrilir ve DONEN deger yazilir. Boylece isaret yonu
hatasi goruntude dogrudan gorunur.

Ornek:
    python scripts/02_preview_zones.py --camera camA
    python scripts/02_preview_zones.py --camera all --with-tracks
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import LineString

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import (  # noqa: E402
    assign_lane,
    load_zone_config,
    point_in_zone,
)

DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
PREVIEW_DIR = ZONES_DIR / "previews"

FILL_ALPHA = 0.25            # poligon dolgusunun saydamligi
BORDER_THICKNESS = 3
DASH_LENGTH_PX = 18          # kesikli cizgi parca uzunlugu
LANE_LABEL_MAX_OFFSET = 400  # etiket noktasi ararken cizgiden en fazla bu kadar uzaklas
LANE_LABEL_STEP = 10
JPEG_QUALITY = 95

# Zone dolgu renkleri (BGR); zone sayisi fazlaysa bastan tekrarlanir.
ZONE_COLORS = (
    (0, 200, 255),    # turuncu
    (0, 255, 128),    # yesil
    (255, 160, 0),    # mavi
    (200, 0, 255),    # mor
)
LANE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (25, 25, 25)

DIRECTION_COLORS = {
    "right": (0, 200, 0),
    "left": (0, 0, 230),
    "up": (255, 60, 0),
    "down": (0, 220, 255),
    "stationary": (150, 150, 150),
}


# --------------------------------------------------------------------------
# cizim yardimcilari
# --------------------------------------------------------------------------

def text_scale(image) -> float:
    """Goruntu boyutuna gore okunur bir yazi olcegi dondurur."""
    return max(0.6, min(1.4, image.shape[1] / 1800.0))


def draw_label(image, text, origin, scale=None, color=TEXT_COLOR, padding=6):
    """Metni koyu bir kutu uzerine yazar (arka plan uzerinde okunur olsun diye).

    origin metnin taban cizgisidir; donus kutunun alt kenaridir.
    """
    scale = scale if scale is not None else text_scale(image)
    thickness = max(1, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    top_left = (x - padding, y - height - padding)
    bottom_right = (x + width + padding, y + baseline + padding)
    cv2.rectangle(image, top_left, bottom_right, BOX_COLOR, -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)
    return bottom_right[1]


def draw_text_block(image, entries, origin, scale=None, swatches=False, padding=10):
    """Cok satirli metni TEK bir koyu kutu icinde yazar (satirlar cakismaz).

    entries : [(metin, renk), ...]
    origin  : kutunun sol ust kosesi
    swatches: True ise her satirin soluna renk orneği kutucugu cizilir
    donus   : (kutu genisligi, kutu yuksekligi)
    """
    scale = scale if scale is not None else text_scale(image)
    thickness = max(1, int(round(scale * 2)))
    sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
             for text, _ in entries]
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    line_height = text_height + baseline + 10
    swatch_width = int(30 * scale) if swatches else 0
    box_width = max(size[0][0] for size in sizes) + 2 * padding + swatch_width
    box_height = line_height * len(entries) + 2 * padding - 10

    x, y = int(origin[0]), int(origin[1])
    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), BOX_COLOR, -1)
    for index, (text, color) in enumerate(entries):
        text_y = y + padding + text_height + index * line_height
        text_x = x + padding + swatch_width
        if swatches and index > 0:
            cv2.rectangle(image, (x + padding, text_y - text_height),
                          (x + padding + swatch_width - 10, text_y), color, -1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
    return box_width, box_height


def draw_dashed_line(image, start, end, color, thickness=3, dash=DASH_LENGTH_PX):
    """Iki nokta arasina kesikli cizgi cizer."""
    x1, y1 = start
    x2, y2 = end
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1e-6:
        return
    steps = max(int(length // dash), 1)
    for index in range(steps + 1):
        if index % 2:
            continue
        t0 = min(index * dash / length, 1.0)
        t1 = min((index + 1) * dash / length, 1.0)
        p0 = (int(round(x1 + (x2 - x1) * t0)), int(round(y1 + (y2 - y1) * t0)))
        p1 = (int(round(x1 + (x2 - x1) * t1)), int(round(y1 + (y2 - y1) * t1)))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def polygon_label_point(zone):
    """Zone_id etiketi icin poligon agirlik merkezini dondurur.

    Agirlik merkezi ice bukey poligonlarda disarida kalabilir; bu durumda
    poligonun icinde kaldigi garanti olan bir temsil noktasi kullanilir.
    """
    from shapely.geometry import Polygon  # yerel import: modul sozlesmesini bozmamak icin

    polygon = Polygon(zone.polygon)
    centroid = polygon.centroid
    if not polygon.covers(centroid):
        centroid = polygon.representative_point()
    return (centroid.x, centroid.y)


def draw_zones(image, config):
    """Poligonlari yari saydam doldurur ve kenarliklarini cizer."""
    overlay = image.copy()
    for index, zone in enumerate(config.zones):
        color = ZONE_COLORS[index % len(ZONE_COLORS)]
        points = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, FILL_ALPHA, image, 1.0 - FILL_ALPHA, 0.0, dst=image)

    for index, zone in enumerate(config.zones):
        color = ZONE_COLORS[index % len(ZONE_COLORS)]
        points = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, color, BORDER_THICKNESS, cv2.LINE_AA)
        for point in zone.polygon:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1, cv2.LINE_AA)


def draw_zone_labels(image, config):
    """zone_id metnini poligonun agirlik merkezine yazar (ok katmaninin ustunde)."""
    for zone in config.zones:
        cx, cy = polygon_label_point(zone)
        draw_label(image, zone.zone_id, (cx - 40, cy), scale=text_scale(image) * 1.1)


def lane_label_point(zone, base_point, normal, sign):
    """Cizginin bir tarafinda, poligon icinde iyi konumlanmis etiket noktasi bulur.

    Dik yonde disari dogru taranir; poligon icinde kalinabilen EN UZAK mesafe
    bulunur ve etiket onun %60'ina konur. Boylece iki etiket cizgiye yapisip
    birbirinin uzerine binmez.

    base_point : kesikli cizginin orta noktasi
    normal     : cizgiye dik birim vektor
    sign       : +1 veya -1 (hangi taraf)
    donus      : (x, y) veya bulunamazsa None
    """
    def at(distance):
        return (base_point[0] + normal[0] * distance * sign,
                base_point[1] + normal[1] * distance * sign)

    farthest = None
    offset = LANE_LABEL_STEP
    while offset <= LANE_LABEL_MAX_OFFSET:
        if point_in_zone(at(offset), zone):
            farthest = offset
        elif farthest is not None:
            break  # poligondan cikildi, daha ilerisine bakmaya gerek yok
        offset += LANE_LABEL_STEP

    if farthest is None:
        return None
    return at(max(LANE_LABEL_STEP, farthest * 0.6))


def draw_lane_divider(image, zone) -> bool:
    """lane_divider'in poligon icinde kalan kismini kesikli cizer, taraflari etiketler.

    Etiket metni sabit degildir: secilen nokta assign_lane'e sorulur ve fonksiyonun
    DONDURDUGU deger yazilir; boylece isaret yonu hatasi gozle gorulur.
    """
    from shapely.geometry import Polygon  # yerel import

    if zone.lane_divider is None:
        return False

    polygon = Polygon(zone.polygon)
    clipped = LineString(zone.lane_divider).intersection(polygon)
    if clipped.is_empty:
        print(f"  UYARI: '{zone.zone_id}' lane_divider'i poligonun icinden gecmiyor; "
              "cizilmedi.")
        return False

    segments = []
    if clipped.geom_type == "LineString":
        segments = [list(clipped.coords)]
    elif clipped.geom_type == "MultiLineString":
        segments = [list(part.coords) for part in clipped.geoms]
    else:
        print(f"  UYARI: '{zone.zone_id}' lane_divider kesisimi cizgi degil "
              f"({clipped.geom_type}); cizilmedi.")
        return False

    longest = max(segments, key=lambda coords: LineString(coords).length)
    for coords in segments:
        for start, end in zip(coords, coords[1:]):
            draw_dashed_line(image, start, end, LANE_COLOR, thickness=3)

    (x1, y1), (x2, y2) = longest[0], longest[-1]
    mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1e-6:
        return True
    normal = (-(y2 - y1) / length, (x2 - x1) / length)

    for sign in (1, -1):
        point = lane_label_point(zone, mid, normal, sign)
        if point is None:
            print(f"  UYARI: '{zone.zone_id}' icin cizginin bir tarafinda etiket "
                  "noktasi bulunamadi.")
            continue
        label = assign_lane(point, zone)  # etiket assign_lane'in gercek ciktisi
        cv2.circle(image, (int(point[0]), int(point[1])), 6, LANE_COLOR, -1, cv2.LINE_AA)
        draw_label(image, label, (point[0] + 12, point[1] + 6), color=LANE_COLOR)
    return True


def draw_info_box(image, config, drawn_dividers):
    """Sol uste kamera bilgisi kutusu cizer."""
    entries = [
        (f"camera_id: {config.camera_id}", TEXT_COLOR),
        (f"zone sayisi: {len(config.zones)}", TEXT_COLOR),
        (f"frame_size: {config.frame_size[0]}x{config.frame_size[1]}", TEXT_COLOR),
        (f"lane_divider: {drawn_dividers} / {len(config.zones)}", TEXT_COLOR),
    ]
    draw_text_block(image, entries, (16, 16), scale=text_scale(image) * 0.8)


def draw_tracks(image, rows):
    """Ziyaret vektorlerini direction_label rengiyle ok olarak cizer."""
    used = {}
    for row in rows:
        try:
            start = (int(round(float(row["start_foot_x"]))),
                     int(round(float(row["start_foot_y"]))))
            end = (int(round(float(row["end_foot_x"]))),
                   int(round(float(row["end_foot_y"]))))
        except (KeyError, TypeError, ValueError):
            continue
        label = row.get("direction_label", "")
        color = DIRECTION_COLORS.get(label, (255, 255, 255))
        used[label] = used.get(label, 0) + 1
        if start == end:
            cv2.circle(image, start, 4, color, -1, cv2.LINE_AA)
            continue
        cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.12)
    return used


def draw_legend(image, used):
    """Sag uste direction_label renk aciklamasini yazar."""
    scale = text_scale(image) * 0.8
    entries = [("direction_label", TEXT_COLOR)]
    entries += [(f"{label} ({used[label]})", DIRECTION_COLORS[label])
                for label in DIRECTION_COLORS if label in used]
    if len(entries) == 1:
        return
    # once olcmek icin gorunmeyen bir kopyaya cizip genisligi ogren
    probe = image[:1, :1].copy()
    width, _ = draw_text_block(probe, entries, (0, 0), scale=scale, swatches=True)
    draw_text_block(image, entries, (image.shape[1] - width - 16, 16),
                    scale=scale, swatches=True)


# --------------------------------------------------------------------------

def read_zone_tracks(path: Path):
    """zone_tracks_<cam>.csv dosyasini okur; yoksa (None, mesaj) dondurur."""
    if not path.is_file():
        return None, f"UYARI: ziyaret ozeti bulunamadi, ok katmani cizilmedi: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, f"UYARI: ziyaret ozeti bos: {path}"
    return rows, None


def process_camera(camera_id, zones_path, zone_tracks_path, with_tracks) -> bool:
    """Tek kamera icin onizleme goruntusunu uretir."""
    print(f"\n=== {camera_id} ===")
    try:
        config = load_zone_config(zones_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}")
        return False

    reference = config.reference_frame
    if not reference:
        print(f"HATA: {zones_path} icinde reference_frame alani bos; "
              "onizleme icin arka plan goruntusu gerekli.")
        return False
    reference_path = Path(reference)
    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path
    if not reference_path.is_file():
        print(f"HATA: referans kare bulunamadi: {reference_path}\n"
              f"      Uretmek icin: python scripts/02_extract_reference_frames.py "
              f"--camera {camera_id}")
        return False

    image = cv2.imread(str(reference_path))
    if image is None:
        print(f"HATA: referans kare cv2 ile okunamadi: {reference_path}")
        return False

    height, width = image.shape[:2]
    if (width, height) != config.frame_size:
        print(f"UYARI: referans kare {width}x{height} ile zones dosyasindaki "
              f"frame_size {config.frame_size[0]}x{config.frame_size[1]} farkli; "
              "poligonlar kaymis gorunebilir.")

    print(f"Referans kare : {reference_path} ({width}x{height})")
    print(f"Bolgeler      : {[zone.zone_id for zone in config.zones]}")

    draw_zones(image, config)
    drawn_dividers = sum(1 for zone in config.zones if draw_lane_divider(image, zone))
    if drawn_dividers == 0:
        print("Not           : hicbir bolgede lane_divider tanimli degil; "
              "upper/lower etiketi cizilmedi.")
    else:
        print(f"lane_divider  : {drawn_dividers} bolgede cizildi "
              "(etiketler assign_lane ciktisidir)")

    used = {}
    if with_tracks:
        rows, message = read_zone_tracks(zone_tracks_path)
        if message:
            print(message)
        if rows:
            used = draw_tracks(image, rows)
            summary = ", ".join(f"{label}={count}" for label, count in sorted(used.items()))
            print(f"Ziyaret oku   : {len(rows)} ok cizildi ({summary})")

    # metin katmani en uste: oklar zone_id etiketini kapatmasin
    draw_zone_labels(image, config)
    draw_info_box(image, config, drawn_dividers)
    if used:
        draw_legend(image, used)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / f"zones_preview_{camera_id}.jpg"
    if not cv2.imwrite(str(out_path), image,
                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
        print(f"HATA: onizleme yazilamadi: {out_path}")
        return False
    print(f"Yazildi       : {out_path}")
    return True


def load_camera_ids(config_path: Path):
    """cameras.yaml icindeki camera_id listesini dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cameras = [c.get("camera_id") for c in (data.get("cameras") or [])
               if isinstance(c, dict) and c.get("camera_id")]
    if not cameras:
        raise SystemExit(f"HATA: cameras.yaml icinde kamera tanimi yok: {config_path}")
    return cameras


def resolve(path, default):
    """Verilen yolu PROJECT_ROOT'a gore cozer; yoksa varsayilani dondurur."""
    if path is None:
        return default
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 2: ROI poligonlarini referans kare uzerinde gorsel olarak kontrol et."
    )
    parser.add_argument("--camera", required=True,
                        help="kamera kimligi; tum kameralar icin 'all'")
    parser.add_argument("--with-tracks", action="store_true",
                        help="zone_tracks_<cam>.csv ziyaret vektorlerini ok olarak ekle")
    parser.add_argument("--zones", type=Path, default=None,
                        help="zones_<cam>.yaml yolu")
    parser.add_argument("--zone-tracks", type=Path, default=None,
                        help="zone_tracks_<cam>.csv yolu")
    args = parser.parse_args()

    if args.camera == "all":
        cameras = load_camera_ids(DEFAULT_CAMERAS_CONFIG)
        if args.zones is not None or args.zone_tracks is not None:
            raise SystemExit("HATA: --camera all ile dosya yollari birlikte kullanilamaz; "
                             "yollar kamera basina turetilir.")
    else:
        cameras = [args.camera]

    results = []
    for camera_id in cameras:
        results.append(process_camera(
            camera_id,
            resolve(args.zones, CONFIG_DIR / f"zones_{camera_id}.yaml"),
            resolve(args.zone_tracks, ZONES_DIR / f"zone_tracks_{camera_id}.csv"),
            args.with_tracks,
        ))

    ok = sum(1 for result in results if result)
    print(f"\nTamamlanan kamera: {ok} / {len(results)}")
    if ok != len(results):
        raise SystemExit("HATA: en az bir kamera icin onizleme uretilemedi.")


if __name__ == "__main__":
    main()

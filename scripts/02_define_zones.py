"""Faz 2 interaktif ROI cizim araci: poligonlari insan cizer, script kaydeder.

Referans kare (veya isi haritasi) uzerinde fare ile poligon cizilir ve sonuc
configs/zones_<camera_id>.yaml dosyasina yazilir. Script hicbir bolgeyi
onermez, tahmin etmez veya kenar tespitiyle uretmez; ROI bir karardir.

Kaydedilen koordinatlar her zaman ORIJINAL piksel uzayindadir; ekranda
olceklenmis gosterim yalnizca goruntuleme icindir.

Ornek:
    python scripts/02_define_zones.py --camera camA
    python scripts/02_define_zones.py --camera camA --background outputs/zones/heatmap_camA.jpg
    python scripts/02_define_zones.py --camera camB --load
"""

import argparse
from pathlib import Path

import cv2
import yaml

# shapely varsa poligon gecerliligi onunla dogrulanir; yoksa dahili kesisim
# kontrolu kullanilir (yeni bagimlilik kurulmaz).
try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    HAS_SHAPELY = True
except ImportError:
    _ShapelyPolygon = None
    HAS_SHAPELY = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_DIR = PROJECT_ROOT / "data" / "reference_frames"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"

WINDOW_NAME = "MCMOT_Pro - ROI cizimi"
DISPLAY_MAX_W = 1600         # ekranda gosterilecek en fazla genislik (px)
DISPLAY_MAX_H = 900          # ekranda gosterilecek en fazla yukseklik (px)
HELP_BAR_HEIGHT = 46         # ust yardim seridi yuksekligi (px)

DEFAULT_ZONE_TYPE = "transition"
MIN_POLYGON_POINTS = 3

COLOR_SAVED = (0, 255, 0)        # tamamlanmis zone (BGR)
COLOR_CURRENT = (0, 255, 255)    # cizilmekte olan poligon
COLOR_RUBBER = (190, 190, 190)   # son noktadan imlece ince cizgi
COLOR_LANE = (255, 0, 255)       # lane_divider
COLOR_BAR_BG = (32, 32, 32)
COLOR_BAR_TEXT = (255, 255, 255)

HELP_TEXT = """
Kisayollar:
  sol tik : poligona nokta ekle
  u       : son noktayi geri al
  c       : mevcut poligonu temizle
  n       : poligonu bitir ve zone_id sor (terminale gecin)
  l       : LANE DIVIDER modu - sonraki 2 tik seritleri ayiran cizgiyi tanimlar
  s       : YAML'a kaydet
  q       : kaydetmeden cik (onay sorulur)
  h       : bu yardimi terminale yaz
NOT: zone_id ve onay sorulari TERMINALDEN okunur; soru cikinca terminale gecin.
"""


# --------------------------------------------------------------------------
# geometri dogrulama
# --------------------------------------------------------------------------

def _orientation(a, b, c):
    """Uc noktanin donus yonu: 1 saat yonunun tersi, -1 saat yonu, 0 dogrusal."""
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _on_segment(a, b, p):
    """p noktasi (a,b) dogru parcasinin sinir kutusunda mi (dogrusal varsayimi)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, p3, p4):
    """Iki dogru parcasi kesisiyor mu (dogrusal ortusme dahil)."""
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def polygon_self_intersects(points):
    """Poligon kendisiyle kesisiyor mu (shapely yoksa kenar-kenar kontrol)."""
    if HAS_SHAPELY:
        try:
            return not _ShapelyPolygon(points).is_valid
        except Exception:
            return True

    count = len(points)
    edges = [(points[i], points[(i + 1) % count]) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if j == i or (j - i) % count == 1 or (i - j) % count == 1:
                continue  # komsu kenarlar ucunu paylasir, dogal
            if segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False


def validate_zones(zones, frame_size):
    """Kayit oncesi tum zone'lari dogrular; Turkce hata listesi dondurur."""
    errors = []
    width, height = frame_size
    seen_ids = set()

    if not zones:
        errors.append("Kaydedilecek zone yok; once poligon cizip 'n' ile bitirin.")

    for index, zone in enumerate(zones, start=1):
        zone_id = zone.get("zone_id") or ""
        label = zone_id or f"#{index}"

        if not zone_id.strip():
            errors.append(f"Zone {label}: zone_id bos olamaz.")
        elif zone_id in seen_ids:
            errors.append(f"Zone {label}: zone_id tekrar ediyor.")
        else:
            seen_ids.add(zone_id)

        polygon = zone.get("polygon") or []
        if len(polygon) < MIN_POLYGON_POINTS:
            errors.append(f"Zone {label}: poligon en az {MIN_POLYGON_POINTS} nokta "
                          f"icermeli (su an {len(polygon)}).")
            continue

        outside = [p for p in polygon
                   if not (0 <= p[0] < width and 0 <= p[1] < height)]
        if outside:
            errors.append(f"Zone {label}: {len(outside)} nokta kare sinirlari "
                          f"disinda ({width}x{height}).")

        if len(set(map(tuple, polygon))) != len(polygon):
            errors.append(f"Zone {label}: ayni nokta birden fazla kez eklenmis.")
        elif polygon_self_intersects([tuple(p) for p in polygon]):
            errors.append(f"Zone {label}: poligon kendisiyle kesisiyor (gecersiz sekil).")

        divider = zone.get("lane_divider")
        if divider is not None:
            if len(divider) != 2:
                errors.append(f"Zone {label}: lane_divider tam 2 nokta olmali "
                              f"(su an {len(divider)}).")
            else:
                bad = [p for p in divider
                       if not (0 <= p[0] < width and 0 <= p[1] < height)]
                if bad:
                    errors.append(f"Zone {label}: lane_divider noktasi kare disinda.")
                if tuple(divider[0]) == tuple(divider[1]):
                    errors.append(f"Zone {label}: lane_divider iki ayni noktadan olusuyor.")
    return errors


# --------------------------------------------------------------------------
# YAML
# --------------------------------------------------------------------------

def build_yaml_text(camera_id, frame_size, reference_frame, zones):
    """Kontrat bicimine uygun YAML metnini elle uretir (nokta ciftleri satir ici)."""
    lines = [
        f"camera_id: {camera_id}",
        f"frame_size: [{frame_size[0]}, {frame_size[1]}]",
        f'reference_frame: "{reference_frame}"',
        "zones:",
    ]
    for zone in zones:
        lines.append(f'  - zone_id: "{zone["zone_id"]}"')
        lines.append(f'    type: "{zone.get("type") or DEFAULT_ZONE_TYPE}"')
        lines.append("    polygon:")
        for x, y in zone["polygon"]:
            lines.append(f"      - [{int(x)}, {int(y)}]")
        divider = zone.get("lane_divider")
        if divider:  # tanimli degilse YAML'a hic yazilmaz (null yazilmaz)
            lines.append("    lane_divider:")
            for x, y in divider:
                lines.append(f"      - [{int(x)}, {int(y)}]")
        notes = (zone.get("notes") or "").replace('"', "'")
        lines.append(f'    notes: "{notes}"')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_zones_yaml(path):
    """Mevcut zones YAML'ini okur; (zones listesi, hata) dondurur."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [], f"{path} okunamadi: {exc}"

    raw_zones = data.get("zones")
    if not isinstance(raw_zones, list):
        return [], f"{path} icinde 'zones' listesi yok."

    zones = []
    for item in raw_zones:
        if not isinstance(item, dict):
            continue
        polygon = [[int(p[0]), int(p[1])] for p in (item.get("polygon") or [])
                   if isinstance(p, (list, tuple)) and len(p) == 2]
        divider = item.get("lane_divider")
        divider = ([[int(p[0]), int(p[1])] for p in divider]
                   if isinstance(divider, list) and len(divider) == 2 else None)
        zones.append({
            "zone_id": str(item.get("zone_id") or ""),
            "type": str(item.get("type") or DEFAULT_ZONE_TYPE),
            "polygon": polygon,
            "lane_divider": divider,
            "notes": str(item.get("notes") or ""),
        })
    return zones, None


# --------------------------------------------------------------------------
# editor
# --------------------------------------------------------------------------

class ZoneEditor:
    """Poligon cizim durumunu ve ekran gosterimini yonetir."""

    def __init__(self, camera_id, background, background_rel, out_path):
        self.camera_id = camera_id
        self.background = background
        self.background_rel = background_rel
        self.out_path = out_path

        height, width = background.shape[:2]
        self.frame_size = (width, height)
        self.scale = min(1.0, DISPLAY_MAX_W / width, DISPLAY_MAX_H / height)
        self.display_size = (max(1, int(round(width * self.scale))),
                             max(1, int(round(height * self.scale))))
        self.display_bg = (background if self.scale == 1.0
                           else cv2.resize(background, self.display_size,
                                           interpolation=cv2.INTER_AREA))

        self.zones = []
        self.current = []          # orijinal piksel uzayinda [x, y] listesi
        self.lane_points = []
        self.lane_mode = False
        self.cursor = None
        self.dirty = False         # kaydedilmemis degisiklik var mi
        self.pending_lane_divider = None   # 'n' ile bitirilecek poligona eklenecek

    # --- koordinat donusumu ---

    def to_original(self, x, y):
        """Ekran koordinatini orijinal piksel uzayina cevirir ve sinirlar."""
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        ox = min(max(ox, 0), self.frame_size[0] - 1)
        oy = min(max(oy, 0), self.frame_size[1] - 1)
        return [ox, oy]

    def to_display(self, point):
        return (int(round(point[0] * self.scale)), int(round(point[1] * self.scale)))

    # --- fare ---

    def on_mouse(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = self.to_original(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            point = self.to_original(x, y)
            if self.lane_mode:
                self.lane_points.append(point)
                print(f"  lane_divider noktasi {len(self.lane_points)}/2: {point}")
                if len(self.lane_points) == 2:
                    self.finish_lane_divider()
            else:
                self.current.append(point)
                print(f"  nokta {len(self.current)}: {point}")
            self.dirty = True

    # --- durum islemleri ---

    def undo(self):
        target = self.lane_points if self.lane_mode else self.current
        if not target:
            print("  Geri alinacak nokta yok.")
            return
        removed = target.pop()
        print(f"  Son nokta geri alindi: {removed} (kalan {len(target)})")

    def clear_current(self):
        if self.lane_mode:
            self.lane_points = []
            print("  LANE DIVIDER noktalari temizlendi.")
        elif self.current:
            print(f"  Mevcut poligon temizlendi ({len(self.current)} nokta silindi).")
            self.current = []
        else:
            print("  Temizlenecek nokta yok.")

    def toggle_lane_mode(self):
        self.lane_mode = not self.lane_mode
        self.lane_points = []
        if self.lane_mode:
            target = ("cizilmekte olan poligon" if self.current
                      else (f"son zone '{self.zones[-1]['zone_id']}'" if self.zones
                            else "HENUZ HEDEF YOK"))
            print(f"  LANE DIVIDER modu ACIK - iki tik bekleniyor. Hedef: {target}")
            if not self.current and not self.zones:
                print("  UYARI: once poligon cizin; lane_divider bagimsiz kaydedilemez.")
        else:
            print("  LANE DIVIDER modu KAPALI (poligon cizimine donuldu).")

    def finish_lane_divider(self):
        """Iki nokta tamamlandiginda lane_divider'i uygun zone'a baglar."""
        divider = list(self.lane_points)
        self.lane_points = []
        self.lane_mode = False
        if self.current:
            self.pending_lane_divider = divider
            print(f"  lane_divider tanimlandi: {divider} -> cizilmekte olan poligona "
                  "'n' ile bitirdiginizde eklenecek.")
        elif self.zones:
            self.zones[-1]["lane_divider"] = divider
            print(f"  lane_divider tanimlandi: {divider} -> zone "
                  f"'{self.zones[-1]['zone_id']}' guncellendi.")
        else:
            print("  HATA: lane_divider baglanacak zone yok; atildi.")
        print("  LANE DIVIDER modu KAPALI.")

    def finish_polygon(self):
        """Mevcut poligonu bitirir; zone_id, type ve notu terminalden sorar."""
        if len(self.current) < MIN_POLYGON_POINTS:
            print(f"  HATA: poligon en az {MIN_POLYGON_POINTS} nokta icermeli "
                  f"(su an {len(self.current)}). Cizime devam edin.")
            return
        print("\n  Poligon tamamlandi. Bilgileri terminale girin:")
        try:
            zone_id = input("    zone_id (or. A-EXIT-01): ").strip()
            zone_type = input(f"    type [{DEFAULT_ZONE_TYPE}]: ").strip() or DEFAULT_ZONE_TYPE
            notes = input("    notes (opsiyonel): ").strip()
        except EOFError:
            print("  HATA: terminal girdisi okunamadi; poligon bitirilemedi.")
            return

        if not zone_id:
            print("  HATA: zone_id bos olamaz; poligon bitirilemedi (noktalar duruyor).")
            return
        if any(z["zone_id"] == zone_id for z in self.zones):
            print(f"  HATA: '{zone_id}' zaten tanimli; farkli bir zone_id verin.")
            return

        zone = {
            "zone_id": zone_id,
            "type": zone_type,
            "polygon": list(self.current),
            "lane_divider": self.pending_lane_divider,
            "notes": notes,
        }
        self.zones.append(zone)
        divider_text = (f", lane_divider {self.pending_lane_divider}"
                        if self.pending_lane_divider else "")
        print(f"  Zone eklendi: '{zone_id}' ({zone_type}), "
              f"{len(self.current)} nokta{divider_text}. Toplam zone: {len(self.zones)}\n")
        self.current = []
        self.pending_lane_divider = None
        self.dirty = True

    # --- kayit ---

    def save(self):
        """Dogrulayip YAML'a yazar; basarili olursa True dondurur."""
        if self.current:
            print(f"  UYARI: bitirilmemis {len(self.current)} noktali poligon var; "
                  "kayda dahil edilmedi ('n' ile bitirin).")

        errors = validate_zones(self.zones, self.frame_size)
        if errors:
            print("\n  KAYIT YAPILMADI - dogrulama hatalari:")
            for err in errors:
                print(f"    - {err}")
            print(f"  (dogrulama motoru: {'shapely' if HAS_SHAPELY else 'dahili kesisim kontrolu'})")
            return False

        text = build_yaml_text(self.camera_id, self.frame_size,
                               self.background_rel, self.zones)
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            print(f"  HATA: uretilen YAML gecersiz, kayit iptal: {exc}")
            return False
        if not isinstance(parsed, dict) or len(parsed.get("zones") or []) != len(self.zones):
            print("  HATA: uretilen YAML beklenen yapida degil, kayit iptal.")
            return False

        if self.out_path.exists():
            print(f"\n  '{self.out_path}' zaten var.")
            try:
                answer = input("  Uzerine yazilsin mi? (e/h): ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("e", "evet", "y", "yes"):
                print("  Kayit iptal edildi; dosyaya dokunulmadi.")
                return False
            backup = self.out_path.with_suffix(self.out_path.suffix + ".bak")
            backup.write_bytes(self.out_path.read_bytes())
            print(f"  Yedek alindi: {backup}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(text, encoding="utf-8")
        self.dirty = False
        print(f"  KAYDEDILDI: {self.out_path} ({len(self.zones)} zone, "
              f"koordinatlar {self.frame_size[0]}x{self.frame_size[1]} orijinal uzayda)")
        return True

    # --- cizim ---

    def render(self):
        """Ekranda gosterilecek goruntuyu uretir (orijinal kare degistirilmez)."""
        canvas = self.display_bg.copy()

        for zone in self.zones:
            points = [self.to_display(p) for p in zone["polygon"]]
            for i, point in enumerate(points):
                cv2.line(canvas, point, points[(i + 1) % len(points)], COLOR_SAVED, 2, cv2.LINE_AA)
                cv2.circle(canvas, point, 3, COLOR_SAVED, -1)
            cx = sum(p[0] for p in points) // len(points)
            cy = sum(p[1] for p in points) // len(points)
            cv2.putText(canvas, zone["zone_id"], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, zone["zone_id"], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SAVED, 1, cv2.LINE_AA)
            if zone.get("lane_divider"):
                a, b = (self.to_display(p) for p in zone["lane_divider"])
                cv2.line(canvas, a, b, COLOR_LANE, 2, cv2.LINE_AA)

        current = [self.to_display(p) for p in self.current]
        for i, point in enumerate(current):
            if i > 0:
                cv2.line(canvas, current[i - 1], point, COLOR_CURRENT, 2, cv2.LINE_AA)
            cv2.circle(canvas, point, 4, COLOR_CURRENT, -1)
        if current and self.cursor is not None and not self.lane_mode:
            cv2.line(canvas, current[-1], self.to_display(self.cursor),
                     COLOR_RUBBER, 1, cv2.LINE_AA)

        lane = [self.to_display(p) for p in self.lane_points]
        for point in lane:
            cv2.circle(canvas, point, 4, COLOR_LANE, -1)
        if len(lane) == 1 and self.cursor is not None:
            cv2.line(canvas, lane[0], self.to_display(self.cursor), COLOR_RUBBER, 1, cv2.LINE_AA)
        if self.pending_lane_divider:
            a, b = (self.to_display(p) for p in self.pending_lane_divider)
            cv2.line(canvas, a, b, COLOR_LANE, 2, cv2.LINE_AA)

        self._draw_help_bar(canvas)
        return canvas

    def _draw_help_bar(self, canvas):
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], HELP_BAR_HEIGHT), COLOR_BAR_BG, -1)
        mode = "LANE DIVIDER" if self.lane_mode else "POLIGON"
        active = f"yeni (#{len(self.zones) + 1})"
        count = len(self.lane_points) if self.lane_mode else len(self.current)
        first = (f"{self.camera_id} | zone: {active} | nokta: {count} | mod: {mode} | "
                 f"kayitli zone: {len(self.zones)}"
                 + ("" if self.scale == 1.0 else f" | gosterim %{self.scale * 100:.0f}"))
        second = "u geri  c temizle  n bitir  l lane  s kaydet  q cikis  h yardim"
        cv2.putText(canvas, first, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_BAR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, second, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOR_BAR_TEXT, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------

def run_editor(editor):
    """cv2 penceresini acar ve tus dongusunu isletir."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, editor.on_mouse)
    print(HELP_TEXT)

    try:
        while True:
            cv2.imshow(WINDOW_NAME, editor.render())
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Pencere kapatildi; kaydedilmeden cikildi.")
                return
            if key in (255, 0):
                continue

            char = chr(key).lower() if 32 <= key < 127 else ""
            if char == "u":
                editor.undo()
            elif char == "c":
                editor.clear_current()
            elif char == "n":
                editor.finish_polygon()
            elif char == "l":
                editor.toggle_lane_mode()
            elif char == "h":
                print(HELP_TEXT)
            elif char == "s":
                editor.save()
            elif char == "q":
                if not editor.dirty:
                    print("Cikiliyor (kaydedilmemis degisiklik yok).")
                    return
                try:
                    answer = input("  Kaydedilmemis degisiklikler var. "
                                   "Kaydetmeden cikilsin mi? (e/h): ").strip().lower()
                except EOFError:
                    answer = "e"
                if answer in ("e", "evet", "y", "yes"):
                    print("Kaydedilmeden cikildi.")
                    return
                print("  Cikis iptal edildi.")
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Faz 2 interaktif ROI cizim araci (poligonlari insan cizer)."
    )
    parser.add_argument("--camera", required=True, help="kamera kimligi (or. camA)")
    parser.add_argument("--background", type=Path, default=None,
                        help="arka plan goruntusu; varsayilan data/reference_frames/ref_<cam>.jpg")
    parser.add_argument("--load", action="store_true",
                        help="mevcut zones YAML'ini yukleyip uzerine duzenle")
    parser.add_argument("--out", type=Path, default=None,
                        help="cikti YAML yolu; varsayilan configs/zones_<cam>.yaml")
    args = parser.parse_args()

    background_path = (args.background if args.background is not None
                       else DEFAULT_REF_DIR / f"ref_{args.camera}.jpg")
    if not background_path.is_absolute():
        background_path = PROJECT_ROOT / background_path
    if not background_path.is_file():
        raise SystemExit(
            f"HATA: arka plan goruntusu bulunamadi: {background_path}\n"
            f"      Once referans kareyi uretin: "
            f"python scripts/02_extract_reference_frames.py --camera {args.camera}"
        )

    background = cv2.imread(str(background_path))
    if background is None:
        raise SystemExit(f"HATA: goruntu cv2 ile okunamadi (bozuk veya desteklenmeyen bicim): "
                         f"{background_path}")

    out_path = (args.out if args.out is not None
                else DEFAULT_CONFIG_DIR / f"zones_{args.camera}.yaml")
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    try:
        background_rel = background_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        background_rel = background_path.as_posix()

    editor = ZoneEditor(args.camera, background, background_rel, out_path)

    height, width = background.shape[:2]
    print(f"Kamera        : {args.camera}")
    print(f"Arka plan     : {background_path} ({width}x{height})")
    if editor.scale < 1.0:
        print(f"Gosterim      : %{editor.scale * 100:.0f} olcekli "
              f"({editor.display_size[0]}x{editor.display_size[1]}); "
              "kaydedilen koordinatlar orijinal uzayda kalir.")
    print(f"Cikti         : {out_path}")
    print(f"Dogrulama     : {'shapely' if HAS_SHAPELY else 'dahili kesisim kontrolu (shapely yok)'}")

    if args.load:
        if out_path.is_file():
            zones, error = load_zones_yaml(out_path)
            if error:
                print(f"UYARI: {error} Bos baslaniyor.")
            else:
                editor.zones = zones
                print(f"Yuklendi      : {len(zones)} zone ({out_path})")
        else:
            print(f"UYARI: --load verildi ama dosya yok: {out_path}. Bos baslaniyor.")

    run_editor(editor)


if __name__ == "__main__":
    main()

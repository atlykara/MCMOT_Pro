"""Faz 2: referans kare uzerinde zone/ROI cizme araci.

Kullanim:
    python scripts/02_define_zones.py --camera camA \
        --image outputs/reference_frames/camA/camA_t60s.jpg

Tus:
    sol tik : nokta ekle
    u       : son noktayi geri al
    n       : mevcut poligonu zone olarak kaydet, yeni zone'a gec
    s       : YAML dosyasina kaydet ve cik
    q / esc : kaydetmeden cik

Not: Bu arac GUI penceresi actigi icin lokal Mac terminalinde calistirilmalidir.
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import make_zone_yaml  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "configs"


COLORS = [
    (0, 255, 255),
    (0, 200, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
]


class ZoneDrawer:
    def __init__(self, image, camera_id: str, zone_prefix: str):
        self.base = image
        self.camera_id = camera_id
        self.zone_prefix = zone_prefix
        self.current: list[list[int]] = []
        self.zones: list[dict] = []

    def add_point(self, x: int, y: int) -> None:
        self.current.append([int(x), int(y)])

    def undo(self) -> None:
        if self.current:
            self.current.pop()

    def finish_current(self) -> None:
        if len(self.current) < 3:
            print("UYARI: Bir zone en az 3 nokta olmali.")
            return
        zone_id = f"{self.zone_prefix}-{len(self.zones) + 1:02d}"
        self.zones.append(
            {
                "zone_id": zone_id,
                "kind": "roi",
                "polygon": self.current.copy(),
                "notes": "",
            }
        )
        print(f"Zone eklendi: {zone_id} ({len(self.current)} nokta)")
        self.current.clear()

    def render(self):
        img = self.base.copy()

        for idx, zone in enumerate(self.zones):
            color = COLORS[idx % len(COLORS)]
            pts = zone["polygon"]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                cv2.line(img, tuple(a), tuple(b), color, 2)
            for p in pts:
                cv2.circle(img, tuple(p), 4, color, -1)
            cv2.putText(img, zone["zone_id"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.current:
            color = (255, 255, 255)
            for p in self.current:
                cv2.circle(img, tuple(p), 4, color, -1)
            for a, b in zip(self.current, self.current[1:]):
                cv2.line(img, tuple(a), tuple(b), color, 2)

        help_text = "sol tik: nokta | u: geri al | n: zone bitir | s: kaydet | q/esc: cik"
        cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 900), 45), (0, 0, 0), -1)
        cv2.putText(img, help_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img


def mouse_callback(event, x, y, flags, drawer: ZoneDrawer):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawer.add_point(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Referans kare uzerinde zone/ROI cizer.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--image", type=Path, required=True, help="Referans JPG yolu")
    parser.add_argument("--zone-prefix", default=None, help="or. A-EXIT, B-ENTRY")
    parser.add_argument("--output", type=Path, default=None, help="YAML cikti yolu")
    args = parser.parse_args()

    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"HATA: image okunamadi: {image_path}")

    zone_prefix = args.zone_prefix or args.camera.upper()
    out_path = args.output or (CONFIG_DIR / f"zones_{args.camera}.yaml")

    drawer = ZoneDrawer(image=image, camera_id=args.camera, zone_prefix=zone_prefix)
    window = f"define zones - {args.camera}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(image.shape[1], 1400), min(image.shape[0], 900))
    cv2.setMouseCallback(window, mouse_callback, drawer)

    print("Zone cizimi basladi.")
    print("Tuslar: sol tik=nokta, u=geri al, n=zone bitir, s=kaydet, q/esc=cik")

    while True:
        cv2.imshow(window, drawer.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            print("Kaydetmeden cikildi.")
            break
        if key == ord("u"):
            drawer.undo()
        elif key == ord("n"):
            drawer.finish_current()
        elif key == ord("s"):
            if drawer.current:
                drawer.finish_current()
            if not drawer.zones:
                print("UYARI: Kaydedilecek zone yok.")
                continue
            h, w = image.shape[:2]
            body = make_zone_yaml(camera_id=args.camera, image_size=(w, h), zones=drawer.zones)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(body, fh, sort_keys=False, allow_unicode=True)
            print(f"Kaydedildi: {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

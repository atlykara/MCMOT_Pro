"""Faz 2: zone YAML dosyasini referans kare uzerinde gorsellestirir."""

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import load_zone_config  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "configs"
OUT_DIR = PROJECT_ROOT / "outputs" / "zones" / "previews"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zone poligonlarini referans kareye cizer.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--zones", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
    zones_path = args.zones or (CONFIG_DIR / f"zones_{args.camera}.yaml")
    output_path = args.output or (OUT_DIR / f"zones_preview_{args.camera}.jpg")

    if not zones_path.is_absolute():
        zones_path = PROJECT_ROOT / zones_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"HATA: image okunamadi: {image_path}")

    config = load_zone_config(zones_path)
    overlay = image.copy()

    colors = [(0, 255, 255), (0, 200, 0), (255, 0, 255), (255, 0, 0)]
    for idx, zone in enumerate(config.zones):
        color = colors[idx % len(colors)]
        pts = [(int(x), int(y)) for x, y in zone.polygon]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(overlay, a, b, color, 4)
        for p in pts:
            cv2.circle(overlay, p, 7, color, -1)
        cv2.putText(overlay, zone.zone_id, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise SystemExit(f"HATA: cikti yazilamadi: {output_path}")
    print(f"Yazildi: {output_path}")


if __name__ == "__main__":
    main()

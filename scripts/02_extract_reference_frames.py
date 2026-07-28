"""Faz 2: referans kare cikarma.

Zone/ROI cizmeden once her kamera icin temsil gucu yuksek birkac kare lazim.
Bu script video icinden istenen saniyelerde JPG kareler cikarir.

Ornek:
    python scripts/02_extract_reference_frames.py --camera camA \
        --source /Users/aliefesarioglu/Desktop/kayseri_cam_1.mp4 \
        --times 5 60 120 180
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "reference_frames"


def load_camera_entry(config_path: Path, camera_id: str) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for entry in config.get("cameras", []):
        if entry.get("camera_id") == camera_id:
            return entry
    raise SystemExit(f"HATA: {camera_id!r} {config_path} icinde bulunamadi.")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_frame(video_path: Path, second: float, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit(f"HATA: {second:.2f}. saniyeden kare okunamadi: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise SystemExit(f"HATA: kare yazilamadi: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kamera videosundan Faz 2 referans kareleri cikarir.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--source", default=None, help="Video yolu; verilmezse configs/cameras.yaml kullanilir")
    parser.add_argument("--times", type=float, nargs="+", default=[5, 30, 60, 120],
                        help="Cikarilacak saniyeler")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    if args.source:
        video_path = resolve_path(args.source)
    else:
        entry = load_camera_entry(args.config, args.camera)
        raw = entry.get("video_path") or ""
        if not raw:
            raise SystemExit(f"HATA: {args.camera} icin video_path bos; --source ver.")
        video_path = resolve_path(raw)

    if not video_path.is_file():
        raise SystemExit(f"HATA: video bulunamadi: {video_path}")

    for second in args.times:
        safe_second = str(second).replace(".", "_")
        out_path = OUT_DIR / args.camera / f"{args.camera}_t{safe_second}s.jpg"
        extract_frame(video_path, second, out_path)
        print(f"OK: {second:g}s -> {out_path}")


if __name__ == "__main__":
    main()

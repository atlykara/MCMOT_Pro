"""Buyuk ham videodan kisa test kesiti cikarir.

configs/cameras.yaml icindeki camera_id'ye ait video_path okunur,
--start saniyesinden itibaren --duration saniyelik kesit kare kare
yeniden kodlanarak data/samples/<camera>_sample.mp4 olarak yazilir.
Orijinal fps ve cozunurluk korunur.

Ornek:
    python scripts/00_make_sample.py --camera camA --start 60 --duration 45
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def load_camera_entry(config_path: Path, camera_id: str) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cameras = config.get("cameras") or []
    for entry in cameras:
        if entry.get("camera_id") == camera_id:
            return entry
    known = ", ".join(str(e.get("camera_id")) for e in cameras)
    raise SystemExit(f"HATA: '{camera_id}' {config_path} icinde yok. Tanimli kameralar: {known}")


def resolve_video_path(entry: dict) -> Path:
    raw = entry.get("video_path") or ""
    if not raw:
        raise SystemExit(f"HATA: '{entry.get('camera_id')}' icin video_path bos. Once configs/cameras.yaml doldurulmali.")
    video_path = Path(raw)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.is_file():
        raise SystemExit(f"HATA: video dosyasi bulunamadi: {video_path}")
    return video_path


def make_sample(video_path: Path, output_path: Path, start: float, duration: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise SystemExit(f"HATA: video fps degeri okunamadi ({fps}): {video_path}")

    start_frame = int(round(start * fps))
    target_frames = int(round(duration * fps))
    if total_frames > 0 and start_frame >= total_frames:
        cap.release()
        video_seconds = total_frames / fps
        raise SystemExit(
            f"HATA: --start {start:g} sn video suresinin ({video_seconds:.1f} sn) disinda."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"HATA: cikti dosyasi acilamadi: {output_path}")

    written = 0
    try:
        while written < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()

    if written == 0:
        if output_path.exists():
            output_path.unlink()
        raise SystemExit("HATA: hic kare okunamadi, cikti uretilmedi.")

    actual_duration = written / fps
    print(f"Kaynak video : {video_path}")
    print(f"Kesit        : {start:g} sn baslangic, hedef {duration:g} sn ({target_frames} kare)")
    print(f"Yazilan kare : {written}")
    print(f"Kesit suresi : {actual_duration:.2f} sn @ {fps:g} fps, {width}x{height}")
    print(f"Cikti        : {output_path}")
    if written < target_frames:
        print("UYARI: video sonuna ulasildi, kesit istenenden kisa.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ham videodan kisa test kesiti cikarir.")
    parser.add_argument("--camera", required=True, help="cameras.yaml icindeki camera_id (or. camA)")
    parser.add_argument("--start", type=float, default=0.0, help="baslangic saniyesi (varsayilan 0)")
    parser.add_argument("--duration", type=float, required=True, help="kesit suresi (saniye)")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="kamera config dosyasi yolu"
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start negatif olamaz")
    if args.duration <= 0:
        parser.error("--duration pozitif olmali")
    if not args.config.is_file():
        raise SystemExit(f"HATA: config dosyasi bulunamadi: {args.config}")

    entry = load_camera_entry(args.config, args.camera)
    video_path = resolve_video_path(entry)
    output_path = SAMPLES_DIR / f"{args.camera}_sample.mp4"
    make_sample(video_path, output_path, args.start, args.duration)


if __name__ == "__main__":
    main()

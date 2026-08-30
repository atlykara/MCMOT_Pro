"""Faz 1: tek kamera arac tespit + takip koşusu.

configs/cameras.yaml'daki kamera girdisini kullanarak videoyu isler,
kontrata uygun tracks_<camera>.jsonl ve (istege bagli) annotated video uretir.

Ornek:
    python scripts/01_run_detect_track.py --camera camA --max-frames 100
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.detect_track import detect_and_track, select_device  # noqa: E402
from mcmot.io_utils import JsonlWriter  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos"

BOX_COLOR = (0, 200, 0)
FOOT_COLOR = (0, 0, 255)


def load_camera_entry(config_path: Path, camera_id: str) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cameras = config.get("cameras") or []
    for entry in cameras:
        if entry.get("camera_id") == camera_id:
            return entry
    known = ", ".join(str(e.get("camera_id")) for e in cameras)
    raise SystemExit(f"HATA: '{camera_id}' {config_path} icinde yok. Tanimli kameralar: {known}")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def annotate(image, records) -> None:
    """Kare uzerine kutu, etiket ve foot_point isaretlerini cizer (yerinde)."""
    for rec in records:
        x1, y1, x2, y2 = (int(v) for v in rec["bbox_xyxy"])
        label = f"id:{rec['track_id']} {rec['class']} {rec['conf']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            image, label, (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 2, cv2.LINE_AA,
        )
        fx, fy = (int(v) for v in rec["foot_point"])
        cv2.circle(image, (fx, fy), 4, FOOT_COLOR, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek kamera arac tespit + takip (Faz 1).")
    parser.add_argument("--camera", required=True, help="cameras.yaml icindeki camera_id (or. camA)")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO model dosyasi (varsayilan yolo11m.pt)")
    parser.add_argument("--source", default=None, help="video yolu; verilmezse cameras.yaml'daki video_path")
    parser.add_argument("--max-frames", type=int, default=None, help="en fazla islenecek kare sayisi")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="eski ciktilari korumak icin dosya eki (or. human_v1_30s)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="tespit guven esigi (varsayilan 0.35)")
    parser.add_argument("--no-video", action="store_true", help="annotated video uretme")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="kamera config dosyasi yolu")
    parser.add_argument("--tracker", default="configs/bytetrack_buffered.yaml",
                        help="ByteTrack ayar dosyasi (standart: configs/bytetrack_buffered.yaml, buffer=90)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="YOLO cikarim cozunurlugu (standart: 960). Yuksek = kucuk araclar daha iyi, daha yavas.")
    args = parser.parse_args()

    # Custom tracker dosyasi PROJECT_ROOT altindaysa mutlak yola cevir;
    # "bytetrack.yaml" gibi built-in isimlere dokunma.
    tracker_arg = args.tracker
    candidate = PROJECT_ROOT / tracker_arg
    if candidate.is_file():
        tracker_arg = str(candidate)
    args.tracker = tracker_arg

    entry = load_camera_entry(args.config, args.camera)

    if args.source is not None:
        video_path = resolve_path(args.source)
    else:
        raw = entry.get("video_path") or ""
        if not raw:
            raise SystemExit(f"HATA: '{args.camera}' icin video_path bos ve --source verilmedi.")
        video_path = resolve_path(raw)
    if not video_path.is_file():
        raise SystemExit(f"HATA: video dosyasi bulunamadi: {video_path}")

    fps = float(entry.get("fps") or 0)
    if fps <= 0:
        fps = probe_fps(video_path)
        print(f"UYARI: cameras.yaml fps degeri yok, videodan okundu: {fps:g}")
    if fps <= 0:
        raise SystemExit("HATA: fps belirlenemedi.")

    device = select_device()
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Video  : {video_path}")

    suffix = f"{args.run_tag}_{args.camera}" if args.run_tag else args.camera
    jsonl_path = TRACKS_DIR / f"tracks_{suffix}.jsonl"
    video_out_path = VIDEOS_DIR / f"annotated_{suffix}.mp4"
    writer = None
    frames_done = 0
    t0 = time.perf_counter()

    try:
        with JsonlWriter(jsonl_path) as jsonl:
            for frame_result in detect_and_track(
                video_path=video_path,
                camera_id=args.camera,
                fps=fps,
                model_name=args.model,
                conf=args.conf,
                max_frames=args.max_frames,
                device=device,
                tracker=args.tracker,
                imgsz=args.imgsz,
            ):
                for record in frame_result.records:
                    jsonl.write(record)

                if not args.no_video:
                    image = frame_result.image
                    if writer is None:
                        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
                        h, w = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        if not writer.isOpened():
                            raise SystemExit(f"HATA: video cikti dosyasi acilamadi: {video_out_path}")
                    annotate(image, frame_result.records)
                    writer.write(image)

                frames_done = frame_result.frame_index + 1
                if frames_done % 100 == 0:
                    print(f"  kare {frames_done} islendi, {jsonl.count} kayit...")

            record_count = jsonl.count
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - t0
    print(f"Islenen kare : {frames_done} ({elapsed:.1f} sn, {frames_done / elapsed:.1f} fps)")
    print(f"JSONL kayit  : {record_count} -> {jsonl_path}")
    if not args.no_video and writer is not None:
        print(f"Annotated    : {video_out_path}")


if __name__ == "__main__":
    main()

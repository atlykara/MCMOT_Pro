"""Kamera videolarindan etiketleme icin zamansal olarak ayrilmis kareler cikarir."""

import argparse
import csv
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/vehicle_dataset"


def split_for_position(position: float) -> str:
    if position < 0.70:
        return "train"
    if position < 0.90:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Etiketleme icin kamera kareleri hazirlar.")
    parser.add_argument("--per-camera", type=int, default=180,
                        help="her kameradan alinacak kare sayisi (varsayilan 180)")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if args.per_camera < 10:
        raise SystemExit("HATA: saglikli zamansal bolme icin --per-camera en az 10 olmali.")

    camera_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/cameras.yaml").read_text(encoding="utf-8")
    )
    for split in ("train", "val", "test"):
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for camera in camera_config["cameras"]:
        camera_id = camera["camera_id"]
        video_path = PROJECT_ROOT / camera["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"HATA: video acilamadi: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        for sample_index in range(args.per_camera):
            position = (sample_index + 0.5) / args.per_camera
            frame_index = min(int(position * frame_count), frame_count - 1)
            split = split_for_position(position)
            timestamp = frame_index / fps
            filename = f"{camera_id}_{sample_index:06d}_{timestamp:07.2f}.jpg"
            output_path = DATASET_ROOT / "images" / split / filename

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"HATA: {camera_id} kare {frame_index} okunamadi.")
            if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise SystemExit(f"HATA: goruntu yazilamadi: {output_path}")
            manifest_rows.append({
                "camera_id": camera_id,
                "source_video": str(video_path.relative_to(PROJECT_ROOT)),
                "frame": frame_index,
                "timestamp": round(timestamp, 3),
                "split": split,
                "image": str(output_path.relative_to(PROJECT_ROOT)),
            })
        cap.release()

    manifest_path = DATASET_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Hazirlanan goruntu: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    print("Siradaki adim: tum goruntuleri etiketleyip YOLO .txt dosyalarini labels/ altina koyun.")


if __name__ == "__main__":
    main()

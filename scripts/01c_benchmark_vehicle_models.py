"""YOLO11s ve YOLO26n'i ayni kamera kesitlerinde operasyonel olarak karsilastirir.

Bu betik ground-truth etiketi olmadigi icin mAP olcmez. Hiz, tespit yogunlugu,
guven ve track boyunca sinif kararliligi raporlar.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

VEHICLE_CLASSES = {1, 2, 3, 5, 7}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_videos() -> list[Path]:
    config = yaml.safe_load((PROJECT_ROOT / "configs/cameras.yaml").read_text(encoding="utf-8"))
    return [PROJECT_ROOT / entry["video_path"] for entry in config["cameras"]]


def benchmark(model_path: Path, videos: list[Path], frames_per_camera: int, imgsz: int,
              conf: float, device: str) -> dict:
    model = YOLO(str(model_path))
    started = time.perf_counter()
    observations = defaultdict(list)
    confidences = []
    detections = 0
    frames = 0

    for video_path in videos:
        camera = video_path.stem.split("_")[0]
        results = model.track(
            source=str(video_path),
            tracker=str(PROJECT_ROOT / "configs/bytetrack_buffered.yaml"),
            persist=False,
            classes=sorted(VEHICLE_CLASSES),
            conf=conf,
            imgsz=imgsz,
            stream=True,
            device=device,
            verbose=False,
        )
        for frame_index, result in enumerate(results):
            if frame_index >= frames_per_camera:
                break
            frames += 1
            boxes = result.boxes
            if boxes is None:
                continue
            classes = boxes.cls.int().tolist()
            scores = boxes.conf.tolist()
            ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(classes)
            detections += len(classes)
            confidences.extend(scores)
            for track_id, class_id, score in zip(ids, classes, scores):
                if track_id >= 0:
                    observations[(camera, track_id)].append((class_id, score))

    elapsed = time.perf_counter() - started
    long_tracks = [items for items in observations.values() if len(items) >= 5]
    stable_tracks = sum(len({class_id for class_id, _ in items}) == 1 for items in long_tracks)
    majority_scores = []
    for items in long_tracks:
        majority_class = Counter(class_id for class_id, _ in items).most_common(1)[0][0]
        majority_scores.extend(score for class_id, score in items if class_id == majority_class)

    return {
        "model": model_path.stem,
        "device": device,
        "frames": frames,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 3),
        "detections": detections,
        "detections_per_frame": round(detections / frames, 3),
        "mean_detection_confidence": round(statistics.fmean(confidences), 4),
        "tracks_at_least_5_frames": len(long_tracks),
        "class_stability_rate": round(stable_tracks / len(long_tracks), 4) if long_tracks else None,
        "mean_majority_class_confidence": round(statistics.fmean(majority_scores), 4)
        if majority_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["models/yolo11s.pt", "models/yolo26n.pt"])
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/model_benchmark.json")
    args = parser.parse_args()

    videos = load_videos()
    fps = cv2.VideoCapture(str(videos[0])).get(cv2.CAP_PROP_FPS)
    frames_per_camera = int(args.seconds * fps)
    device = select_device()
    results = [
        benchmark(PROJECT_ROOT / model, videos, frames_per_camera, args.imgsz, args.conf, device)
        for model in args.models
    ]
    report = {
        "note": "Etiketsiz operasyonel karsilastirma; mAP/recall degildir.",
        "settings": vars(args) | {"device": device},
        "results": results,
    }
    report["settings"]["output"] = str(report["settings"]["output"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

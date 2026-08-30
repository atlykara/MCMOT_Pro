"""Tek kamera arac tespit + takip cekirdegi (Faz 1).

Ultralytics YOLO ile kare kare tespit + ByteTrack takibi yapar ve her arac
icin veri kontratina uygun kayit uretir. Re-ID, gorunum embedding'i veya
plaka tanima KULLANILMAZ; takip yalnizca hareket tabanli ByteTrack'tir.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# COCO sinif id -> kontrat sinif adi
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck", 1: "bicycle", 3: "motorcycle"}
KAYSERI_VEHICLE_NAMES = {
    "car",
    "van_minibus",
    "pickup",
    "bus",
    "truck",
    "special_vehicle",
    "motorcycle",
}

# size_class esikleri: bbox alani / kare alani
SIZE_SMALL_MAX = 0.01
SIZE_MEDIUM_MAX = 0.04


@dataclass
class FrameResult:
    """Tek karenin cikti paketi: kare goruntusu + kontrat kayitlari."""

    frame_index: int
    image: np.ndarray
    records: list


def select_device() -> str:
    """CUDA veya Apple MPS varsa hizlandirici, yoksa CPU secer."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def vehicle_class_mapping(model_names: dict) -> dict[int, str]:
    """COCO veya Kayseri modelinin sinif numaralarini kontrat adlarina cevirir."""
    names = {int(class_id): str(name) for class_id, name in model_names.items()}
    if set(names.values()) == KAYSERI_VEHICLE_NAMES:
        return {class_id: name for class_id, name in names.items() if name in KAYSERI_VEHICLE_NAMES}
    return {class_id: name for class_id, name in VEHICLE_CLASSES.items() if class_id in names}


def size_class_of(bbox_area: float, frame_area: float) -> str:
    ratio = bbox_area / frame_area if frame_area > 0 else 0.0
    if ratio < SIZE_SMALL_MAX:
        return "small"
    if ratio <= SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


def compute_hints(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> dict:
    """Tek kareden ucuz gorunum ipuclari uretir (model kullanilmaz).

    dominant_color: bbox'in alt yarisindaki piksellerin medyan BGR degeri.
    """
    h, w = image.shape[:2]
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2, yi2 = min(int(x2), w), min(int(y2), h)
    y_mid = (yi1 + yi2) // 2

    lower_half = image[y_mid:yi2, xi1:xi2]
    if lower_half.size == 0:
        lower_half = image[yi1:yi2, xi1:xi2]
    if lower_half.size > 0:
        dominant_color = [int(v) for v in np.median(lower_half.reshape(-1, 3), axis=0)]
    else:
        dominant_color = [0, 0, 0]

    bbox_w, bbox_h = x2 - x1, y2 - y1
    aspect_ratio = round(bbox_w / bbox_h, 2) if bbox_h > 0 else 0.0
    return {
        "dominant_color": dominant_color,
        "size_class": size_class_of(bbox_w * bbox_h, float(w * h)),
        "aspect_ratio": aspect_ratio,
    }


def detect_and_track(
    video_path: Path,
    camera_id: str,
    fps: float,
    model_name: str = "yolo11m.pt",
    conf: float = 0.35,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    tracker: str = "bytetrack.yaml",
    imgsz: int = 640,
) -> Iterator[FrameResult]:
    """Videoyu kare kare isler, her kare icin FrameResult uretir.

    Kayitlar akis halinde uretilir; kareler bellekte biriktirilmez.
    track_id atanmamis tespitler atlanir.
    """
    from ultralytics import YOLO

    if fps <= 0:
        raise ValueError(f"fps pozitif olmali, gelen: {fps}")
    if device is None:
        device = select_device()

    model = YOLO(model_name)
    class_mapping = vehicle_class_mapping(model.names)
    if not class_mapping:
        raise ValueError(f"Modelde desteklenen arac sinifi bulunamadi: {model.names}")
    results = model.track(
        source=str(video_path),
        tracker=tracker,
        persist=True,
        classes=sorted(class_mapping),
        conf=conf,
        imgsz=imgsz,
        stream=True,
        device=device,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        if max_frames is not None and frame_index >= max_frames:
            break

        image = result.orig_img
        records = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxys = boxes.xyxy.tolist()
            for track_id, cls_id, det_conf, xyxy in zip(ids, classes, confs, xyxys):
                if cls_id not in class_mapping:
                    continue
                x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy)
                records.append(
                    {
                        "timestamp": frame_index / fps,
                        "frame": frame_index,
                        "camera_id": camera_id,
                        "track_id": int(track_id),
                        "class": class_mapping[cls_id],
                        "conf": round(float(det_conf), 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "foot_point": [round((x1 + x2) / 2, 1), y2],
                        "hints": compute_hints(image, x1, y1, x2, y2),
                    }
                )

        yield FrameResult(frame_index=frame_index, image=image, records=records)

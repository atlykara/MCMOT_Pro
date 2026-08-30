"""Faz 2: ROI/zone odakli annotated video uretir.

Faz 1 videosu tum ekrandaki tum araclari cizer. Bu script ise:
  - configs/zones_<camera>.yaml poligonlarini cizer
  - sadece foot_point'i zone icinde olan araclara kutu/id basar
  - opsiyonel olarak ROI disini karartir

Ornek:
    python scripts/02_render_zone_video.py --camera camA \
      --source data/cams/camA_20fps.mp4
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import load_zone_config, zones_for_point  # noqa: E402


TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
CONFIG_DIR = PROJECT_ROOT / "configs"
VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos"


BOX_COLOR = (0, 220, 0)
FOOT_COLOR = (0, 0, 255)
ZONE_COLOR = (0, 255, 255)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_zone_records_by_frame(tracks_path: Path, zones_path: Path) -> dict[int, list[dict]]:
    zone_config = load_zone_config(zones_path)
    by_frame: dict[int, list[dict]] = defaultdict(list)

    for rec in iter_jsonl(tracks_path):
        foot = (float(rec["foot_point"][0]), float(rec["foot_point"][1]))
        matched_zones = zones_for_point(foot, zone_config.zones)
        if not matched_zones:
            continue
        rec = dict(rec)
        rec["zone_ids"] = [z.zone_id for z in matched_zones]
        by_frame[int(rec["frame"])].append(rec)

    return by_frame


def draw_zones(image: np.ndarray, zones_path: Path) -> None:
    zone_config = load_zone_config(zones_path)
    for zone in zone_config.zones:
        pts = [(int(x), int(y)) for x, y in zone.polygon]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(image, a, b, ZONE_COLOR, 3)
        for p in pts:
            cv2.circle(image, p, 5, ZONE_COLOR, -1)
        cv2.putText(image, zone.zone_id, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, ZONE_COLOR, 2)


def make_zone_mask(shape: tuple[int, int, int], zones_path: Path) -> np.ndarray:
    zone_config = load_zone_config(zones_path)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for zone in zone_config.zones:
        pts = np.array([[(int(x), int(y)) for x, y in zone.polygon]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
    return mask


def dim_outside_zones(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dimmed = (image * 0.35).astype(np.uint8)
    return np.where(mask[:, :, None] == 255, image, dimmed)


def annotate_records(image: np.ndarray, records: list[dict]) -> None:
    for rec in records:
        x1, y1, x2, y2 = (int(v) for v in rec["bbox_xyxy"])
        zones = ",".join(rec.get("zone_ids", []))
        label = f"id:{rec['track_id']} {rec['class']} {rec['conf']:.2f} {zones}"
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 7, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
        fx, fy = (int(v) for v in rec["foot_point"])
        cv2.circle(image, (fx, fy), 5, FOOT_COLOR, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI/zone odakli annotated video uretir.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--source", type=Path, required=True, help="Ham video yolu")
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--zones", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-dim", action="store_true", help="ROI disini karartma")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    tracks_path = args.tracks or (TRACKS_DIR / f"tracks_{args.camera}.jsonl")
    zones_path = args.zones or (CONFIG_DIR / f"zones_{args.camera}.yaml")
    output_path = args.output or (VIDEOS_DIR / f"zone_annotated_{args.camera}.mp4")

    if not tracks_path.is_absolute():
        tracks_path = PROJECT_ROOT / tracks_path
    if not zones_path.is_absolute():
        zones_path = PROJECT_ROOT / zones_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not source.is_file():
        raise SystemExit(f"HATA: video yok: {source}")
    if not tracks_path.is_file():
        raise SystemExit(f"HATA: tracks yok: {tracks_path}")
    if not zones_path.is_file():
        raise SystemExit(f"HATA: zones yok: {zones_path}")

    print("Zone icindeki kayıtlar indeksleniyor...")
    by_frame = load_zone_records_by_frame(tracks_path, zones_path)
    print(f"Zone icinde arac gorulen kare sayisi: {len(by_frame)}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"HATA: cikti video acilamadi: {output_path}")

    mask = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            if mask is None:
                mask = make_zone_mask(frame.shape, zones_path)

            if not args.no_dim:
                frame = dim_outside_zones(frame, mask)

            draw_zones(frame, zones_path)
            annotate_records(frame, by_frame.get(frame_idx, []))
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  {frame_idx} kare yazildi...")
    finally:
        cap.release()
        writer.release()

    print(f"Bitti: {frame_idx} kare -> {output_path}")


if __name__ == "__main__":
    main()

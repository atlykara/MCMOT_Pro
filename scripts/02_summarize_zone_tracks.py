"""Faz 2: zone icindeki track hareketlerini ozetler.

Bu script'in sorusu:
    "Bir track_id, cizdigimiz zone icinde nasil hareket etti?"

Girdi:
    outputs/tracks/tracks_<camera>.jsonl
    configs/zones_<camera>.yaml

Cikti:
    outputs/zones/zone_tracks_<camera>.csv

Not:
    direction_label goruntu koordinatina gore verilir:
      - right / left: x ekseni baskin
      - down / up    : y ekseni baskin (OpenCV'de y asagi dogru artar)
    Bu etiket trafik yonu degil, goruntu uzerindeki hareket yonudur.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import load_zone_config, zones_for_point  # noqa: E402


TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONES_OUT_DIR = PROJECT_ROOT / "outputs" / "zones"


FIELDS = [
    "camera_id",
    "track_id",
    "zone_id",
    "zone_kind",
    "class",
    "enter_timestamp",
    "exit_timestamp",
    "duration_s",
    "enter_frame",
    "exit_frame",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "distance_px",
    "speed_px_s",
    "direction_label",
    "angle_deg",
    "num_points",
    "mean_conf",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def direction_label(dx: float, dy: float, min_distance: float) -> str:
    """Basit goruntu-yon etiketi uretir."""
    dist = math.hypot(dx, dy)
    if dist < min_distance:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def summarize_zone_tracks(tracks_path: Path, zones_path: Path, min_distance: float) -> list[dict]:
    zone_config = load_zone_config(zones_path)

    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    zone_by_id = {zone.zone_id: zone for zone in zone_config.zones}

    for rec in iter_jsonl(tracks_path):
        foot = (float(rec["foot_point"][0]), float(rec["foot_point"][1]))
        for zone in zones_for_point(foot, zone_config.zones):
            grouped[(int(rec["track_id"]), zone.zone_id)].append(rec)

    rows = []
    for (track_id, zone_id), points in grouped.items():
        points.sort(key=lambda r: (float(r["timestamp"]), int(r["frame"])))
        first = points[0]
        last = points[-1]
        zone = zone_by_id[zone_id]

        sx, sy = (float(first["foot_point"][0]), float(first["foot_point"][1]))
        ex, ey = (float(last["foot_point"][0]), float(last["foot_point"][1]))
        dx = ex - sx
        dy = ey - sy
        dist = math.hypot(dx, dy)
        duration = float(last["timestamp"]) - float(first["timestamp"])
        mean_conf = sum(float(p["conf"]) for p in points) / len(points)
        angle = math.degrees(math.atan2(dy, dx)) if dist > 0 else 0.0

        rows.append(
            {
                "camera_id": first["camera_id"],
                "track_id": track_id,
                "zone_id": zone_id,
                "zone_kind": zone.kind,
                "class": first["class"],
                "enter_timestamp": round(float(first["timestamp"]), 3),
                "exit_timestamp": round(float(last["timestamp"]), 3),
                "duration_s": round(duration, 3),
                "enter_frame": int(first["frame"]),
                "exit_frame": int(last["frame"]),
                "start_x": round(sx, 1),
                "start_y": round(sy, 1),
                "end_x": round(ex, 1),
                "end_y": round(ey, 1),
                "dx": round(dx, 1),
                "dy": round(dy, 1),
                "distance_px": round(dist, 1),
                "speed_px_s": round(dist / duration, 2) if duration > 0 else 0.0,
                "direction_label": direction_label(dx, dy, min_distance=min_distance),
                "angle_deg": round(angle, 1),
                "num_points": len(points),
                "mean_conf": round(mean_conf, 4),
            }
        )

    return sorted(rows, key=lambda r: (r["enter_timestamp"], r["track_id"], r["zone_id"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zone icindeki track hareket ozetlerini uretir.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--zones", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-distance", type=float, default=25.0,
                        help="Bu pikselin altindaki hareket stationary sayilir")
    args = parser.parse_args()

    tracks_path = args.tracks or (TRACKS_DIR / f"tracks_{args.camera}.jsonl")
    zones_path = args.zones or (CONFIG_DIR / f"zones_{args.camera}.yaml")
    output_path = args.output or (ZONES_OUT_DIR / f"zone_tracks_{args.camera}.csv")

    if not tracks_path.is_absolute():
        tracks_path = PROJECT_ROOT / tracks_path
    if not zones_path.is_absolute():
        zones_path = PROJECT_ROOT / zones_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not tracks_path.is_file():
        raise SystemExit(f"HATA: tracks dosyasi yok: {tracks_path}")
    if not zones_path.is_file():
        raise SystemExit(f"HATA: zones dosyasi yok: {zones_path}")

    rows = summarize_zone_tracks(tracks_path, zones_path, min_distance=args.min_distance)
    write_csv(output_path, rows)

    by_dir = defaultdict(int)
    for row in rows:
        by_dir[row["direction_label"]] += 1

    print(f"Yazildi: {output_path}")
    print(f"Toplam zone-track: {len(rows)}")
    print("Yon dagilimi:")
    for label, count in sorted(by_dir.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()

"""Faz 2: goruntu yonunu fiziksel hareket etiketine cevirir.

Girdi:
    outputs/zones/zone_tracks_<camera>.csv
    configs/direction_mapping.yaml

Cikti:
    outputs/zones/zone_tracks_mapped_<camera>.csv

Eklenen kolonlar:
    mid_y          : track'in zone icindeki ortalama dikey konumu
    lane_group     : upper/lower
    movement_label : camA_to_camB / camB_to_camA / other

Bu adim Faz 3 icin kritik: artik sadece "left/right" degil,
"bu arac hangi kameradan hangi kameraya gidiyor olabilir?" bilgisini tasiriz.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONES_OUT_DIR = PROJECT_ROOT / "outputs" / "zones"
DEFAULT_MAPPING = PROJECT_ROOT / "configs" / "direction_mapping.yaml"


EXTRA_FIELDS = ["mid_y", "lane_group", "movement_label"]


def load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def lane_group_for(row: dict, lane_split_y: float) -> tuple[float, str]:
    start_y = float(row["start_y"])
    end_y = float(row["end_y"])
    mid_y = (start_y + end_y) / 2.0
    return mid_y, "upper" if mid_y < lane_split_y else "lower"


def map_row(row: dict, mapping: dict) -> dict:
    camera_id = row["camera_id"]
    zone_id = row["zone_id"]
    direction = row["direction_label"]

    camera_cfg = (mapping.get("cameras") or {}).get(camera_id)
    if not camera_cfg:
        raise KeyError(f"Mapping icinde camera yok: {camera_id}")

    lane_split_y = float(camera_cfg.get("lane_split_y", 540))
    mid_y, lane_group = lane_group_for(row, lane_split_y)

    zone_cfg = (camera_cfg.get("zones") or {}).get(zone_id)
    if not zone_cfg:
        raise KeyError(f"Mapping icinde zone yok: {camera_id}/{zone_id}")

    movement_label = (zone_cfg.get(lane_group) or {}).get(direction, "other")

    out = dict(row)
    out["mid_y"] = round(mid_y, 1)
    out["lane_group"] = lane_group
    out["movement_label"] = movement_label
    return out


def apply_mapping(input_path: Path, output_path: Path, mapping_path: Path) -> list[dict]:
    mapping = load_mapping(mapping_path)

    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [map_row(row, mapping) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Zone track yonlerini fiziksel hareket etiketine cevirir.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    args = parser.parse_args()

    input_path = args.input or (ZONES_OUT_DIR / f"zone_tracks_{args.camera}.csv")
    output_path = args.output or (ZONES_OUT_DIR / f"zone_tracks_mapped_{args.camera}.csv")
    mapping_path = args.mapping

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path

    if not input_path.is_file():
        raise SystemExit(f"HATA: input yok: {input_path}")
    if not mapping_path.is_file():
        raise SystemExit(f"HATA: mapping yok: {mapping_path}")

    rows = apply_mapping(input_path, output_path, mapping_path)
    by_lane = Counter(row["lane_group"] for row in rows)
    by_direction = Counter(row["direction_label"] for row in rows)
    by_movement = Counter(row["movement_label"] for row in rows)

    print(f"Yazildi: {output_path}")
    print(f"Toplam satir: {len(rows)}")
    print(f"Lane dagilimi: {dict(by_lane)}")
    print(f"Goruntu yon dagilimi: {dict(by_direction)}")
    print(f"Hareket etiketi dagilimi: {dict(by_movement)}")


if __name__ == "__main__":
    main()

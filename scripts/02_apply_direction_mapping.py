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


EXTRA_FIELDS = ["movement_label", "quality_ok"]


def load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def passes_quality(row: dict, quality: dict) -> bool:
    """Track kalite esiklerini geciyor mu? (gurultu/duran filtresi)"""
    dist = float(row["distance_px"])
    npts = int(row["num_points"])
    dur = float(row["duration_s"])
    return (
        dist >= float(quality.get("min_distance_px", 25))
        and npts >= int(quality.get("min_num_points", 5))
        and dur >= float(quality.get("min_duration_s", 0.3))
    )


def map_row(row: dict, mapping: dict) -> dict:
    """ZONE-TABANLI eslesme: aracin bulundugu ROI hareket yonunu belirler.

    - zone_id -> movement_label (config'teki zone_movement)
    - kalite esigi altindaki track'ler (duran/kisa) -> other
    """
    camera_id = row["camera_id"]
    zone_id = row["zone_id"]

    camera_cfg = (mapping.get("cameras") or {}).get(camera_id)
    if not camera_cfg:
        raise KeyError(f"Mapping icinde camera yok: {camera_id}")

    zone_movement = camera_cfg.get("zone_movement") or {}
    base_label = zone_movement.get(zone_id, "other")

    ok = passes_quality(row, mapping.get("quality") or {})
    movement_label = base_label if ok else "other"

    out = dict(row)
    out["movement_label"] = movement_label
    out["quality_ok"] = int(ok)
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
    by_movement = Counter(row["movement_label"] for row in rows)
    by_zone = Counter(row["zone_id"] for row in rows)
    n_quality_fail = sum(1 for row in rows if not int(row["quality_ok"]))

    print(f"Yazildi: {output_path}")
    print(f"Toplam satir: {len(rows)}")
    print(f"Zone dagilimi: {dict(by_zone)}")
    print(f"Kalite esigi altinda (other'a dusen): {n_quality_fail}")
    print(f"Hareket etiketi dagilimi: {dict(by_movement)}")


if __name__ == "__main__":
    main()

"""Faz 2: tracks JSONL kayitlarini zone event CSV'ye donusturur.

Girdi:
    outputs/tracks/tracks_<camera>.jsonl
    configs/zones_<camera>.yaml

Cikti:
    outputs/zones/zone_events_<camera>.csv

Mantik:
    Her track kaydinda foot_point hangi zone'un icinde diye bakariz.
    Bir track_id bir zone'a ilk kez girince ENTER, son kez cikinca EXIT olayi yazariz.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import load_zone_config, zones_for_point  # noqa: E402


TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
ZONES_OUT_DIR = PROJECT_ROOT / "outputs" / "zones"
CONFIG_DIR = PROJECT_ROOT / "configs"


EVENT_FIELDS = [
    "camera_id",
    "track_id",
    "zone_id",
    "zone_kind",
    "event",
    "timestamp",
    "frame",
    "foot_x",
    "foot_y",
    "class",
    "conf",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_event(rec: dict, zone, event: str) -> dict:
    foot_x, foot_y = rec["foot_point"]
    return {
        "camera_id": rec["camera_id"],
        "track_id": int(rec["track_id"]),
        "zone_id": zone.zone_id,
        "zone_kind": zone.kind,
        "event": event,
        "timestamp": round(float(rec["timestamp"]), 3),
        "frame": int(rec["frame"]),
        "foot_x": round(float(foot_x), 1),
        "foot_y": round(float(foot_y), 1),
        "class": rec["class"],
        "conf": round(float(rec["conf"]), 4),
    }


def assign_zone_events(tracks_path: Path, zones_path: Path) -> list[dict]:
    zone_config = load_zone_config(zones_path)

    # Her (track_id, zone_id) icin "su an iceride miydi?" durumunu tutar.
    active: dict[tuple[int, str], dict] = {}
    events: list[dict] = []
    seen_keys: set[tuple[int, str]] = set()

    for rec in iter_jsonl(tracks_path):
        track_id = int(rec["track_id"])
        foot = (float(rec["foot_point"][0]), float(rec["foot_point"][1]))
        inside_zones = zones_for_point(foot, zone_config.zones)
        inside_ids = {zone.zone_id for zone in inside_zones}

        # Yeni girisler.
        for zone in inside_zones:
            key = (track_id, zone.zone_id)
            seen_keys.add(key)
            if key not in active:
                event = make_event(rec, zone, "enter")
                events.append(event)
                active[key] = {"zone": zone, "last_rec": rec}
            else:
                active[key]["last_rec"] = rec

        # Cikislar: bu track icin daha once aktif olan ama bu karede icinde olmadigi zone'lar.
        active_keys_for_track = [key for key in active if key[0] == track_id]
        for key in active_keys_for_track:
            _, zone_id = key
            if zone_id not in inside_ids:
                state = active.pop(key)
                events.append(make_event(state["last_rec"], state["zone"], "exit"))

    # Video bittiginde hala zone icinde görünen track'ler icin kapanis.
    for state in active.values():
        events.append(make_event(state["last_rec"], state["zone"], "exit"))

    return sorted(events, key=lambda e: (e["timestamp"], e["frame"], e["track_id"], e["zone_id"], e["event"]))


def write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz 2 zone event CSV uretir.")
    parser.add_argument("--camera", required=True, help="or. camA, camB")
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--zones", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tracks_path = args.tracks or (TRACKS_DIR / f"tracks_{args.camera}.jsonl")
    zones_path = args.zones or (CONFIG_DIR / f"zones_{args.camera}.yaml")
    output_path = args.output or (ZONES_OUT_DIR / f"zone_events_{args.camera}.csv")

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

    events = assign_zone_events(tracks_path, zones_path)
    write_events(output_path, events)

    by_event = defaultdict(int)
    by_zone = defaultdict(int)
    for event in events:
        by_event[event["event"]] += 1
        by_zone[event["zone_id"]] += 1

    print(f"Yazildi: {output_path}")
    print(f"Toplam event: {len(events)} | enter={by_event['enter']} exit={by_event['exit']}")
    for zone_id, count in sorted(by_zone.items()):
        print(f"  {zone_id}: {count} event")


if __name__ == "__main__":
    main()

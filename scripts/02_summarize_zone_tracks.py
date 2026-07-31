"""Faz 2 ziyaret ozeti: zone_events + tracks -> zone_tracks_<cam>.csv.

Her (track_id, zone_id, ziyaret) icin tek satirlik ozet uretir: kalis suresi,
net yer degistirme, goruntu yonu, baskin serit, sinif ve ortalama guven.

Kapsam tek kameradir. Uretilen yon etiketi yalnizca GORUNTU yonudur
(left/right/up/down/stationary); fiziksel yon etiketi bu script'in isi degildir.

Ornek:
    python scripts/02_summarize_zone_tracks.py --camera camA
    python scripts/02_summarize_zone_tracks.py --camera all
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import (  # noqa: E402
    assign_lane,
    find_zone,
    image_direction,
    load_zone_config,
    load_zone_params,
)

DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
CONFIG_DIR = PROJECT_ROOT / "configs"

CSV_COLUMNS = (
    "camera_id", "zone_id", "track_id", "visit_index",
    "t_enter", "t_exit", "frame_enter", "frame_exit",
    "dwell_s", "n_frames_inside",
    "start_foot_x", "start_foot_y", "end_foot_x", "end_foot_y",
    "dx", "dy", "distance_px", "speed_px_s",
    "direction_label", "lane", "class_mode", "conf_mean", "exit_reason",
)

FLOAT_DECIMALS = 3
LANE_UNKNOWN = "unknown"
TRACK_END_WARN_RATIO = 0.20      # track_end orani bu esigi asarsa UYARI
STATIONARY_WARN_RATIO = 0.40     # stationary orani bu esigi asarsa UYARI

# Kayit demeti alan sirasi
IDX_FRAME, IDX_TS, IDX_FX, IDX_FY, IDX_CLASS, IDX_CONF, IDX_INSIDE = range(7)


def round3(value: float) -> float:
    """Float kolonlari sabit ondalikla yuvarlar (tekrarlanabilirlik icin)."""
    return round(float(value), FLOAT_DECIMALS)


def percentile(values, q):
    """Sirali listeden dogrusal ara degerlemeyle yuzdelik dondurur; bosta None."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values, label, unit=""):
    """min/p05/p50/p95/max satiri uretir."""
    if not values:
        return f"  {label:<14} veri yok"
    return (f"  {label:<14} min {min(values):>8.2f} | p05 {percentile(values, 0.05):>8.2f} | "
            f"p50 {percentile(values, 0.50):>8.2f} | p95 {percentile(values, 0.95):>8.2f} | "
            f"max {max(values):>8.2f}{unit}")


def mode_of(values, tie_value=None):
    """En sik gecen degeri dondurur (baskin deger).

    Beraberlik yalnizca EN YUKSEK sayiya sahip degerler arasinda cozulur:
    tie_value verilmisse o dondurulur (lane icin "unknown"), verilmemisse esit
    sayidaki degerler arasindan alfabetik olarak ilki secilir (deterministik).
    """
    if not values:
        return tie_value
    counts = Counter(values)
    best = max(counts.values())
    tied = sorted(key for key, count in counts.items() if count == best)
    if len(tied) > 1 and tie_value is not None:
        return tie_value
    return tied[0]


def read_events(events_path: Path):
    """zone_events CSV'sini okur ve (track_id, zone_id) bazinda ziyaretlere esler.

    Her enter, kendisinden sonraki ilk exit ile eslesir; her cift bir ziyarettir
    ve visit_index her (track_id, zone_id) icin 0'dan baslar.

    donus: (ziyaret listesi, hata listesi)
    """
    visits = []
    errors = []
    grouped: dict[tuple[int, str], list[dict]] = {}

    with events_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("track_id", "zone_id", "event", "frame", "timestamp",
                               "exit_reason") if c not in (reader.fieldnames or [])]
        if missing:
            return [], [f"{events_path}: beklenen kolonlar eksik: {missing}"]
        for row in reader:
            key = (int(row["track_id"]), row["zone_id"])
            grouped.setdefault(key, []).append(row)

    for (track_id, zone_id), rows in grouped.items():
        rows.sort(key=lambda r: (int(r["frame"]), r["event"]))
        pending = None
        visit_index = 0
        for row in rows:
            if row["event"] == "enter":
                if pending is not None:
                    errors.append(f"track {track_id} / {zone_id}: kapanmamis enter "
                                  f"(frame {pending['frame']}) uzerine yeni enter geldi.")
                pending = row
            elif row["event"] == "exit":
                if pending is None:
                    errors.append(f"track {track_id} / {zone_id}: enter'i olmayan exit "
                                  f"(frame {row['frame']}).")
                    continue
                visits.append({
                    "track_id": track_id,
                    "zone_id": zone_id,
                    "visit_index": visit_index,
                    "frame_enter": int(pending["frame"]),
                    "frame_exit": int(row["frame"]),
                    "t_enter": float(pending["timestamp"]),
                    "t_exit": float(row["timestamp"]),
                    "exit_reason": row["exit_reason"],
                })
                visit_index += 1
                pending = None
        if pending is not None:
            errors.append(f"track {track_id} / {zone_id}: exit'i olmayan enter "
                          f"(frame {pending['frame']}).")
    return visits, errors


def read_tracks(jsonl_path: Path, config, params):
    """JSONL'i filtreleyip track_id bazinda biriktirir; ROI icinde olmayi isaretler.

    Filtre P2.4 ile ayni: conf < min_conf veya sinif allowed_classes disinda olan
    kayitlar atilir. Her kayit icin hangi bolgede oldugu find_zone ile bulunur.
    """
    min_conf = params["min_conf"]
    allowed = set(params["allowed_classes"])
    tracks: dict[int, list[tuple]] = {}
    counts = {"read": 0, "dropped": 0, "kept": 0}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            counts["read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["dropped"] += 1
                continue

            foot = rec.get("foot_point")
            if (rec.get("conf") is None or rec.get("class") is None
                    or rec.get("frame") is None or rec.get("track_id") is None
                    or not (isinstance(foot, list) and len(foot) == 2)):
                counts["dropped"] += 1
                continue
            if rec["conf"] < min_conf or rec["class"] not in allowed:
                counts["dropped"] += 1
                continue

            zone_id = find_zone((foot[0], foot[1]), config)
            tracks.setdefault(int(rec["track_id"]), []).append(
                (int(rec["frame"]), rec.get("timestamp"), foot[0], foot[1],
                 rec["class"], rec["conf"], zone_id)
            )
            counts["kept"] += 1

    for records in tracks.values():
        records.sort(key=lambda r: r[IDX_FRAME])
    return tracks, counts


def summarize_visit(camera_id, visit, records, zone, params):
    """Tek ziyaret icin ozet satirini uretir; ROI icinde kayit yoksa None.

    Tum ozet alanlari (baskin serit, sinif, ortalama guven dahil) ziyaret
    araligindaki ROI ICINDEKI kayitlardan hesaplanir; aralikta olup ROI disinda
    kalan kareler ozete girmez.
    """
    inside = [r for r in records
              if visit["frame_enter"] <= r[IDX_FRAME] <= visit["frame_exit"]
              and r[IDX_INSIDE] == zone.zone_id]
    if not inside:
        return None

    start, end = inside[0], inside[-1]
    dx = end[IDX_FX] - start[IDX_FX]
    dy = end[IDX_FY] - start[IDX_FY]
    distance = math.hypot(dx, dy)
    dwell = visit["t_exit"] - visit["t_enter"]
    speed = distance / dwell if dwell > 0 else 0.0

    lanes = [assign_lane((r[IDX_FX], r[IDX_FY]), zone) for r in inside]
    classes = [r[IDX_CLASS] for r in inside]
    confs = [r[IDX_CONF] for r in inside]

    return {
        "camera_id": camera_id,
        "zone_id": zone.zone_id,
        "track_id": visit["track_id"],
        "visit_index": visit["visit_index"],
        "t_enter": round3(visit["t_enter"]),
        "t_exit": round3(visit["t_exit"]),
        "frame_enter": visit["frame_enter"],
        "frame_exit": visit["frame_exit"],
        "dwell_s": round3(dwell),
        "n_frames_inside": len(inside),
        "start_foot_x": round3(start[IDX_FX]),
        "start_foot_y": round3(start[IDX_FY]),
        "end_foot_x": round3(end[IDX_FX]),
        "end_foot_y": round3(end[IDX_FY]),
        "dx": round3(dx),
        "dy": round3(dy),
        "distance_px": round3(distance),
        "speed_px_s": round3(speed),
        "direction_label": image_direction(dx, dy, params["direction_min_distance_px"]),
        "lane": mode_of(lanes, tie_value=LANE_UNKNOWN),
        "class_mode": mode_of(classes),
        "conf_mean": round3(sum(confs) / len(confs)),
        "exit_reason": visit["exit_reason"],
        "_dwell_raw": dwell,
    }


def print_summary(rows, dropped_short, missing_inside, params):
    """Saglik kontrollerini ve dagilimlari ekrana yazar."""
    print(f"\nToplam ziyaret       : {len(rows) + dropped_short + missing_inside}")
    print(f"Yazilan ziyaret      : {len(rows)}")
    print(f"Atilan kisa ziyaret  : {dropped_short} "
          f"(dwell_s < {params['min_dwell_s']})")
    if missing_inside:
        print(f"UYARI: {missing_inside} ziyarette aralikta ROI icinde kayit bulunamadi; "
              "ozet uretilmedi.")
    if not rows:
        print("Yazilacak ziyaret kalmadi; dagilim uretilmedi.")
        return

    def distribution(field):
        counts = Counter(row[field] for row in rows)
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    print("\ndirection_label dagilimi:")
    stationary = 0
    for label, count in distribution("direction_label"):
        print(f"  {label:<12} {count:>6}  (%{100 * count / len(rows):.1f})")
        if label == "stationary":
            stationary = count

    print("\nlane dagilimi:")
    for label, count in distribution("lane"):
        print(f"  {label:<12} {count:>6}  (%{100 * count / len(rows):.1f})")

    print("\nclass_mode dagilimi:")
    for label, count in distribution("class_mode"):
        print(f"  {label:<12} {count:>6}  (%{100 * count / len(rows):.1f})")

    print("\nexit_reason dagilimi:")
    track_end = 0
    for label, count in distribution("exit_reason"):
        print(f"  {label:<12} {count:>6}  (%{100 * count / len(rows):.1f})")
        if label == "track_end":
            track_end = count

    print("\nIstatistikler:")
    print(describe([row["dwell_s"] for row in rows], "dwell_s", " sn"))
    print(describe([row["distance_px"] for row in rows], "distance_px", " px"))
    print(describe([row["speed_px_s"] for row in rows], "speed_px_s", " px/sn"))
    print(describe([float(row["n_frames_inside"]) for row in rows], "n_frames_inside"))

    track_end_ratio = track_end / len(rows)
    if track_end_ratio > TRACK_END_WARN_RATIO:
        print(f"\nUYARI: ziyaretlerin %{100 * track_end_ratio:.1f}'i track_end ile bitiyor "
              f"(esik %{100 * TRACK_END_WARN_RATIO:.0f}). ROI muhtemelen goruntu kenarina "
              "cok yakin; araclar bolgeden cikmadan takip bitiyor.")
    stationary_ratio = stationary / len(rows)
    if stationary_ratio > STATIONARY_WARN_RATIO:
        print(f"UYARI: ziyaretlerin %{100 * stationary_ratio:.1f}'i stationary "
              f"(esik %{100 * STATIONARY_WARN_RATIO:.0f}). ROI cok ince olabilir veya "
              f"direction_min_distance_px ({params['direction_min_distance_px']} px) "
              "bu bolge icin fazla buyuk.")


def process_camera(camera_id, events_path, tracks_path, zones_path, params_path,
                   out_path) -> bool:
    """Tek kamera icin ziyaret ozetini uretir ve CSV'ye yazar."""
    print(f"\n=== {camera_id} ===")
    print(f"Olaylar     : {events_path}")
    print(f"Takip       : {tracks_path}")

    for path, label in ((events_path, "olay CSV'si"), (tracks_path, "takip dosyasi")):
        if not path.is_file():
            print(f"HATA: {label} bulunamadi: {path}")
            return False
    try:
        config = load_zone_config(zones_path)
        params = load_zone_params(params_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}")
        return False

    zones_by_id = {zone.zone_id: zone for zone in config.zones}
    visits, errors = read_events(events_path)
    if errors:
        print("HATA: olay dosyasinda eslesmeyen kayitlar var:")
        for error in errors:
            print(f"  - {error}")
        return False
    if not visits:
        print("HATA: olay dosyasinda eslesen ziyaret bulunamadi.")
        return False

    tracks, counts = read_tracks(tracks_path, config, params)
    print(f"Kayit       : okunan {counts['read']}, filtrelenen {counts['dropped']}, "
          f"kalan {counts['kept']}")

    rows = []
    dropped_short = 0
    missing_inside = 0
    for visit in visits:
        zone = zones_by_id.get(visit["zone_id"])
        if zone is None:
            print(f"HATA: '{visit['zone_id']}' bolgesi {zones_path} icinde tanimli degil.")
            return False
        summary = summarize_visit(camera_id, visit, tracks.get(visit["track_id"], []),
                                  zone, params)
        if summary is None:
            missing_inside += 1
            continue
        if summary["_dwell_raw"] < params["min_dwell_s"]:
            dropped_short += 1
            continue
        del summary["_dwell_raw"]
        rows.append(summary)

    rows.sort(key=lambda r: (r["track_id"], r["visit_index"], r["zone_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows, dropped_short, missing_inside, params)
    print(f"\nYazildi     : {out_path} ({len(rows)} satir)")
    return True


def load_camera_ids(config_path: Path):
    """cameras.yaml icindeki camera_id listesini dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cameras = [c.get("camera_id") for c in (data.get("cameras") or [])
               if isinstance(c, dict) and c.get("camera_id")]
    if not cameras:
        raise SystemExit(f"HATA: cameras.yaml icinde kamera tanimi yok: {config_path}")
    return cameras


def resolve(path, default):
    """Verilen yolu PROJECT_ROOT'a gore cozer; yoksa varsayilani dondurur."""
    if path is None:
        return default
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 2: bolge ziyaretlerini tek satirlik ozetlere indirger."
    )
    parser.add_argument("--camera", required=True,
                        help="kamera kimligi; tum kameralar icin 'all'")
    parser.add_argument("--events", type=Path, default=None,
                        help="zone_events_<cam>.csv yolu")
    parser.add_argument("--tracks", type=Path, default=None,
                        help="tracks_<cam>.jsonl yolu")
    parser.add_argument("--zones", type=Path, default=None,
                        help="zones_<cam>.yaml yolu")
    parser.add_argument("--params", type=Path, default=None,
                        help="zone_params.yaml yolu")
    parser.add_argument("--out", type=Path, default=None,
                        help="cikti CSV yolu (varsayilan outputs/zones/zone_tracks_<cam>.csv)")
    args = parser.parse_args()

    if args.camera == "all":
        cameras = load_camera_ids(DEFAULT_CAMERAS_CONFIG)
        if any(v is not None for v in (args.events, args.tracks, args.zones, args.out)):
            raise SystemExit("HATA: --camera all ile dosya yollari birlikte kullanilamaz; "
                             "yollar kamera basina turetilir.")
    else:
        cameras = [args.camera]

    params_path = resolve(args.params, CONFIG_DIR / "zone_params.yaml")

    results = []
    for camera_id in cameras:
        results.append(process_camera(
            camera_id,
            resolve(args.events, ZONES_DIR / f"zone_events_{camera_id}.csv"),
            resolve(args.tracks, TRACKS_DIR / f"tracks_{camera_id}.jsonl"),
            resolve(args.zones, CONFIG_DIR / f"zones_{camera_id}.yaml"),
            params_path,
            resolve(args.out, ZONES_DIR / f"zone_tracks_{camera_id}.csv"),
        ))

    ok = sum(1 for r in results if r)
    print(f"\nTamamlanan kamera: {ok} / {len(results)}")
    if ok != len(results):
        raise SystemExit("HATA: en az bir kamera islenemedi (yukaridaki mesajlara bakin).")


if __name__ == "__main__":
    main()

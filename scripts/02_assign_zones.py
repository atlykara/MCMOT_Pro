"""Faz 2 bolge olaylari: tracks_<cam>.jsonl -> zone_events_<cam>.csv.

Her track'in her bolgeye giris/cikisini histerezisli bir durum makinesiyle
uretir. Sahte olaylar sonradan temizlenmez; debounce uretim aninda uygulanir.

Kapsam tek kameradir: kameralar arasi hicbir islem yapilmaz. Script yalnizca
JSONL ve YAML okur; video acmaz, model yuklemez, girdi dosyalarini degistirmez.

Ornek:
    python scripts/02_assign_zones.py --camera camA
    python scripts/02_assign_zones.py --camera all
    python scripts/02_assign_zones.py --camera camB --max-frames 3000
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import (  # noqa: E402
    assign_lane,
    find_zone,
    load_zone_config,
    load_zone_params,
)

DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
CONFIG_DIR = PROJECT_ROOT / "configs"

CSV_COLUMNS = (
    "camera_id", "zone_id", "track_id", "event", "frame", "timestamp",
    "foot_x", "foot_y", "lane", "class", "conf", "exit_reason",
)

EVENT_ENTER = "enter"
EVENT_EXIT = "exit"
REASON_LEFT_ZONE = "left_zone"
REASON_TRACK_END = "track_end"

# Kayit demeti alan sirasi: (frame, timestamp, foot_x, foot_y, class, conf, zone_id)
IDX_FRAME, IDX_TS, IDX_FX, IDX_FY, IDX_CLASS, IDX_CONF, IDX_ZONE = range(7)


def fmt_number(value) -> str:
    """Sayiyi kaynaktaki degeri koruyarak deterministik metne cevirir."""
    return repr(float(value))


def read_tracks(jsonl_path: Path, config, params, max_frames):
    """JSONL'i satir satir okur, filtreler ve track_id bazli biriktirir.

    Tum dosya belleğe alinmaz; kayit basina yalnizca gereken alanlar kucuk bir
    demet olarak saklanir. Bolge atamasi okuma sirasinda find_zone ile yapilir.

    donus: (tracks sozlugu, sayac sozlugu)
    """
    min_conf = params["min_conf"]
    allowed = set(params["allowed_classes"])

    tracks: dict[int, list[tuple]] = {}
    counts = {"read": 0, "bad_line": 0, "over_max_frames": 0,
              "drop_conf": 0, "drop_class": 0, "drop_field": 0, "kept": 0}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            counts["read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_line"] += 1
                continue

            frame = rec.get("frame")
            track_id = rec.get("track_id")
            conf = rec.get("conf")
            cls = rec.get("class")
            foot = rec.get("foot_point")
            timestamp = rec.get("timestamp")

            if (frame is None or track_id is None or conf is None or cls is None
                    or timestamp is None
                    or not (isinstance(foot, list) and len(foot) == 2)):
                counts["drop_field"] += 1
                continue

            if max_frames is not None and frame >= max_frames:
                counts["over_max_frames"] += 1
                continue
            if conf < min_conf:
                counts["drop_conf"] += 1
                continue
            if cls not in allowed:
                counts["drop_class"] += 1
                continue

            zone_id = find_zone((foot[0], foot[1]), config)
            tracks.setdefault(int(track_id), []).append(
                (int(frame), timestamp, foot[0], foot[1], cls, conf, zone_id)
            )
            counts["kept"] += 1

    return tracks, counts


def build_event(camera_id, zone, record, event, exit_reason) -> dict:
    """Bir kayittan CSV satiri sozlugu uretir (lane olay anindaki noktadan)."""
    point = (record[IDX_FX], record[IDX_FY])
    return {
        "camera_id": camera_id,
        "zone_id": zone.zone_id,
        "event": event,
        "frame": record[IDX_FRAME],
        "timestamp": fmt_number(record[IDX_TS]),
        "foot_x": fmt_number(record[IDX_FX]),
        "foot_y": fmt_number(record[IDX_FY]),
        "lane": assign_lane(point, zone),
        "class": record[IDX_CLASS],
        "conf": fmt_number(record[IDX_CONF]),
        "exit_reason": exit_reason,
    }


def process_track(camera_id, track_id, records, config, params):
    """Tek track icin tum bolgelerde histerezisli giris/cikis olaylarini uretir.

    Durum makinesi (track_id, zone_id) ciftine ozeldir:
      OUTSIDE: art arda min_inside_frames KAYIT iceride gorulunce ENTER yazilir;
               olayin frame/timestamp'i bu serinin ILK kaydidir.
      INSIDE : art arda min_outside_frames KAYIT disarida gorulunce EXIT yazilir;
               olayin frame/timestamp'i iceride gorulen SON kayittir.
    Track'in hic tespit edilmedigi kareler kayit uretmedigi icin dizide yoktur;
    dolayisiyla "disarida gorulen kare" olarak SAYILMAZ, dogal olarak atlanir.
    Veri biterken durum hala INSIDE ise son iceri kaydinda track_end cikisi yazilir.

    donus: (olay listesi, naif olay sayisi)
    """
    min_inside = params["min_inside_frames"]
    min_outside = params["min_outside_frames"]

    events = []
    naive_events = 0

    for zone in config.zones:
        inside_state = False
        inside_run = 0
        outside_run = 0
        run_first = None       # icerideki serinin ilk kaydi
        last_inside = None     # iceride gorulen son kayit
        naive_prev_inside = False

        for record in records:
            is_inside = record[IDX_ZONE] == zone.zone_id

            # histerezis olmasaydi kac olay olurdu (karsilastirma icin)
            if is_inside != naive_prev_inside:
                naive_events += 1
                naive_prev_inside = is_inside

            if not inside_state:
                if is_inside:
                    if inside_run == 0:
                        run_first = record
                    inside_run += 1
                    if inside_run >= min_inside:
                        events.append(build_event(camera_id, zone, run_first,
                                                  EVENT_ENTER, ""))
                        inside_state = True
                        last_inside = record
                        outside_run = 0
                        inside_run = 0
                        run_first = None
                else:
                    inside_run = 0
                    run_first = None
            else:
                if is_inside:
                    last_inside = record
                    outside_run = 0
                else:
                    outside_run += 1
                    if outside_run >= min_outside:
                        events.append(build_event(camera_id, zone, last_inside,
                                                  EVENT_EXIT, REASON_LEFT_ZONE))
                        inside_state = False
                        outside_run = 0
                        inside_run = 0
                        run_first = None

        if inside_state and last_inside is not None:
            events.append(build_event(camera_id, zone, last_inside,
                                      EVENT_EXIT, REASON_TRACK_END))
        if naive_prev_inside:
            naive_events += 1  # naif sayimda da acik kalan giris kapatilir

    return events, naive_events


def write_events(out_path: Path, rows) -> None:
    """Olaylari CSV'ye yazar; satir sonu ve alan sirasi sabittir."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process_camera(camera_id, tracks_path, zones_path, params_path, out_path,
                   max_frames) -> bool:
    """Tek kamera icin olaylari uretir ve CSV'ye yazar."""
    print(f"\n=== {camera_id} ===")
    print(f"Takip dosyasi : {tracks_path}")
    print(f"Bolge config  : {zones_path}")
    print(f"Parametreler  : {params_path}")

    if not tracks_path.is_file():
        print(f"HATA: takip dosyasi bulunamadi: {tracks_path}")
        return False
    try:
        config = load_zone_config(zones_path)
        params = load_zone_params(params_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}")
        return False

    print(f"Esikler       : min_conf={params['min_conf']}, "
          f"allowed_classes={params['allowed_classes']}, "
          f"min_inside_frames={params['min_inside_frames']}, "
          f"min_outside_frames={params['min_outside_frames']}")
    print(f"Bolgeler      : {[z.zone_id for z in config.zones]}")
    if max_frames is not None:
        print(f"Sinir         : yalnizca frame < {max_frames} isleniyor")

    tracks, counts = read_tracks(tracks_path, config, params, max_frames)

    all_events = []
    naive_total = 0
    tracks_without_zone = 0
    for track_id in sorted(tracks):
        records = sorted(tracks[track_id], key=lambda r: r[IDX_FRAME])
        if all(r[IDX_ZONE] is None for r in records):
            tracks_without_zone += 1
        events, naive_events = process_track(camera_id, track_id, records,
                                             config, params)
        for event in events:
            event["track_id"] = track_id
        all_events.extend(events)
        naive_total += naive_events

    # tekrarlanabilirlik: (track_id, frame, zone_id, event) sirasi
    all_events.sort(key=lambda e: (e["track_id"], e["frame"], e["zone_id"], e["event"]))

    enter_count = sum(1 for e in all_events if e["event"] == EVENT_ENTER)
    exit_count = sum(1 for e in all_events if e["event"] == EVENT_EXIT)
    if enter_count != exit_count:
        print(f"HATA: enter ({enter_count}) ve exit ({exit_count}) sayilari esit degil; "
              "CSV yazilmadi.")
        return False

    write_events(out_path, all_events)

    # --- ozet ---
    dropped = (counts["drop_conf"] + counts["drop_class"] + counts["drop_field"]
               + counts["bad_line"] + counts["over_max_frames"])
    print(f"\nOkunan satir  : {counts['read']}")
    print(f"Filtrelenen   : {dropped} "
          f"(conf<{params['min_conf']}: {counts['drop_conf']}, "
          f"sinif disi: {counts['drop_class']}, "
          f"alan eksik: {counts['drop_field']}, "
          f"bozuk satir: {counts['bad_line']}, "
          f"frame sinirinin disinda: {counts['over_max_frames']})")
    print(f"Kalan kayit   : {counts['kept']}")
    print(f"Benzersiz track: {len(tracks)}")
    print(f"Hicbir bolgeye girmeyen track: {tracks_without_zone} / {len(tracks)}")

    print("\nBolge basina olay:")
    print("  zone_id                        enter    exit  (track_end)")
    for zone in config.zones:
        z_enter = sum(1 for e in all_events
                      if e["zone_id"] == zone.zone_id and e["event"] == EVENT_ENTER)
        z_exit = sum(1 for e in all_events
                     if e["zone_id"] == zone.zone_id and e["event"] == EVENT_EXIT)
        z_track_end = sum(1 for e in all_events
                          if e["zone_id"] == zone.zone_id
                          and e["exit_reason"] == REASON_TRACK_END)
        print(f"  {zone.zone_id:<28} {z_enter:>6} {z_exit:>7}  {z_track_end:>10}")

    actual_total = len(all_events)
    suppressed = naive_total - actual_total
    print(f"\nDebounce etkisi:")
    print(f"  histerezis olmasaydi olay sayisi : {naive_total}")
    print(f"  histerezisle uretilen olay sayisi: {actual_total}")
    print(f"  ELENEN SAHTE GECIS               : {suppressed}"
          + (f" (%{100 * suppressed / naive_total:.1f})" if naive_total else ""))
    print(f"\nYazildi       : {out_path} ({enter_count} enter / {exit_count} exit)")
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
    """Verilen yolu (varsa) PROJECT_ROOT'a gore cozer, yoksa varsayilani dondurur."""
    if path is None:
        return default
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 2: tek kamera bolge giris/cikis olaylarini uretir (histerezisli)."
    )
    parser.add_argument("--camera", required=True,
                        help="kamera kimligi; tum kameralar icin 'all'")
    parser.add_argument("--tracks", type=Path, default=None,
                        help="tracks_<cam>.jsonl yolu (varsayilan outputs/tracks/)")
    parser.add_argument("--zones", type=Path, default=None,
                        help="zones_<cam>.yaml yolu (varsayilan configs/)")
    parser.add_argument("--params", type=Path, default=None,
                        help="zone_params.yaml yolu (varsayilan configs/zone_params.yaml)")
    parser.add_argument("--out", type=Path, default=None,
                        help="cikti CSV yolu (varsayilan outputs/zones/zone_events_<cam>.csv)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="yalnizca frame < N olan kayitlari isle (hizli deneme)")
    args = parser.parse_args()

    if args.camera == "all":
        cameras = load_camera_ids(DEFAULT_CAMERAS_CONFIG)
        if any(v is not None for v in (args.tracks, args.zones, args.out)):
            raise SystemExit("HATA: --camera all ile --tracks/--zones/--out birlikte "
                             "kullanilamaz; yollar kamera basina turetilir.")
    else:
        cameras = [args.camera]

    params_path = resolve(args.params, CONFIG_DIR / "zone_params.yaml")

    results = []
    for camera_id in cameras:
        tracks_path = resolve(args.tracks, TRACKS_DIR / f"tracks_{camera_id}.jsonl")
        zones_path = resolve(args.zones, CONFIG_DIR / f"zones_{camera_id}.yaml")
        out_path = resolve(args.out, ZONES_DIR / f"zone_events_{camera_id}.csv")
        results.append(process_camera(camera_id, tracks_path, zones_path,
                                      params_path, out_path, args.max_frames))

    ok = sum(1 for r in results if r)
    print(f"\nTamamlanan kamera: {ok} / {len(results)}")
    if ok != len(results):
        raise SystemExit("HATA: en az bir kamera islenemedi (yukaridaki mesajlara bakin).")


if __name__ == "__main__":
    main()

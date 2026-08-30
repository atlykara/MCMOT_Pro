"""Iki kamera videosunu FIFO eslestirme konsoluyla birlikte oynatir/yazar.

Offline videolar kamera soketi gibi davranir: her ROI exit/entry olayi zaman
sirasiyla FifoMatcher'a gonderilir. Ayni cekirdek daha sonra gercek kamera veya
mesaj kuyrugu ureticisine baglanabilir.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.fifo_matcher import DIRECTIONS, FifoMatcher, TrackEvent  # noqa: E402
from mcmot.zones import load_zone_config, zones_for_point  # noqa: E402


PANEL_WIDTH = 640
VIEW_WIDTH = 640
VIEW_HEIGHT = 360
OUTPUT_SIZE = (VIEW_WIDTH + PANEL_WIDTH, VIEW_HEIGHT * 2)
ZONE_COLOR = (0, 220, 255)
ACTIVE_COLOR = (80, 230, 120)
MATCH_COLOR = (255, 130, 255)
UNMATCHED_COLOR = (150, 155, 160)   # eslesmemis: notr gri, one cikmasin

# Her match_id'ye kendine ozgu bir renk. Ayni match_id iki kamerada da AYNI
# rengi alir; boylece bir arac camA'da hangi renkle cerceveleniyorsa camB'de de
# ayni renkle cercevelenir ve es oldugu gozle dogrudan gorulur.
_COLOR_CACHE = {}


def match_color(match_id):
    """match_id -> belirgin, tekrar etmeyen BGR renk (altin oran ile hue dagitimi).

    Altin oran adimi ardisik match_id'lerin renk tonunu birbirinden mumkun
    oldugunca uzaga tasir; boylece ayni karede bulunan iki eslesme karismaz.
    """
    if match_id in _COLOR_CACHE:
        return _COLOR_CACHE[match_id]
    digits = "".join(ch for ch in str(match_id) if ch.isdigit())
    index = int(digits) if digits else abs(hash(match_id))
    hue = int((index * 137.508) % 180)          # OpenCV HSV'de hue 0-179
    sat = 235 if index % 2 == 0 else 200        # doygunlukta hafif varyasyon
    val = 255 if index % 3 else 225
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    _COLOR_CACHE[match_id] = color
    return color


def load_confirmed_matches():
    """Kalibre edilmis boru hattinin urettigi eslesmeleri okur.

    Doner: {(kamera, track_id): (match_id, guven)}. Hem kaynak hem hedef track
    ayni match_id'ye baglanir; renklendirme bu esleme uzerinden yapilir.
    """
    result = {}
    match_dir = PROJECT_ROOT / "outputs" / "matching"
    for name in ("matches.csv", "matches_ambiguous.csv"):
        path = match_dir / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mid = row["match_id"]
                conf = row.get("confidence", "")
                result[(row["src_camera"], int(row["src_track"]))] = (mid, conf)
                result[(row["dst_camera"], int(row["dst_track"]))] = (mid, conf)
    return result


def load_yaml(path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def camera_paths():
    rows = load_yaml(PROJECT_ROOT / "configs" / "cameras.yaml").get("cameras") or []
    result = {}
    for row in rows:
        path = Path(row["video_path"])
        result[row["camera_id"]] = path if path.is_absolute() else PROJECT_ROOT / path
    return result


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def records_by_frame(camera):
    tracks_path = PROJECT_ROOT / "outputs" / "tracks" / f"tracks_{camera}.jsonl"
    zones_path = PROJECT_ROOT / "configs" / f"zones_{camera}.yaml"
    zone_cfg = load_zone_config(zones_path)
    result = defaultdict(list)
    for record in iter_jsonl(tracks_path):
        foot = (float(record["foot_point"][0]), float(record["foot_point"][1]))
        zones = zones_for_point(foot, zone_cfg.zones)
        if zones:
            item = dict(record)
            item["zone_ids"] = [zone.zone_id for zone in zones]
            result[int(item["frame"])].append(item)
    return result, zone_cfg


def is_source(camera, direction):
    return ((direction == "camA_to_camB" and camera == "camA")
            or (direction == "camB_to_camA" and camera == "camB"))


def load_events(duration_s, clock_offset_s):
    events = []
    sequence = 0
    zones_dir = PROJECT_ROOT / "outputs" / "zones"
    for camera in ("camA", "camB"):
        path = zones_dir / f"zone_tracks_mapped_{camera}.csv"
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                direction = row["movement_label"]
                if direction not in DIRECTIONS:
                    continue
                source = is_source(camera, direction)
                raw_time = float(row["exit_timestamp"] if source else row["enter_timestamp"])
                timestamp = raw_time + (clock_offset_s if camera == "camB" else 0.0)
                if duration_s is not None and timestamp > duration_s:
                    continue
                events.append((
                    timestamp,
                    0 if source else 1,
                    sequence,
                    TrackEvent(
                        timestamp=timestamp,
                        camera_id=camera,
                        track_id=int(row["track_id"]),
                        vehicle_class=row["class"],
                        movement_label=direction,
                        event_type="exit" if source else "entry",
                    ),
                ))
                sequence += 1
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    return events


def draw_zones(frame, zone_cfg):
    for zone in zone_cfg.zones:
        points = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(frame, [points], True, ZONE_COLOR, 2)
        x, y = points[0]
        cv2.putText(frame, zone.zone_id, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ZONE_COLOR, 2, cv2.LINE_AA)


def annotate_frame(frame, camera, records, matched_ids, timestamp):
    """Eslesmis araci kendi match rengiyle, eslesmemisi notr griyle cerceveler.

    Ayni match_id iki kamerada da ayni rengi aldigi icin, camA'daki bir aracin
    camB'deki esi renginden taninir. Eslesmis kutular daha kalin cizilir ve
    etiketi zemin uzerine yazilir ki uzak/kucuk araclarda da okunabilsin.
    """
    for record in records:
        key = (camera, int(record["track_id"]))
        entry = matched_ids.get(key)
        match_id = entry[0] if isinstance(entry, tuple) else entry
        x1, y1, x2, y2 = (int(value) for value in record["bbox_xyxy"])
        if match_id:
            color = match_color(match_id)
            thickness = 3
            label = str(match_id)
        else:
            color = UNMATCHED_COLOR
            thickness = 1
            label = f"#{record['track_id']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        scale = 0.5 if match_id else 0.42
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(th + 4, y1 - 4)
        if match_id:
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (15, 20, 28), -1)
    cv2.putText(frame, f"{camera}  t={timestamp:05.2f}s  ROI active={len(records)}",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2,
                cv2.LINE_AA)


def fit_view(frame):
    return cv2.resize(frame, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA)


def text(panel, value, x, y, color=(220, 225, 232), scale=0.52, thickness=1):
    cv2.putText(panel, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def active_lines(records):
    unique = {}
    for record in records:
        unique[int(record["track_id"])] = record
    return [
        f"#{tid:<4} {row['class']:<5} {','.join(row.get('zone_ids', []))}"
        for tid, row in sorted(unique.items())
    ]


CAMA_ACCENT = (115, 190, 78)   # yesil = camA (BGR)
CAMB_ACCENT = (210, 160, 70)   # mavi  = camB (BGR)
IN_ROI_COLOR = (90, 230, 120)  # o an ROI icinde = yesil
LEFT_COLOR = (150, 155, 160)   # ROI'den cikti = gri


def roi_state(records, zone_id, frame_index, fps, lookback_s=12.0, linger_s=3.0):
    """Bir zone icin o karedeki durumu cikarir.

    Doner: (green, gray)
      green: su an ROI icindeki track'ler -> (track_id, rec, dwell_s)
      gray : son linger_s icinde ROI'den cikmis track'ler -> (track_id, rec, age_s)
    records_by_frame onceden hesapli oldugu icin bu fonksiyon durumsuz;
    ileri-geri sarmada da dogru calisir.
    """
    lookback = int(lookback_s * fps)
    frames_seen = defaultdict(set)
    last_rec = {}
    last_frame = {}
    for f in range(max(0, frame_index - lookback), frame_index + 1):
        for rec in records.get(f, []):
            if zone_id in rec.get("zone_ids", []):
                tid = int(rec["track_id"])
                frames_seen[tid].add(f)
                # f artan sirada ilerledigi icin son atama zaten en yeni karedir;
                # burada max() cagirmak ic donguyu karesel hale getiriyordu.
                last_rec[tid] = rec
                last_frame[tid] = f
    green, gray = [], []
    for tid, fset in frames_seen.items():
        last = last_frame[tid]
        rec = last_rec[tid]
        if last == frame_index:
            dwell = 1
            g = frame_index
            while (g - 1) in fset:
                dwell += 1
                g -= 1
            green.append((tid, rec, dwell / fps))
        else:
            age = (frame_index - last) / fps
            if age <= linger_s:
                gray.append((tid, rec, age))
    green.sort(key=lambda x: -x[2])   # en uzun suredir iceride olan ustte
    gray.sort(key=lambda x: x[2])     # en yeni cikan ustte
    return green, gray


def draw_panel(matcher, now, records_a, records_b, frame_index, fps, matched_ids):
    panel = np.full((VIEW_HEIGHT * 2, PANEL_WIDTH, 3), (24, 29, 38), dtype=np.uint8)
    text(panel, "ROI DOLULUK KONSOLU", 16, 28, (255, 255, 255), 0.72, 2)
    text(panel, f"t={now:05.2f}s  yesil=ROI icinde  gri=cikti"
                f"  eslesme:{len(matcher.matches)}",
         16, 52, (160, 190, 220), 0.46)

    def cell(title, green, gray, x, y, width, height, accent, max_lines):
        cv2.rectangle(panel, (x, y), (x + width, y + height), (16, 20, 27), -1)
        cv2.rectangle(panel, (x, y), (x + width, y + 26), accent, -1)
        text(panel, f"{title}  icinde:{len(green)}", x + 8, y + 19, (14, 17, 22), 0.5, 2)
        cursor = y + 48
        shown = 0

        def row(tid, rec, suffix, base_color):
            """Tek satir; eslesmisse satir basina match rengiyle kare gosterge koyar."""
            nonlocal cursor, shown
            entry = matched_ids.get((rec["camera_id"], tid))
            mid = entry[0] if isinstance(entry, tuple) else entry
            tx = x + 8
            if mid:
                col = match_color(mid)
                cv2.rectangle(panel, (tx, cursor - 9), (tx + 10, cursor + 1), col, -1)
                tx += 16
            text(panel, f"#{tid:<4} {rec['class']:<5} {suffix}", tx, cursor, base_color, 0.46)
            if mid:
                text(panel, str(mid), tx + 150, cursor, match_color(mid), 0.44)
            cursor += 22
            shown += 1

        for tid, rec, dwell in green:
            if shown >= max_lines:
                break
            row(tid, rec, f"ROI {dwell:4.1f}s", IN_ROI_COLOR)
        for tid, rec, age in gray:
            if shown >= max_lines:
                break
            row(tid, rec, f"cikti {age:3.1f}s", LEFT_COLOR)
        extra = len(green) + len(gray) - shown
        if extra > 0:
            text(panel, f"... +{extra} daha", x + 8, cursor, (120, 128, 140), 0.42)

    # 2x2 izgara. Satir1: camA_exit | camB_entry ; Satir2: camB_exit | camA_entry
    cw, ch = 292, 300
    lx, rx = 16, 332
    y1, y2 = 72, 388
    maxl = 11
    st = {
        "camA_exit": roi_state(records_a, "camA_exit", frame_index, fps),
        "camA_entry": roi_state(records_a, "camA_entry", frame_index, fps),
        "camB_exit": roi_state(records_b, "camB_exit", frame_index, fps),
        "camB_entry": roi_state(records_b, "camB_entry", frame_index, fps),
    }
    cell("camA_exit", *st["camA_exit"], lx, y1, cw, ch, CAMA_ACCENT, maxl)
    cell("camB_entry", *st["camB_entry"], rx, y1, cw, ch, CAMB_ACCENT, maxl)
    cell("camB_exit", *st["camB_exit"], lx, y2, cw, ch, CAMB_ACCENT, maxl)
    cell("camA_entry", *st["camA_entry"], rx, y2, cw, ch, CAMA_ACCENT, maxl)
    return panel


def write_results(matcher, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    match_fields = ["match_id", "movement_label", "src_camera", "src_track",
                    "dst_camera", "dst_track", "delta_t"]
    with (output_dir / "fifo_matches.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=match_fields)
        writer.writeheader()
        for match in matcher.matches:
            writer.writerow({
                "match_id": match.match_id,
                "movement_label": match.movement_label,
                "src_camera": match.source.camera_id,
                "src_track": match.source.track_id,
                "dst_camera": match.target.camera_id,
                "dst_track": match.target.track_id,
                "delta_t": match.delta_t,
            })

    event_fields = ["timestamp", "camera_id", "track_id", "vehicle_class",
                    "movement_label", "event_type"]
    for name, rows in (("fifo_expired.csv", matcher.expired),
                       ("fifo_unmatched_entries.csv", matcher.unmatched_entries)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=event_fields)
            writer.writeheader()
            for event in rows:
                writer.writerow(event.__dict__)


def new_matcher(delay_s, width_s, fifo_cfg):
    return FifoMatcher(
        delay_s=delay_s,
        window_width_s=width_s,
        max_queue_size=int(fifo_cfg.get("max_queue_size", 30)),
        history_size=int(fifo_cfg.get("history_size", 10)),
    )


def advance_events(matcher, events, event_index, target_time, matched_ids):
    while event_index < len(events) and events[event_index][0] <= target_time + 1e-6:
        event = events[event_index][3]
        match = matcher.process(event)
        if match:
            matched_ids[(match.source.camera_id, match.source.track_id)] = match.match_id
            matched_ids[(match.target.camera_id, match.target.track_id)] = match.match_id
        event_index += 1
    matcher.expire(target_time)
    return event_index


def replay_state(events, target_time, delay_s, width_s, fifo_cfg):
    """Geri/ileri sarma sonrasi FIFO'yu hedef zamana kadar yeniden kurar."""
    matcher = new_matcher(delay_s, width_s, fifo_cfg)
    matched_ids = {}
    event_index = advance_events(matcher, events, 0, target_time, matched_ids)
    return matcher, event_index, matched_ids


def main():
    parser = argparse.ArgumentParser(description="Iki kamera + FIFO eslestirme konsolu")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "matching.yaml")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs" / "matching" / "fifo_console_30s.mp4")
    parser.add_argument("--show", action="store_true",
                        help="Duraklatma ve sarma kontrollu canli pencereyi ac")
    parser.add_argument("--match-source", choices=("csv", "fifo"), default="csv",
                        help="Cerceve renklerini besleyen eslesme kaynagi: "
                             "csv = kalibre boru hattinin matches.csv'si (varsayilan), "
                             "fifo = konsolun canli urettigi eslesmeler")
    parser.add_argument("--record", action="store_true",
                        help="--show sirasindaki atlama ve tekrarları da videoya yaz")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_yaml(config_path)
    timing = config.get("timing") or {}
    fifo_cfg = config.get("fifo") or {}
    duration_s = args.duration if args.duration is not None else timing.get("analysis_duration_s")
    duration_s = float(duration_s) if duration_s is not None else None
    delay_s = float(timing.get("delay_s", 0.7))
    width_s = float(timing.get("window_width_s", 0.8))
    clock_offset_s = float(timing.get("clock_offset_s", 0.0))

    matcher = new_matcher(delay_s, width_s, fifo_cfg)
    events = load_events(duration_s, clock_offset_s)
    # Renklendirme kaynagi: 'csv' kalibre edilmis boru hattinin dogrulanmis
    # eslesmelerini kullanir (mutual-best; GT'de %100 kesinlik). 'fifo' ise
    # konsolun canli urettigi eslesmeleri kullanir.
    confirmed = load_confirmed_matches() if args.match_source == "csv" else {}
    if args.match_source == "csv":
        print(f"Renklendirme: matches.csv ({len(confirmed)} track eslesmeye bagli)")
    records_a, zones_a = records_by_frame("camA")
    records_b, zones_b = records_by_frame("camB")
    paths = camera_paths()
    cap_a = cv2.VideoCapture(str(paths["camA"]))
    cap_b = cv2.VideoCapture(str(paths["camB"]))
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise SystemExit("HATA: kamera videolari acilamadi")

    fps = min(cap_a.get(cv2.CAP_PROP_FPS), cap_b.get(cv2.CAP_PROP_FPS)) or 20.0
    available_frames = int(min(cap_a.get(cv2.CAP_PROP_FRAME_COUNT),
                               cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    max_frames = min(available_frames, int(duration_s * fps)) if duration_s is not None else available_frames
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    should_write = not args.show or args.record
    writer = None
    if should_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, OUTPUT_SIZE)
        if not writer.isOpened():
            raise SystemExit(f"HATA: cikti videosu acilamadi: {output_path}")

    window_name = "Kayseri MOBESE FIFO"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, OUTPUT_SIZE[0], OUTPUT_SIZE[1])
        cv2.createTrackbar("Timeline", window_name, 0, max(0, max_frames - 1), lambda value: None)

    event_index = 0
    frame_index = 0
    matched_ids = {}
    paused = False
    left_keys = {81, 2424832, 65361, 63234, ord("j"), ord("a")}
    right_keys = {83, 2555904, 65363, 63235, ord("l"), ord("d")}
    try:
        while frame_index < max_frames:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break
            now = frame_index / fps
            event_index = advance_events(matcher, events, event_index, now, matched_ids)

            active_a = records_a.get(frame_index, [])
            active_b = records_b.get(frame_index, [])
            draw_zones(frame_a, zones_a)
            draw_zones(frame_b, zones_b)
            # csv modunda renkler dogrulanmis eslesmelerden, fifo modunda
            # konsolun canli urettigi eslesmelerden gelir.
            color_ids = confirmed if args.match_source == "csv" else matched_ids
            annotate_frame(frame_a, "camA", active_a, color_ids, now)
            annotate_frame(frame_b, "camB", active_b, color_ids, now)
            cameras = np.vstack((fit_view(frame_a), fit_view(frame_b)))
            panel = draw_panel(matcher, now, records_a, records_b, frame_index, fps,
                               color_ids)
            canvas = np.hstack((cameras, panel))
            if writer is not None:
                writer.write(canvas)

            if args.show:
                cv2.imshow(window_name, canvas)
                cv2.setTrackbarPos("Timeline", window_name, frame_index)
                key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / fps)))
                if key in (ord("q"), 27):
                    break
                slider_frame = cv2.getTrackbarPos("Timeline", window_name)
                target_frame = None
                if slider_frame != frame_index:
                    target_frame = slider_frame
                elif key == ord(" "):
                    paused = not paused
                elif key in left_keys:
                    target_frame = frame_index - int(fps)
                elif key in right_keys:
                    target_frame = frame_index + int(fps)
                elif key == ord("["):
                    target_frame = frame_index - int(5 * fps)
                elif key == ord("]"):
                    target_frame = frame_index + int(5 * fps)
                elif key in (ord("r"), 268632065):
                    target_frame = 0

                if target_frame is not None:
                    frame_index = max(0, min(max_frames - 1, int(target_frame)))
                    now = frame_index / fps
                    cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    matcher, event_index, matched_ids = replay_state(
                        events, now, delay_s, width_s, fifo_cfg
                    )
                    continue
                if paused:
                    cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    cap_b.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    continue
            frame_index += 1
    finally:
        cap_a.release()
        cap_b.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if should_write:
        final_time = frame_index / fps + matcher.max_delay_s + 0.01
        matcher.expire(final_time)
        write_results(matcher, output_path.parent)
        print(f"Video: {output_path} ({frame_index} kare, {frame_index / fps:.1f} sn)")
    else:
        print(f"Etkilesimli izleme kapandi: t={frame_index / fps:.1f} sn (dosyalar degistirilmedi)")
    print(f"FIFO match: {len(matcher.matches)}")
    print(f"Timeout exit: {len(matcher.expired)}")
    print(f"Eslesmeyen entry: {len(matcher.unmatched_entries)}")


if __name__ == "__main__":
    main()

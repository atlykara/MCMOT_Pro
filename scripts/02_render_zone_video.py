"""Faz 2 hareketli kontrol: ROI disini karartilmis, kutulanmis dogrulama videosu.

Kutular tracks_<cam>.jsonl'den okunur; hicbir model yuklenmez ve yeniden
tespit/takip calistirilmaz. Yalnizca foot_point'i ROI icinde olan araclar
kutulanir. Cikti outputs/videos altina yazilir.

Ornek:
    python scripts/02_render_zone_video.py --camera camA
    python scripts/02_render_zone_video.py --camera camB --start-frame 600 --max-frames 900 --scale 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.zones import (  # noqa: E402
    assign_lane,
    find_zone,
    load_zone_config,
)

DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
CONFIG_DIR = PROJECT_ROOT / "configs"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
VIDEOS_DIR = PROJECT_ROOT / "outputs" / "videos"

DEFAULT_MAX_FRAMES = 1800    # 30 fps'te ~1 dakika; kontrol icin yeterli
FRAME_COUNT_TOLERANCE = 60   # video ile takip kare sayisi arasindaki kabul edilen fark
OUTSIDE_DARKEN = 0.65        # ROI disi bu oranda siyahla harmanlanir
PROGRESS_EVERY = 100
DASH_LENGTH_PX = 14

ZONE_COLOR = (0, 200, 255)
LANE_COLOR = (255, 0, 255)
BOX_COLOR = (0, 255, 0)
FOOT_COLOR = (0, 165, 255)
TEXT_COLOR = (255, 255, 255)
LABEL_BG = (25, 25, 25)


def draw_label(image, text, origin, scale, color=TEXT_COLOR, padding=4):
    """Metni koyu kutu uzerine yazar; origin metnin taban cizgisidir.

    Etiket kare disina tasmasin diye konum goruntu sinirlarina cekilir.
    """
    thickness = max(1, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                scale, thickness)
    frame_h, frame_w = image.shape[:2]
    x = min(max(int(origin[0]), padding + 1), max(padding + 1, frame_w - width - padding - 1))
    y = min(max(int(origin[1]), height + padding + 1), frame_h - baseline - 1)
    cv2.rectangle(image, (x - padding, y - height - padding),
                  (x + width + padding, y + baseline), LABEL_BG, -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_dashed_line(image, start, end, color, thickness=2, dash=DASH_LENGTH_PX):
    """Iki nokta arasina kesikli cizgi cizer."""
    x1, y1 = start
    x2, y2 = end
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1e-6:
        return
    steps = max(int(length // dash), 1)
    for index in range(steps + 1):
        if index % 2:
            continue
        t0 = min(index * dash / length, 1.0)
        t1 = min((index + 1) * dash / length, 1.0)
        p0 = (int(round(x1 + (x2 - x1) * t0)), int(round(y1 + (y2 - y1) * t0)))
        p1 = (int(round(x1 + (x2 - x1) * t1)), int(round(y1 + (y2 - y1) * t1)))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def load_camera_entry(config_path: Path, camera_id: str):
    """cameras.yaml icinden kamera girdisini dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for entry in (data.get("cameras") or []):
        if isinstance(entry, dict) and entry.get("camera_id") == camera_id:
            return entry
    known = ", ".join(str(c.get("camera_id")) for c in (data.get("cameras") or [])
                      if isinstance(c, dict))
    raise SystemExit(f"HATA: '{camera_id}' cameras.yaml icinde bulunamadi. "
                     f"Tanimli kameralar: {known}")


def read_frame_records(jsonl_path: Path, config, first_frame, last_frame):
    """Yalnizca istenen kare araligindaki kayitlari kare bazinda toplar.

    Tum dosya bellege alinmaz; aralik disindaki kayitlar atlanir. Her kayit icin
    foot_point'in hangi bolgede oldugu find_zone ile belirlenir ve ROI DISINDA
    kalanlar hic saklanmaz (cizilmeyecekleri icin).
    """
    per_frame: dict[int, list[dict]] = {}
    kept = 0
    max_frame = -1
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = record.get("frame")
            if isinstance(frame, int):
                max_frame = max(max_frame, frame)
            if not isinstance(frame, int) or not (first_frame <= frame <= last_frame):
                continue
            foot = record.get("foot_point")
            bbox = record.get("bbox_xyxy")
            if not (isinstance(foot, list) and len(foot) == 2
                    and isinstance(bbox, list) and len(bbox) == 4):
                continue
            zone_id = find_zone((foot[0], foot[1]), config)
            if zone_id is None:
                continue  # ROI disi: kutulanmaz
            per_frame.setdefault(frame, []).append({
                "track_id": record.get("track_id"),
                "class": record.get("class"),
                "conf": record.get("conf"),
                "bbox": bbox,
                "foot": foot,
                "zone_id": zone_id,
                "timestamp": record.get("timestamp"),
            })
            kept += 1
    return per_frame, kept, max_frame


def build_mask(config, size, scale):
    """ROI maskesini bir kez uretir (bolgeler sabit oldugu icin her karede degil)."""
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in config.zones:
        points = np.array([[int(round(x * scale)), int(round(y * scale))]
                           for x, y in zone.polygon], dtype=np.int32)
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
    return mask.astype(bool)[:, :, None]


def draw_static_layer(image, config, scale, text_size):
    """Zone kenarligi, zone_id ve lane_divider gibi kare bagimsiz katmani cizer."""
    for zone in config.zones:
        points = np.array([[int(round(x * scale)), int(round(y * scale))]
                           for x, y in zone.polygon], dtype=np.int32)
        cv2.polylines(image, [points.reshape((-1, 1, 2))], True, ZONE_COLOR, 2, cv2.LINE_AA)
        top = min(points, key=lambda p: p[1])
        draw_label(image, zone.zone_id, (top[0] + 8, top[1] + 24), text_size, ZONE_COLOR)
        if zone.lane_divider is not None:
            (x1, y1), (x2, y2) = zone.lane_divider
            draw_dashed_line(image, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale),
                             LANE_COLOR, thickness=2)


def render(camera_id, video_path, jsonl_path, zones_path, out_path,
           start_frame, max_frames, fps_out, scale) -> bool:
    """Videoyu kare kare isler ve ROI odakli dogrulama videosunu yazar."""
    print(f"\n=== {camera_id} ===")
    try:
        config = load_zone_config(zones_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}")
        return False
    if not jsonl_path.is_file():
        print(f"HATA: takip dosyasi bulunamadi: {jsonl_path}")
        return False
    if not video_path.is_file():
        print(f"HATA: video dosyasi bulunamadi: {video_path}")
        return False

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"HATA: video cv2 ile acilamadi (codec veya dosya sorunu): {video_path}")
        return False

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames and start_frame >= total_frames:
            print(f"HATA: --start-frame {start_frame} videonun kare sayisindan buyuk "
                  f"({total_frames}).")
            return False

        last_frame = start_frame + max_frames - 1
        out_size = (max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))))
        text_size = max(0.45, min(1.0, out_size[0] / 1800.0))

        if (source_width, source_height) != config.frame_size:
            print(f"UYARI: video {source_width}x{source_height} ile zones dosyasindaki "
                  f"frame_size {config.frame_size[0]}x{config.frame_size[1]} farkli; "
                  "poligonlar kaymis gorunebilir.")

        per_frame, kept, max_track_frame = read_frame_records(
            jsonl_path, config, start_frame, last_frame)
        print(f"Video         : {video_path} ({source_width}x{source_height}, "
              f"{total_frames} kare)")
        print(f"Kare araligi  : {start_frame} - {last_frame}")
        print(f"ROI ici kayit : {kept} (yalnizca bunlar kutulanir)")

        # Kare numaralari ancak takip dosyasi BU videodan uretildiyse ortusur.
        if total_frames and max_track_frame >= 0 and total_frames > max_track_frame + 1 + FRAME_COUNT_TOLERANCE:
            print(f"UYARI: takip dosyasindaki en buyuk kare {max_track_frame}, video ise "
                  f"{total_frames} kare iceriyor. Takip buyuk olasilikla BASKA (daha kisa) "
                  "bir kaynaktan uretilmis; kare numaralari ortusmedigi icin kutular "
                  "yanlis karelere dusebilir.\n"
                  "       Dogru kaynagi --source ile verin, ornegin: "
                  f"--source data/samples/{camera_id}_sample.mp4")
        print(f"Cikti         : {out_size[0]}x{out_size[1]} @ {fps_out:g} fps")

        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, out_size)
        if not writer.isOpened():
            print(f"HATA: video yazici acilamadi (mp4v): {out_path}")
            return False

        mask = build_mask(config, out_size, scale)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        written = 0
        boxes_drawn = 0
        frame_index = start_frame
        try:
            while frame_index <= last_frame:
                # okumadan ONCE alinirsa bu, okunacak karenin kendi zaman damgasidir
                pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(f"Not           : video {frame_index}. karede bitti.")
                    break

                if scale != 1.0:
                    frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)

                # ROI disini karart: goz yalnizca bolgeye odaklansin
                darkened = cv2.convertScaleAbs(frame, alpha=1.0 - OUTSIDE_DARKEN, beta=0)
                frame = np.where(mask, frame, darkened)

                draw_static_layer(frame, config, scale, text_size)

                records = per_frame.get(frame_index, [])
                zones_by_id = {zone.zone_id: zone for zone in config.zones}
                for record in records:
                    x1, y1, x2, y2 = (int(round(v * scale)) for v in record["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    zone = zones_by_id.get(record["zone_id"])
                    lane = assign_lane(tuple(record["foot"]), zone) if zone else "unknown"
                    label = f"id={record['track_id']} {record['class']} {lane}"
                    draw_label(frame, label, (x1 + 2, max(y1 - 6, 16)), text_size, BOX_COLOR)
                    foot = (int(round(record["foot"][0] * scale)),
                            int(round(record["foot"][1] * scale)))
                    cv2.circle(frame, foot, 6, FOOT_COLOR, -1, cv2.LINE_AA)
                    cv2.circle(frame, foot, 6, (0, 0, 0), 1, cv2.LINE_AA)
                    boxes_drawn += 1

                # zaman damgasi once takip kaydindan; o karede kayit yoksa videonun kendi
                # konumundan alinir (yeniden hesaplanmaz)
                timestamp = records[0]["timestamp"] if records else None
                if not isinstance(timestamp, (int, float)) and pos_msec and pos_msec > 0:
                    timestamp = pos_msec / 1000.0
                stamp = f"t={timestamp:.3f} sn" if isinstance(timestamp, (int, float)) else "t=?"
                draw_label(frame, f"{camera_id}  kare {frame_index}  {stamp}",
                           (12, 28), text_size)
                draw_label(frame, f"ROI ici arac: {len(records)}", (12, 28 + int(34 * text_size / 0.6)),
                           text_size)

                writer.write(frame)
                written += 1
                frame_index += 1
                if written % PROGRESS_EVERY == 0:
                    print(f"  ... {written} kare yazildi (kare {frame_index - 1})")
        finally:
            writer.release()

        print(f"Yazilan kare  : {written}, cizilen kutu: {boxes_drawn}")
        print(f"Yazildi       : {out_path}")
        return written > 0
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 2: ROI odakli dogrulama videosu uretir (model yuklenmez)."
    )
    parser.add_argument("--camera", required=True, help="kamera kimligi (or. camA)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="baslangic kare numarasi (varsayilan 0)")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=f"islenecek kare sayisi (varsayilan {DEFAULT_MAX_FRAMES})")
    parser.add_argument("--fps-out", type=float, default=None,
                        help="cikti kare hizi; verilmezse cameras.yaml'daki fps")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="cikti olcegi, 0.5 = yari cozunurluk (varsayilan 1.0)")
    parser.add_argument("--source", type=Path, default=None,
                        help="video yolu; verilmezse cameras.yaml'daki video_path. "
                             "Takip dosyasi hangi videodan uretildiyse O video verilmeli.")
    parser.add_argument("--tracks", type=Path, default=None, help="tracks_<cam>.jsonl yolu")
    parser.add_argument("--zones", type=Path, default=None, help="zones_<cam>.yaml yolu")
    args = parser.parse_args()

    if args.start_frame < 0:
        raise SystemExit("HATA: --start-frame negatif olamaz.")
    if args.max_frames < 1:
        raise SystemExit("HATA: --max-frames en az 1 olmali.")
    if not 0 < args.scale <= 1.0:
        raise SystemExit("HATA: --scale 0 ile 1 arasinda olmali.")

    entry = load_camera_entry(DEFAULT_CAMERAS_CONFIG, args.camera)
    if args.source is not None:
        video_path = args.source
    else:
        raw_video = entry.get("video_path")
        if not raw_video:
            raise SystemExit(f"HATA: cameras.yaml icinde '{args.camera}' icin video_path yok.")
        video_path = Path(raw_video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    fps_out = args.fps_out
    if fps_out is None:
        fps_out = entry.get("fps")
        if not isinstance(fps_out, (int, float)) or fps_out <= 0:
            raise SystemExit(f"HATA: cameras.yaml icinde '{args.camera}' fps degeri yok; "
                             "--fps-out ile verin.")
        fps_out = float(fps_out)
    if fps_out <= 0:
        raise SystemExit("HATA: --fps-out pozitif olmali.")

    tracks_path = (args.tracks if args.tracks is not None
                   else TRACKS_DIR / f"tracks_{args.camera}.jsonl")
    if not tracks_path.is_absolute():
        tracks_path = PROJECT_ROOT / tracks_path
    zones_path = (args.zones if args.zones is not None
                  else CONFIG_DIR / f"zones_{args.camera}.yaml")
    if not zones_path.is_absolute():
        zones_path = PROJECT_ROOT / zones_path

    out_path = VIDEOS_DIR / f"zone_annotated_{args.camera}.mp4"

    ok = render(args.camera, video_path, tracks_path, zones_path, out_path,
                args.start_frame, args.max_frames, fps_out, args.scale)
    if not ok:
        raise SystemExit("HATA: video uretilemedi (yukaridaki mesajlara bakin).")


if __name__ == "__main__":
    main()

"""Faz 3: aday track'ler icin BASIT gorsel ozellikler cikarir (boyut + renk).

Proje kural yorumu: derin/ogrenilmis Re-ID embedding YASAK; elle-tanimli basit
ozellikler (bbox boyutu, ortalama renk) serbest. Bu ozellikler handoff skorunu
guclendirir (ozellikle ayni sinif farkli araclari ayirmak icin, or. iki "truck").

Cikarilan ozellikler (her track'in EN NET karesinden = max conf):
    area       : bbox alani (piksel^2)
    size_norm  : area / (o kameradaki medyan area)  -> perspektife gore normalize
    aspect     : bbox en/boy orani (w/h) -> sekil ipucu (kompakt vs uzun)
    height     : bbox yuksekligi -> arac tipi ipucu (kamyon yuksek, sedan alcak)
    h, s, v    : bbox ic %60 bolgesinin ortalama HSV rengi (arka plan haric)

Neden HSV: RGB'ye gore aydinlatmaya daha az duyarli; iki kamera arasi daha saglam.
Neden ic %60: bbox kenarlarinda yol/arka plan olur; merkez arac govdesidir.

Girdi : outputs/tracks/tracks_stitched_<cam>.jsonl + ham video
Cikti : outputs/matching/track_features_<cam>.csv
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
MATCH_DIR = PROJECT_ROOT / "outputs" / "matching"
CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"


def wanted_tracks(camera, duration_s=None):
    """Sadece hareketli (movement != other) track'ler icin ozellik cikar."""
    path = ZONES_DIR / f"zone_tracks_mapped_{camera}.csv"
    ids = set()
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["movement_label"] != "other"
                    and (duration_s is None or float(r["enter_timestamp"]) <= duration_s)):
                ids.add(str(r["track_id"]))
    return ids


def best_frame_per_track(camera, wanted):
    """Her istenen track icin en yuksek conf'lu kare + bbox."""
    by_track = defaultdict(list)
    # ID-UZAYI: aday track'leri zone_tracks_mapped'ten gelir; o da RAW
    # tracks_<cam>.jsonl id'lerini kullanir. Stitching global_id'yi baştan
    # numaralandirdigi icin (raw!=stitched), ozellikleri de RAW dosyadan
    # cikarmaliyiz; yoksa color/size cogu track icin yanlis-anahtarli/eksik olur.
    tracks_path = TRACKS_DIR / f"tracks_{camera}.jsonl"
    with open(tracks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            tid = str(r["track_id"])
            if tid in wanted:
                by_track[tid].append(r)
    targets = {}
    for tid, recs in by_track.items():
        rec = max(recs, key=lambda r: r["conf"])
        targets[tid] = (int(rec["frame"]), rec["bbox_xyxy"])
    return targets


def extract_features(img, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    w, h = x2 - x1, y2 - y1
    if w <= 2 or h <= 2:
        return None
    area = w * h
    aspect = round(w / h, 3)
    # ic %60 bolge (kenar/arka plan haric)
    mx, my = int(w * 0.2), int(h * 0.2)
    roi = img[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    return {"area": area, "aspect": aspect, "height": h,
            "h": round(float(hsv[0]), 1), "s": round(float(hsv[1]), 1), "v": round(float(hsv[2]), 1)}


def scan_video(video_path, targets):
    """Videoyu sirali tara; hedef karelerde bbox'tan ozellik cikar."""
    frame_to_tracks = defaultdict(list)
    for tid, (f, bb) in targets.items():
        frame_to_tracks[f].append((tid, bb))
    max_frame = max(frame_to_tracks) if frame_to_tracks else -1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"HATA: video acilamadi: {video_path}")
    feats = {}
    fidx = 0
    while fidx <= max_frame:
        ok, img = cap.read()
        if not ok:
            break
        if fidx in frame_to_tracks:
            for tid, bb in frame_to_tracks[fidx]:
                f = extract_features(img, bb)
                if f:
                    feats[tid] = f
        fidx += 1
    cap.release()
    return feats


def main():
    ap = argparse.ArgumentParser(description="Faz 3: aday track gorsel ozellikleri (boyut+renk).")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--source", default=None, help="video yolu; verilmezse cameras.yaml kullanilir")
    ap.add_argument("--duration", type=float, default=None,
                    help="Yalnizca ilk N saniyedeki track'ler")
    args = ap.parse_args()

    source = args.source
    if source is None:
        with CAMERAS_CONFIG.open("r", encoding="utf-8") as fh:
            camera_rows = (yaml.safe_load(fh) or {}).get("cameras") or []
        camera_cfg = next(
            (row for row in camera_rows if row.get("camera_id") == args.camera),
            None,
        )
        if camera_cfg is None:
            raise SystemExit(f"HATA: cameras.yaml icinde kamera yok: {args.camera}")
        source = camera_cfg["video_path"]
    video_path = Path(source)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    wanted = wanted_tracks(args.camera, args.duration)
    targets = best_frame_per_track(args.camera, wanted)
    print(f"{args.camera}: {len(targets)} hareketli track icin ozellik cikariliyor...")
    feats = scan_video(video_path, targets)

    # boyutu kameraya gore normalize et (medyan area'ya oran)
    areas = [f["area"] for f in feats.values()]
    med = statistics.median(areas) if areas else 1.0
    for f in feats.values():
        f["size_norm"] = round(f["area"] / med, 3)

    out = MATCH_DIR / f"track_features_{args.camera}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["camera", "track_id", "area", "size_norm", "aspect", "height", "h", "s", "v"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for tid, f in sorted(feats.items(), key=lambda kv: int(kv[0])):
            w.writerow({"camera": args.camera, "track_id": tid, **f})
    print(f"Yazildi: {out} ({len(feats)} track, medyan area={med:.0f})")


if __name__ == "__main__":
    main()

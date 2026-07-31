"""Faz 2 girdi envanteri: Faz 1 tracks_<cam>.jsonl ciktilarinin gercek semasi ve sagligi.

Script SALT OKUR. Hicbir girdi dosyasini degistirmez; tek yazdigi dosya
docs/faz2_girdi_envanteri.md raporudur.

Kapsam tek kameradir: camA ve camB birbirinden bagimsiz raporlanir, kameralar
arasi hicbir karsilastirma veya birlestirme yapilmaz (o Faz 3'un isidir).

Ornek:
    python scripts/02_check_tracks.py
    python scripts/02_check_tracks.py --tracks-dir outputs/tracks --cameras-config configs/cameras.yaml
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "faz2_girdi_envanteri.md"

MISSING = "EKSIK"

# --- kontrol esikleri (bu script salt okur; yeni config dosyasi uretmez) ---
FOOT_POINT_TOL = 0.06        # foot_point ile bbox turevi arasinda kabul edilen px sapmasi
TIMESTAMP_CHECK_RECORDS = 1000   # timestamp ~ frame/fps kontrolu icin ilk N kayit
TIMESTAMP_MAX_DEV_SEC = 0.005    # bu sapmanin uzerindeki timestamp farki UYARI
SHORT_TRACK_MIN_FRAMES = 5   # bu kareden kisa track'ler ayrica sayilir
FPS_TOL = 0.5                # cameras.yaml fps ile video fps farki bu esigi asarsa BUYUK UYARI

# tracks_<cam>.jsonl kontrat alanlari (PROJE_REHBERI Bolum 6 / src/mcmot/io_utils.py)
EXPECTED_TRACK_FIELDS = (
    "timestamp", "frame", "camera_id", "track_id", "class",
    "conf", "bbox_xyxy", "foot_point", "hints",
)
EXPECTED_HINT_FIELDS = ("dominant_color", "size_class", "aspect_ratio")


# --------------------------------------------------------------------------
# yardimcilar
# --------------------------------------------------------------------------

def fmt(value, nd=2):
    """Sayiyi sabit basamakla bicimler; None ise EKSIK dondurur."""
    if value is None:
        return MISSING
    if isinstance(value, float) and value != value:  # NaN
        return MISSING
    return f"{value:.{nd}f}"


def quantile(series, q):
    """Bos olmayan seriden yuzdelik dondurur; yoksa None."""
    clean = series.dropna() if series is not None else None
    if clean is None or clean.empty:
        return None
    return float(clean.quantile(q))


def series_stats(series, quantiles=(0.05, 0.5, 0.95)):
    """min / istenen yuzdelikler / max sozlugu dondurur."""
    clean = series.dropna() if series is not None else None
    if clean is None or clean.empty:
        return {"min": None, "max": None, **{f"p{int(q * 100):02d}": None for q in quantiles}}
    stats = {"min": float(clean.min()), "max": float(clean.max())}
    for q in quantiles:
        stats[f"p{int(q * 100):02d}"] = float(clean.quantile(q))
    return stats


def resolve_path(raw):
    """Config'teki yolu PROJECT_ROOT'a gore cozer."""
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------
# okuma
# --------------------------------------------------------------------------

def load_cameras_config(config_path):
    """cameras.yaml'i okur; (kamera listesi, hata mesaji) dondurur."""
    if not config_path.is_file():
        return [], f"{MISSING}: kamera config bulunamadi: {config_path}"
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        return [], f"HATA: cameras.yaml okunamadi ({exc})"
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return [], f"{MISSING}: cameras.yaml icinde 'cameras' listesi yok veya bos"
    return [c for c in cameras if isinstance(c, dict)], None


def load_tracks(jsonl_path, max_records=0):
    """JSONL'i okuyup DataFrame ve sema bilgisi dondurur; bozuk satirda cokmez."""
    info = {
        "exists": jsonl_path.is_file(),
        "size_bytes": None,
        "total_lines": 0,
        "parsed_lines": 0,
        "bad_lines": 0,
        "first_keys": None,
        "first_hint_keys": None,
        "diff_key_count": 0,
        "diff_hint_key_count": 0,
        "truncated": False,
    }
    if not info["exists"]:
        return pd.DataFrame(), info

    info["size_bytes"] = jsonl_path.stat().st_size
    rows = []
    first_keys = None
    first_hint_keys = None

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            info["total_lines"] += 1
            if max_records and info["parsed_lines"] >= max_records:
                info["truncated"] = True
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                info["bad_lines"] += 1
                continue
            if not isinstance(rec, dict):
                info["bad_lines"] += 1
                continue

            info["parsed_lines"] += 1
            keys = tuple(rec.keys())
            hints = rec.get("hints")
            hint_keys = tuple(hints.keys()) if isinstance(hints, dict) else None

            if first_keys is None:
                first_keys = keys
                first_hint_keys = hint_keys
                info["first_keys"] = list(keys)
                info["first_hint_keys"] = list(hint_keys) if hint_keys is not None else None
            else:
                if keys != first_keys:
                    info["diff_key_count"] += 1
                if hint_keys != first_hint_keys:
                    info["diff_hint_key_count"] += 1

            bbox = rec.get("bbox_xyxy")
            bbox = bbox if isinstance(bbox, list) and len(bbox) == 4 else [None] * 4
            foot = rec.get("foot_point")
            foot = foot if isinstance(foot, list) and len(foot) == 2 else [None, None]

            rows.append({
                "timestamp": rec.get("timestamp"),
                "frame": rec.get("frame"),
                "camera_id": rec.get("camera_id"),
                "track_id": rec.get("track_id"),
                "class": rec.get("class"),
                "conf": rec.get("conf"),
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "fx": foot[0], "fy": foot[1],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ("timestamp", "conf", "x1", "y1", "x2", "y2", "fx", "fy"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("frame", "track_id"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["bbox_w"] = df["x2"] - df["x1"]
        df["bbox_h"] = df["y2"] - df["y1"]
    return df, info


def probe_video(video_path):
    """Videoyu cv2 ile acip gercek fps / cozunurluk / kare sayisi okur."""
    info = {"path": video_path, "exists": False, "opened": False,
            "fps": None, "width": None, "height": None, "frame_count": None}
    if video_path is None:
        return info
    info["exists"] = video_path.is_file()
    if not info["exists"]:
        return info

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return info
        info["opened"] = True
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info["fps"] = float(fps) if fps and fps > 0 else None
        info["width"] = int(width) if width and width > 0 else None
        info["height"] = int(height) if height and height > 0 else None
        info["frame_count"] = int(count) if count and count > 0 else None
    finally:
        cap.release()
    return info


# --------------------------------------------------------------------------
# analiz
# --------------------------------------------------------------------------

def analyze_camera(camera_id, entry, tracks_dir, max_records):
    """Tek kamera icin dosya, sema, istatistik, saglik ve video bilgisini toplar."""
    result = {"camera_id": camera_id, "warnings": []}
    warn = result["warnings"].append

    # --- config alanlari ---
    entry = entry or {}
    cfg_fps = entry.get("fps")
    cfg_fps = float(cfg_fps) if isinstance(cfg_fps, (int, float)) else None
    result["config"] = {
        "in_config": bool(entry),
        "video_path": entry.get("video_path") or MISSING,
        "fps": cfg_fps,
        "clock_offset_seconds": entry.get("clock_offset_seconds"),
        "notes": entry.get("notes") or MISSING,
        # frame_size cameras.yaml semasinda yok; tahmin edilmez.
        "frame_size": entry.get("frame_size") or MISSING,
    }
    if not entry:
        warn(f"{camera_id}: cameras.yaml icinde kamera girdisi {MISSING}.")
    if cfg_fps is None:
        warn(f"{camera_id}: cameras.yaml fps degeri {MISSING}; timestamp tutarlilik kontrolu yapilamadi.")

    # --- video ---
    video = probe_video(resolve_path(entry.get("video_path")))
    result["video"] = video
    if entry.get("video_path") and not video["exists"]:
        warn(f"{camera_id}: video dosyasi {MISSING}: {video['path']}")
    elif video["exists"] and not video["opened"]:
        warn(f"{camera_id}: video cv2 ile acilamadi: {video['path']}")
    if video["opened"] and cfg_fps is not None and video["fps"] is not None:
        if abs(video["fps"] - cfg_fps) > FPS_TOL:
            warn(f"BUYUK UYARI - {camera_id}: cameras.yaml fps={cfg_fps:g} ile "
                 f"videonun gercek fps={video['fps']:.3f} degeri farkli "
                 f"(fark {abs(video['fps'] - cfg_fps):.3f} > {FPS_TOL}). "
                 "Zaman ekseni hatali olur; once bu duzeltilmeli.")

    # --- tracks dosyasi ---
    jsonl_path = tracks_dir / f"tracks_{camera_id}.jsonl"
    df, info = load_tracks(jsonl_path, max_records=max_records)
    result["path"] = jsonl_path
    result["file"] = info
    result["df_len"] = len(df)

    if not info["exists"]:
        warn(f"{camera_id}: takip dosyasi {MISSING}: {jsonl_path}")
        result["empty"] = True
        return result
    if info["bad_lines"]:
        warn(f"{camera_id}: JSON olarak cozulemeyen {info['bad_lines']} satir atlandi.")
    if df.empty:
        warn(f"{camera_id}: takip dosyasinda gecerli kayit yok.")
        result["empty"] = True
        return result
    result["empty"] = False

    # --- sema kontrolu ---
    first_keys = info["first_keys"] or []
    hint_keys = info["first_hint_keys"]
    missing_fields = [f for f in EXPECTED_TRACK_FIELDS if f not in first_keys]
    missing_hints = ([h for h in EXPECTED_HINT_FIELDS if h not in hint_keys]
                     if hint_keys is not None else list(EXPECTED_HINT_FIELDS))
    result["missing_fields"] = missing_fields
    result["missing_hints"] = missing_hints
    if missing_fields:
        warn(f"{camera_id}: ilk kayitta kontrat alanlari {MISSING}: {missing_fields}")
    if hint_keys is None:
        warn(f"{camera_id}: ilk kayitta 'hints' sozlugu {MISSING}.")
    elif missing_hints:
        warn(f"{camera_id}: ilk kayitta hints alanlari {MISSING}: {missing_hints}")
    if info["diff_key_count"]:
        warn(f"{camera_id}: ilk kayittan farkli ust duzey anahtar setine sahip "
             f"{info['diff_key_count']} kayit var.")
    if info["diff_hint_key_count"]:
        warn(f"{camera_id}: ilk kayittan farkli hints anahtar setine sahip "
             f"{info['diff_hint_key_count']} kayit var.")

    # --- zaman / kare ---
    ts = df["timestamp"]
    fr = df["frame"]
    ts_min, ts_max = (float(ts.min()), float(ts.max())) if ts.notna().any() else (None, None)
    fr_min, fr_max = (int(fr.min()), int(fr.max())) if fr.notna().any() else (None, None)
    result["time"] = {
        "ts_min": ts_min, "ts_max": ts_max,
        "ts_span": (ts_max - ts_min) if ts_min is not None else None,
        "frame_min": fr_min, "frame_max": fr_max,
    }
    if ts.isna().any():
        warn(f"{camera_id}: {int(ts.isna().sum())} kayitta timestamp {MISSING}/sayisal degil.")
    if fr.isna().any():
        warn(f"{camera_id}: {int(fr.isna().sum())} kayitta frame {MISSING}/sayisal degil.")

    # --- temel dagilimlar ---
    result["unique_tracks"] = int(df["track_id"].dropna().nunique())
    result["class_counts"] = df["class"].value_counts(dropna=False).to_dict()
    result["conf"] = series_stats(df["conf"], quantiles=(0.05, 0.5, 0.95))
    result["bbox_w"] = series_stats(df["bbox_w"])
    result["bbox_h"] = series_stats(df["bbox_h"])
    result["foot_x"] = series_stats(df["fx"])
    result["foot_y"] = series_stats(df["fy"])

    # --- foot_point sinir kontrolu (sinir kaynagi: video cozunurlugu) ---
    bounds = {"source": MISSING, "width": None, "height": None,
              "out_x": None, "out_y": None}
    if video["opened"] and video["width"] and video["height"]:
        bounds["source"] = f"video ({video['path'].name})"
        bounds["width"], bounds["height"] = video["width"], video["height"]
        out_x = int(((df["fx"] < 0) | (df["fx"] > video["width"])).sum())
        out_y = int(((df["fy"] < 0) | (df["fy"] > video["height"])).sum())
        bounds["out_x"], bounds["out_y"] = out_x, out_y
        if out_x or out_y:
            warn(f"{camera_id}: {out_x} kayitta foot_point x, {out_y} kayitta foot_point y "
                 f"kare sinirlari disinda ({video['width']}x{video['height']}).")
    else:
        warn(f"{camera_id}: kare boyutu {MISSING} (cameras.yaml'da frame_size alani yok, "
             "video da okunamadi); foot_point sinir kontrolu yapilamadi.")
    result["bounds"] = bounds

    # --- timestamp ~ frame/fps tutarliligi ---
    tscheck = {"checked": 0, "fps": cfg_fps, "max_dev": None,
               "mean_dev": None, "over_tol": None}
    if cfg_fps and cfg_fps > 0:
        head = df.head(TIMESTAMP_CHECK_RECORDS).dropna(subset=["timestamp", "frame"])
        if not head.empty:
            dev = head["timestamp"] - head["frame"] / cfg_fps
            tscheck["checked"] = int(len(head))
            tscheck["max_dev"] = float(dev.abs().max())
            tscheck["mean_dev"] = float(dev.mean())
            tscheck["over_tol"] = int((dev.abs() > TIMESTAMP_MAX_DEV_SEC).sum())
            if tscheck["over_tol"]:
                warn(f"UYARI - {camera_id}: ilk {tscheck['checked']} kayitta timestamp, "
                     f"frame/fps ({cfg_fps:g}) ile tutarsiz; {tscheck['over_tol']} kayitta "
                     f"sapma > {TIMESTAMP_MAX_DEV_SEC} sn (maks {tscheck['max_dev']:.4f} sn, "
                     f"ortalama isaretli sapma {tscheck['mean_dev']:+.4f} sn).")
    result["ts_check"] = tscheck

    # --- foot_point = bbox alt-orta kontrolu ---
    fp = {"bad_x": None, "bad_y": None, "bad_total": None, "tol": FOOT_POINT_TOL}
    valid = df.dropna(subset=["x1", "x2", "y2", "fx", "fy"])
    if not valid.empty:
        bad_x_mask = ((valid["fx"] - (valid["x1"] + valid["x2"]) / 2).abs() > FOOT_POINT_TOL)
        bad_y_mask = ((valid["fy"] - valid["y2"]).abs() > FOOT_POINT_TOL)
        fp["bad_x"] = int(bad_x_mask.sum())
        fp["bad_y"] = int(bad_y_mask.sum())
        fp["bad_total"] = int((bad_x_mask | bad_y_mask).sum())
        fp["checked"] = int(len(valid))
        if fp["bad_total"]:
            warn(f"UYARI - {camera_id}: {fp['bad_total']} kayitta foot_point, bbox alt-orta "
                 f"noktasiyla uyusmuyor (tolerans {FOOT_POINT_TOL}).")
    else:
        fp["checked"] = 0
        warn(f"{camera_id}: foot_point/bbox kontrolu icin gecerli kayit yok.")
    result["foot_check"] = fp

    # --- frame sirasi azalmiyor mu ---
    frames = df["frame"].dropna()
    decreasing = int((frames.diff() < 0).sum()) if len(frames) > 1 else 0
    result["frame_order"] = {"decreasing_steps": decreasing}
    if decreasing:
        warn(f"UYARI - {camera_id}: dosya sirasinda {decreasing} yerde frame numarasi azaliyor.")

    # --- track omru (kac karede gorunuyor) ---
    grouped = df.dropna(subset=["track_id"]).groupby("track_id")
    appear = grouped["frame"].count()
    span = grouped["frame"].max() - grouped["frame"].min() + 1
    life = {"p05": quantile(appear, 0.05), "p50": quantile(appear, 0.50),
            "p95": quantile(appear, 0.95),
            "max": float(appear.max()) if not appear.empty else None,
            "min": float(appear.min()) if not appear.empty else None,
            "short_count": int((appear < SHORT_TRACK_MIN_FRAMES).sum()) if not appear.empty else 0,
            "total": int(len(appear)),
            "span_p50": quantile(span, 0.50),
            "span_max": float(span.max()) if not span.empty else None}
    result["lifetime"] = life
    if life["short_count"]:
        warn(f"{camera_id}: {life['short_count']} / {life['total']} track "
             f"{SHORT_TRACK_MIN_FRAMES} kareden kisa (parcalanma veya hayalet tespit adayi).")

    # --- ayni frame icinde ayni track_id ---
    dup_mask = df.dropna(subset=["frame", "track_id"]).duplicated(subset=["frame", "track_id"], keep=False)
    dup_rows = int(dup_mask.sum())
    result["duplicates"] = {"rows": dup_rows}
    if dup_rows:
        warn(f"UYARI - {camera_id}: ayni frame icinde ayni track_id {dup_rows} kayitta "
             "tekrar ediyor (olmamali).")

    # --- video kare sayisi ile takip kare araligi ---
    if video["frame_count"] and fr_max is not None and fr_max >= video["frame_count"]:
        warn(f"{camera_id}: en buyuk frame ({fr_max}) videonun kare sayisindan "
             f"({video['frame_count']}) kucuk degil; kare indeksleme kontrol edilmeli.")
    return result


# --------------------------------------------------------------------------
# raporlama
# --------------------------------------------------------------------------

def render_camera(res):
    """Tek kamera bolumunu markdown satirlari olarak uretir."""
    cam = res["camera_id"]
    cfg = res["config"]
    info = res["file"]
    lines = [f"## Kamera - {cam}", ""]

    # config
    lines += [
        "### cameras.yaml girdisi", "",
        "| alan | deger |", "|---|---|",
        f"| camera_id | {cam} |",
        f"| video_path | {cfg['video_path']} |",
        f"| fps | {cfg['fps'] if cfg['fps'] is not None else MISSING} |",
        f"| clock_offset_seconds | {cfg['clock_offset_seconds'] if cfg['clock_offset_seconds'] is not None else MISSING} |",
        f"| frame_size | {cfg['frame_size']} |",
        f"| notes | {cfg['notes']} |",
        "",
    ]

    # dosya
    size_mb = info["size_bytes"] / (1024 * 1024) if info["size_bytes"] else None
    lines += [
        "### Takip dosyasi", "",
        "| alan | deger |", "|---|---|",
        f"| yol | `{res['path']}` |",
        f"| dosya var mi | {'evet' if info['exists'] else MISSING} |",
        f"| satir sayisi | {info['total_lines']} |",
        f"| cozulen kayit | {info['parsed_lines']} |",
        f"| bozuk satir | {info['bad_lines']} |",
        f"| dosya boyutu | {fmt(size_mb)} MB ({info['size_bytes'] if info['size_bytes'] is not None else MISSING} bayt) |",
        "",
    ]
    if info["truncated"]:
        lines += [f"> NOT: `--max-records` sinirlandi; istatistikler ilk "
                  f"{info['parsed_lines']} kayittan hesaplandi.", ""]

    if res.get("empty"):
        lines += ["Kayit okunamadigi icin istatistik uretilmedi.", ""]
        lines += render_video(res)
        return lines

    # sema
    keys = info["first_keys"] or []
    hints = info["first_hint_keys"]
    lines += [
        "### Sema (ilk kayit)", "",
        f"- Ust duzey anahtarlar ({len(keys)}): `{keys}`",
        f"- `hints` anahtarlari: " + (f"`{hints}`" if hints is not None else MISSING),
        f"- Kontratta olup ilk kayitta olmayan alan: "
        + (f"`{res['missing_fields']}`" if res["missing_fields"] else "yok"),
        f"- Kontratta olup hints icinde olmayan alan: "
        + (f"`{res['missing_hints']}`" if res["missing_hints"] else "yok"),
        f"- Farkli anahtar setine sahip kayit: {info['diff_key_count']} (hints: {info['diff_hint_key_count']})",
        "",
    ]

    # zaman / kare
    t = res["time"]
    lines += [
        "### Zaman ve kare araligi", "",
        "| olcum | deger |", "|---|---|",
        f"| timestamp min | {fmt(t['ts_min'], 3)} |",
        f"| timestamp max | {fmt(t['ts_max'], 3)} |",
        f"| timestamp span (sn) | {fmt(t['ts_span'], 3)} |",
        f"| frame min | {t['frame_min'] if t['frame_min'] is not None else MISSING} |",
        f"| frame max | {t['frame_max'] if t['frame_max'] is not None else MISSING} |",
        f"| benzersiz track_id | {res['unique_tracks']} |",
        "",
    ]

    # class dagilimi
    lines += ["### class dagilimi", "", "| class | kayit |", "|---|---|"]
    total = sum(res["class_counts"].values()) or 1
    for name, count in sorted(res["class_counts"].items(), key=lambda kv: -kv[1]):
        label = name if isinstance(name, str) else MISSING
        lines.append(f"| {label} | {count} ({100 * count / total:.1f}%) |")
    lines.append("")

    # conf / bbox / foot_point
    c, bw, bh = res["conf"], res["bbox_w"], res["bbox_h"]
    fx, fy = res["foot_x"], res["foot_y"]
    lines += [
        "### conf, bbox ve foot_point istatistikleri", "",
        "| olcum | min | p05 | p50 | p95 | max |", "|---|---|---|---|---|---|",
        f"| conf | {fmt(c['min'], 3)} | {fmt(c['p05'], 3)} | {fmt(c['p50'], 3)} | {fmt(c['p95'], 3)} | {fmt(c['max'], 3)} |",
        f"| bbox genislik (px) | {fmt(bw['min'], 1)} | {fmt(bw['p05'], 1)} | {fmt(bw['p50'], 1)} | {fmt(bw['p95'], 1)} | {fmt(bw['max'], 1)} |",
        f"| bbox yukseklik (px) | {fmt(bh['min'], 1)} | {fmt(bh['p05'], 1)} | {fmt(bh['p50'], 1)} | {fmt(bh['p95'], 1)} | {fmt(bh['max'], 1)} |",
        f"| foot_point x | {fmt(fx['min'], 1)} | {fmt(fx['p05'], 1)} | {fmt(fx['p50'], 1)} | {fmt(fx['p95'], 1)} | {fmt(fx['max'], 1)} |",
        f"| foot_point y | {fmt(fy['min'], 1)} | {fmt(fy['p05'], 1)} | {fmt(fy['p50'], 1)} | {fmt(fy['p95'], 1)} | {fmt(fy['max'], 1)} |",
        "",
    ]

    b = res["bounds"]
    lines += [
        "### foot_point kare sinirlari icinde mi", "",
        "| alan | deger |", "|---|---|",
        f"| sinir kaynagi | {b['source']} |",
        f"| kare boyutu | " + (f"{b['width']}x{b['height']}" if b["width"] else MISSING) + " |",
        f"| x sinir disi kayit | {b['out_x'] if b['out_x'] is not None else MISSING} |",
        f"| y sinir disi kayit | {b['out_y'] if b['out_y'] is not None else MISSING} |",
        "",
        "> `cameras.yaml` icinde `frame_size` alani tanimli degil; kare sinirlari "
        "yalnizca videodan okunur, tahmin edilmez.",
        "",
    ]

    # saglik
    ts = res["ts_check"]
    fp = res["foot_check"]
    life = res["lifetime"]
    ts_row = (f"| timestamp ~ frame/fps | ilk {ts['checked']} kayit, fps="
              f"{ts['fps'] if ts['fps'] is not None else MISSING} | "
              f"maks sapma {fmt(ts['max_dev'], 4)} sn, ortalama {fmt(ts['mean_dev'], 4)} sn, "
              f"tolerans disi {ts['over_tol'] if ts['over_tol'] is not None else MISSING} |")
    lines += [
        "### Saglik kontrolleri", "",
        "| kontrol | kapsam | sonuc |", "|---|---|---|",
        ts_row,
        f"| foot_point = bbox alt-orta | {fp['checked']} kayit, tolerans {fp['tol']} | "
        f"bozuk x {fp['bad_x'] if fp['bad_x'] is not None else MISSING}, "
        f"bozuk y {fp['bad_y'] if fp['bad_y'] is not None else MISSING}, "
        f"toplam bozuk {fp['bad_total'] if fp['bad_total'] is not None else MISSING} |",
        f"| frame sirasi azalmiyor mu | {info['parsed_lines']} kayit | "
        f"azalan gecis {res['frame_order']['decreasing_steps']} |",
        f"| ayni frame'de ayni track_id | {info['parsed_lines']} kayit | "
        f"tekrar eden kayit {res['duplicates']['rows']} |",
        "",
    ]

    lines += [
        "### Track omru (kac karede gorunuyor)", "",
        "| olcum | deger |", "|---|---|",
        f"| toplam track | {life['total']} |",
        f"| min kare | {fmt(life['min'], 0)} |",
        f"| p05 kare | {fmt(life['p05'], 1)} |",
        f"| p50 kare | {fmt(life['p50'], 1)} |",
        f"| p95 kare | {fmt(life['p95'], 1)} |",
        f"| max kare | {fmt(life['max'], 0)} |",
        f"| medyan kare araligi (ilk-son) | {fmt(life['span_p50'], 1)} |",
        f"| en genis kare araligi | {fmt(life['span_max'], 0)} |",
        f"| {SHORT_TRACK_MIN_FRAMES} kareden kisa track | {life['short_count']} |",
        "",
    ]

    lines += render_video(res)
    return lines


def render_video(res):
    """Video sondaj bolumunu markdown satirlari olarak uretir."""
    v = res["video"]
    cfg_fps = res["config"]["fps"]
    if v["opened"] and v["fps"] is not None and cfg_fps is not None:
        fps_cmp = ("uyumlu" if abs(v["fps"] - cfg_fps) <= FPS_TOL
                   else f"**BUYUK UYARI - FARKLI** (fark {abs(v['fps'] - cfg_fps):.3f})")
    else:
        fps_cmp = MISSING
    duration = (v["frame_count"] / v["fps"]) if (v["frame_count"] and v["fps"]) else None
    return [
        "### Video dosyasi", "",
        "| alan | deger |", "|---|---|",
        f"| yol | " + (f"`{v['path']}`" if v["path"] else MISSING) + " |",
        f"| dosya var mi | {'evet' if v['exists'] else MISSING} |",
        f"| cv2 ile acildi mi | {'evet' if v['opened'] else 'hayir'} |",
        f"| gercek fps | {fmt(v['fps'], 3)} |",
        f"| config fps | {cfg_fps if cfg_fps is not None else MISSING} |",
        f"| fps karsilastirmasi | {fps_cmp} |",
        f"| genislik x yukseklik | " + (f"{v['width']}x{v['height']}" if v["width"] else MISSING) + " |",
        f"| toplam kare | {v['frame_count'] if v['frame_count'] is not None else MISSING} |",
        f"| sure (sn) | {fmt(duration, 1)} |",
        "",
    ]


def build_report(results, config_error, tracks_dir, config_path, extra_notes):
    """Tum kameralar icin markdown raporu uretir."""
    lines = [
        "# Faz 2 - Girdi Envanteri", "",
        "Bu rapor `scripts/02_check_tracks.py` tarafindan uretildi. Script salt okur; "
        "girdi dosyalari degistirilmez.", "",
        f"- Takip dizini: `{tracks_dir}`",
        f"- Kamera config: `{config_path}`",
        "- Kapsam: her kamera bagimsiz. Kameralar arasi karsilastirma veya birlestirme "
        "yapilmadi (Faz 3 isi).",
        "",
    ]
    if config_error:
        lines += [f"> {config_error}", ""]
    for note in extra_notes:
        lines += [f"> {note}", ""]

    # ozet
    lines += [
        "## Ozet", "",
        "| kamera | tracks kaydi | benzersiz track | timestamp span (sn) | video fps (gercek/config) | uyari |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        info = res["file"]
        v = res["video"]
        span = res.get("time", {}).get("ts_span") if not res.get("empty") else None
        fps_cell = (f"{fmt(v['fps'], 2)} / "
                    f"{res['config']['fps'] if res['config']['fps'] is not None else MISSING}")
        lines.append(
            f"| {res['camera_id']} | "
            f"{info['parsed_lines'] if info['exists'] else MISSING} | "
            f"{res.get('unique_tracks', MISSING) if not res.get('empty') else MISSING} | "
            f"{fmt(span, 2)} | {fps_cell} | {len(res['warnings'])} |"
        )
    lines.append("")

    all_warnings = [w for res in results for w in res["warnings"]]
    lines += ["## Uyarilar", ""]
    if all_warnings:
        lines += [f"- {w}" for w in all_warnings]
    else:
        lines += ["Uyari uretilmedi."]
    lines.append("")

    for res in results:
        lines += render_camera(res)
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Faz 2 girdi envanteri: tracks_<cam>.jsonl semasi ve saglik kontrolu (salt okur)."
    )
    parser.add_argument("--tracks-dir", type=Path, default=DEFAULT_TRACKS_DIR,
                        help="tracks_<cam>.jsonl dosyalarinin bulundugu dizin")
    parser.add_argument("--cameras-config", type=Path, default=DEFAULT_CAMERAS_CONFIG,
                        help="kamera config dosyasi yolu")
    parser.add_argument("--max-records", type=int, default=0,
                        help="kamera basina okunacak en fazla kayit (0 = tumu; hizli deneme icin)")
    args = parser.parse_args()

    tracks_dir = args.tracks_dir if args.tracks_dir.is_absolute() else PROJECT_ROOT / args.tracks_dir
    config_path = (args.cameras_config if args.cameras_config.is_absolute()
                   else PROJECT_ROOT / args.cameras_config)

    cameras, config_error = load_cameras_config(config_path)
    extra_notes = []

    entries = {}
    order = []
    for entry in cameras:
        cam = entry.get("camera_id")
        if not cam:
            extra_notes.append(f"{MISSING}: cameras.yaml icinde camera_id'siz girdi atlandi.")
            continue
        entries[cam] = entry
        order.append(cam)

    # config'te olmayan ama diskte duran takip dosyalari da raporlanir.
    if tracks_dir.is_dir():
        for path in sorted(tracks_dir.glob("tracks_*.jsonl")):
            cam = path.stem[len("tracks_"):]
            if cam not in entries:
                entries[cam] = {}
                order.append(cam)
                extra_notes.append(
                    f"`{path.name}` diskte var ama cameras.yaml icinde girdisi {MISSING}."
                )
    else:
        extra_notes.append(f"{MISSING}: takip dizini bulunamadi: {tracks_dir}")

    if not order:
        extra_notes.append(f"{MISSING}: raporlanacak kamera bulunamadi.")

    results = [analyze_camera(cam, entries[cam], tracks_dir, args.max_records) for cam in order]

    report = build_report(results, config_error, tracks_dir, config_path, extra_notes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nRapor yazildi: {REPORT_PATH}")
    warn_count = sum(len(r["warnings"]) for r in results)
    print(f"Toplam uyari: {warn_count}")


if __name__ == "__main__":
    main()

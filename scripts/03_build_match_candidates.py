"""Faz 3: kameralar arasi handoff ADAY ciftlerini uretir.

BU SCRIPT KESIN ESLESME URETMEZ. Sadece "olabilir" diyen aday ciftleri ve
her biri icin bir guven skoru uretir. Kesin secim 03_assign_matches.py'de yapilir.

HANDOFF MANTIGI (zaman yonu):
    camA_to_camB : arac camA'dan cikip camB'ye girer.
                   kaynak = camA.exit_timestamp, hedef = camB.enter_timestamp
                   gecikme = camB_enter - camA_exit  (POZITIF olmali)
    camB_to_camA : tersi.
                   kaynak = camB.exit_timestamp, hedef = camA.enter_timestamp
                   gecikme = camA_enter - camB_exit  (POZITIF olmali)

KURALLAR:
    - iki taraf da AYNI movement_label, ve != other
    - gecikme zaman penceresi icinde: window_min < dt < window_max (saniye)
    - sinif uyumu skora katki verir (car<->car daha guvenilir)
    - bir kaynak track BIRDEN COK hedefe aday olabilir (recall-oncelikli)

ONEMLI VARSAYIM: iki kameranin saatleri senkron (clock_offset ile duzeltilebilir).
    Senkronsa dt dagilimi makul pozitif bir pencerede kumelenir; degilse sacilir.

Girdi : outputs/zones/zone_tracks_mapped_camA.csv
        outputs/zones/zone_tracks_mapped_camB.csv
Cikti : outputs/matching/match_candidates.csv
"""

import argparse
import csv
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
MATCH_DIR = PROJECT_ROOT / "outputs" / "matching"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "matching.yaml"

DIRECTIONS = ("camA_to_camB", "camB_to_camA")


def load_mapped(camera):
    path = ZONES_DIR / f"zone_tracks_mapped_{camera}.csv"
    if not path.is_file():
        raise SystemExit(f"HATA: {path} yok. Once Faz 2 zincirini calistir.")
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_features(camera):
    """track_id -> {size_norm, aspect, h, s, v}. Yoksa bos dict (ozellik opsiyonel)."""
    path = MATCH_DIR / f"track_features_{camera}.csv"
    feats = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                feats[str(r["track_id"])] = {
                    "size_norm": float(r["size_norm"]), "aspect": float(r["aspect"]),
                    "h": float(r["h"]), "s": float(r["s"]), "v": float(r["v"]),
                }
    return feats


def size_similarity(fa, fb):
    """Boyut orani (min/max) + en/boy orani benzerligi -> [0..1]."""
    sr = min(fa["size_norm"], fb["size_norm"]) / max(fa["size_norm"], fb["size_norm"], 1e-6)
    ar = min(fa["aspect"], fb["aspect"]) / max(fa["aspect"], fb["aspect"], 1e-6)
    return 0.6 * sr + 0.4 * ar


def color_similarity(fa, fb):
    """HSV benzerligi, HUE-BASKIN. Ayrim renk tonundadir; parlaklik/doygunluk
    iki kamerada da acik renkli araclarda benzer cikip ayrimi bogar, o yuzden az agirlik.
    Hue dairesel (0-180); 60 derece fark ~ tamamen farkli renk kabul edilir."""
    dh = abs(fa["h"] - fb["h"])
    dh = min(dh, 180 - dh)              # dairesel hue farki
    hue_sim = max(0.0, 1 - dh / 60.0)   # 60 = keskin esik (farkli ton -> hizli 0)
    ds = abs(fa["s"] - fb["s"]) / 255.0
    dv = abs(fa["v"] - fb["v"]) / 255.0
    sv_sim = 1 - 0.5 * (ds + dv)
    return round(0.75 * hue_sim + 0.25 * sv_sim, 4)


def moving_tracks(rows, label):
    """Belirli movement_label'a sahip (other olmayan) track'ler."""
    return [r for r in rows if r["movement_label"] == label]


def time_similarity(dt, delay_s, window_width_s):
    """Gecis penceresinin merkezine yakinligi [0..1] olarak verir.

    NOT (2026-08-06): DAR pencerede (0.05-0.70) ucgen merkez-cezasi, plateau'dan
    daha iyi (16 vs 14 rapor, ayni %100 precision) - dar bandda zaman ayrimi
    guveni artiriyor. GENIS pencerede (0.05-2.20) tersine felakete yol aciyordu
    ama genis pencere zaten recall'u dusurdugu icin (belirsizlik mevcut dogru
    eslesmeleri boluyor) terk edildi. Yavas handoff populasyonu (dt 0.9-2.0s)
    icin dogru cozum pencere degil, FIFO SIRALAMA kisiti.
    """
    center = delay_s + window_width_s / 2.0
    half_width = window_width_s / 2.0
    if half_width <= 0:
        raise ValueError("window_width_s pozitif olmali")
    return max(0.0, 1.0 - abs(dt - center) / half_width)


def predicted_delay(distance_px, adaptive):
    """ROI-mesafesinden beklenen handoff gecikmesi (sn). Bkz. configs adaptive_delay."""
    return adaptive["intercept"] + adaptive["slope"] * distance_px


def score_candidate(time_score, class_match, size_sim, color_sim):
    """Skor [0..1] = zaman + sinif + boyut + renk benzerliginin agirlikli toplami.

    time_score cagiran tarafindan hesaplanir (sabit pencere merkezine ya da
    adaptif tahmine yakinlik). Ozellik yoksa zaman+sinif'a geri dusulur.
    """
    class_score = 1.0 if class_match else 0.0
    if size_sim is None or color_sim is None:
        return round(0.7 * time_score + 0.3 * class_score, 4)
    return round(0.35 * time_score + 0.15 * class_score
                 + 0.15 * size_sim + 0.35 * color_sim, 4)


def build_candidates(rows_a, rows_b, delay_s, window_width_s, clock_offset,
                     duration_s, feats, adaptive=None, conditional=None):
    """Tum yon+zaman-tutarli aday ciftleri uretir. feats = {'camA':{...},'camB':{...}}.

    adaptive (dict, enabled=True): sabit pencere yerine her kaynak aracin
    ROI-mesafesinden gecikmeyi tahmin edip tahmin +/- tolerance_s penceresi
    kurar; time_score tahmine yakinliktan hesaplanir.
    """
    use_adaptive = bool(adaptive and adaptive.get("enabled"))
    use_conditional = bool(conditional and conditional.get("enabled")) and not use_adaptive
    cond_threshold = float((conditional or {}).get("distance_threshold_px", 0) or 0)
    cond_delay = float((conditional or {}).get("short_delay_s", 0.70))
    cond_width = float((conditional or {}).get("short_window_width_s", 1.60))
    cond_predict = (conditional or {}).get("short_prediction") or None
    cond_tol = float((conditional or {}).get("short_tolerance_s", 0.60))
    cands = []
    for label in DIRECTIONS:
        a_tracks = moving_tracks(rows_a, label)
        b_tracks = moving_tracks(rows_b, label)
        for src, dst, src_cam, dst_cam in _source_target(a_tracks, b_tracks, label):
            src_exit = float(src["exit_timestamp"])
            dst_enter_local = float(dst["enter_timestamp"])
            if duration_s is not None and (src_exit > duration_s or dst_enter_local > duration_s):
                continue
            dst_enter = dst_enter_local + clock_offset
            dt = dst_enter - src_exit

            try:
                src_dist = float(src.get("distance_px") or 0.0)
            except (TypeError, ValueError):
                src_dist = 0.0

            if use_adaptive:
                center = predicted_delay(src_dist, adaptive)
                half = float(adaptive["tolerance_s"])
            elif use_conditional and src_dist < cond_threshold:
                # Kisa mesafeli kaynak: cikis damgasi erken basilmis, arac gec
                # varir. Yalnizca BU araclar icin gec pencereye bakilir.
                # Merkez, band ortasi degil MESAFE-REGRESYONUNUN tahminidir:
                # bu alt kumede mesafe-gecikme iliskisi guclu (r=-0.87), ve
                # rekabet az oldugu icin global uygulamada basarisiz olan
                # adaptif model burada ise yarar.
                if cond_predict:
                    center = predicted_delay(src_dist, cond_predict)
                    half = cond_tol
                else:
                    center = cond_delay + cond_width / 2.0
                    half = cond_width / 2.0
            else:
                center = delay_s + window_width_s / 2.0
                half = window_width_s / 2.0
            lo, hi = center - half, center + half
            if not (lo <= dt <= hi):
                continue
            tscore = max(0.0, 1.0 - abs(dt - center) / half) if half > 0 else 0.0

            class_match = src["class"] == dst["class"]
            fa = feats.get(src_cam, {}).get(str(src["track_id"]))
            fb = feats.get(dst_cam, {}).get(str(dst["track_id"]))
            if fa and fb:
                ssim = round(size_similarity(fa, fb), 4)
                csim = round(color_similarity(fa, fb), 4)
            else:
                ssim = csim = None
            cands.append({
                "movement_label": label,
                "src_camera": src_cam, "src_track": src["track_id"], "src_class": src["class"],
                "dst_camera": dst_cam, "dst_track": dst["track_id"], "dst_class": dst["class"],
                "src_exit_t": round(src_exit, 3),
                "dst_enter_t": round(dst_enter, 3),
                "delta_t": round(dt, 3),
                "time_score": round(tscore, 4),
                "class_match": int(class_match),
                "size_sim": ssim if ssim is not None else "",
                "color_sim": csim if csim is not None else "",
                "score": score_candidate(tscore, class_match, ssim, csim),
            })
    cands.sort(key=lambda c: (-c["score"], c["delta_t"]))
    return cands


def _source_target(a_tracks, b_tracks, label):
    """Yone gore kaynak/hedef kamerayi belirler.

    camA_to_camB: kaynak camA (cikan), hedef camB (giren)
    camB_to_camA: kaynak camB (cikan), hedef camA (giren)
    """
    if label == "camA_to_camB":
        for src in a_tracks:
            for dst in b_tracks:
                yield src, dst, "camA", "camB"
    else:  # camB_to_camA
        for src in b_tracks:
            for dst in a_tracks:
                yield src, dst, "camB", "camA"


def load_config(path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main():
    ap = argparse.ArgumentParser(description="Faz 3: kameralar arasi handoff aday ciftleri.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--delay", type=float, default=None,
                    help="Config'teki delay_s icin gecici komut satiri degeri")
    ap.add_argument("--duration", type=float, default=None,
                    help="Yalnizca videonun ilk N saniyesini kullan")
    ap.add_argument("--clock-offset", type=float, default=None,
                    help="camB zamanina eklenecek senkron duzeltmesi (sn)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    timing = cfg.get("timing") or {}
    delay_s = args.delay if args.delay is not None else float(timing.get("delay_s", 0.7))
    window_width_s = float(timing.get("window_width_s", 0.8))
    clock_offset = (args.clock_offset if args.clock_offset is not None
                    else float(timing.get("clock_offset_s", 0.0)))
    duration_s = (args.duration if args.duration is not None
                  else timing.get("analysis_duration_s"))
    duration_s = float(duration_s) if duration_s is not None else None

    rows_a = load_mapped("camA")
    rows_b = load_mapped("camB")
    feats = {"camA": load_features("camA"), "camB": load_features("camB")}
    n_feat = len(feats["camA"]) + len(feats["camB"])
    print(f"Gorsel ozellik yuklendi: {n_feat} track (boyut+renk)" if n_feat else
          "UYARI: gorsel ozellik yok; sadece zaman+sinif kullanilacak.")
    adaptive = cfg.get("adaptive_delay") or {}
    conditional = cfg.get("conditional_window") or {}
    if conditional.get("enabled") and not adaptive.get("enabled"):
        print(f"Kosullu pencere: mesafe < {conditional['distance_threshold_px']}px olan "
              f"kaynaklar {conditional['short_delay_s']:.2f}-"
              f"{conditional['short_delay_s'] + conditional['short_window_width_s']:.2f} sn")
    cands = build_candidates(rows_a, rows_b, delay_s, window_width_s,
                             clock_offset, duration_s, feats, adaptive, conditional)

    out_path = args.output or (MATCH_DIR / "match_candidates.csv")
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["movement_label", "src_camera", "src_track", "src_class",
              "dst_camera", "dst_track", "dst_class", "src_exit_t", "dst_enter_t",
              "delta_t", "time_score", "class_match", "size_sim", "color_sim", "score"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(cands)

    # Ozet + varsayim dogrulama sinyalleri
    from collections import Counter
    by_dir = Counter(c["movement_label"] for c in cands)
    n_src = len({(c["src_camera"], c["src_track"]) for c in cands})
    print(f"Yazildi: {out_path}")
    print(f"Ayar: delay={delay_s:.3f} sn | pencere={delay_s:.3f}-{delay_s + window_width_s:.3f} sn "
          f"| clock_offset={clock_offset:+.3f} sn | sure={duration_s or 'tam video'}")
    print(f"Toplam aday cift: {len(cands)}")
    print(f"  yon dagilimi: {dict(by_dir)}")
    print(f"  benzersiz kaynak track: {n_src} (her biri ortalama {len(cands)/max(n_src,1):.1f} adaya baglaniyor)")
    if cands:
        dts = [c["delta_t"] for c in cands]
        dts.sort()
        print(f"  delta_t: min={dts[0]:.1f} medyan={dts[len(dts)//2]:.1f} max={dts[-1]:.1f} sn")


if __name__ == "__main__":
    main()

"""Faz 1.5: tek kamera ici track dikisleme (track stitching).

PROBLEM:
    Yogun trafikte bir arac ortulunce (occlusion) ByteTrack ID'sini kaybeder;
    tekrar gorununce YENI bir track_id alir. Boylece tek bir fiziksel arac
    birden cok track parcasina bolunur (fragmentation).

COZUM (Re-ID/embedding YOK; sadece geometri + zaman):
    A parcasi biter, kisa sure sonra B parcasi baslar. Eger A'nin son hiziyla
    tahmin edilen konum B'nin baslangicina yakinsa ve sinif ayniysa, bunlar
    ayni fiziksel arac kabul edilir ve ortak bir global_id alir.

TASARIM KARARLARI (bkz. devir notlari):
    - Sadece HAREKETLI araclar (min_speed) birlestirilir; duran araclar Faz 3'u
      etkilemez ve yan yana durduklarinda karisma riski yuksektir.
    - Aday secimi motion-aware'dir (naif yakinlik degil): tahmin hatasi kucuk olmali.
    - Greedy eslestirme + Union-Find ile zincirler tek global_id'de toplanir.
    - Cikti downstream'i bozmaz: track_id alani global_id ile degistirilir,
      orijinal deger orig_track_id olarak saklanir.

Girdi : outputs/tracks/tracks_<camera>.jsonl
Cikti : outputs/tracks/tracks_stitched_<camera>.jsonl
        outputs/tracks/stitch_map_<camera>.csv   (orig_track_id -> global_id)
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _segment_velocity(recs, head):
    """Track'in bas (head) veya son (tail) ~5 karesinden hiz vektoru."""
    seg = recs[:5] if head else recs[-5:]
    if len(seg) < 2:
        return (0.0, 0.0)
    df = seg[-1]["frame"] - seg[0]["frame"]
    if df == 0:
        return (0.0, 0.0)
    vx = (seg[-1]["foot_point"][0] - seg[0]["foot_point"][0]) / df
    vy = (seg[-1]["foot_point"][1] - seg[0]["foot_point"][1]) / df
    return (vx, vy)


def summarize_tracks(records_by_tid):
    """Her track icin ozet: ilk/son kare-konum, giris VE cikis hiz vektoru, sinif.

    v_in  = track'in ilk karelerindeki hiz (B adayi icin: nasil basladi)
    v_out = track'in son karelerindeki hiz (A adayi icin: nereye gidiyordu)
    """
    summary = {}
    for tid, recs in records_by_tid.items():
        recs.sort(key=lambda r: r["frame"])
        v_out = _segment_velocity(recs, head=False)
        v_in = _segment_velocity(recs, head=True)
        summary[tid] = {
            "tid": tid,
            "f0": recs[0]["frame"], "f1": recs[-1]["frame"],
            "p0": tuple(recs[0]["foot_point"]), "p1": tuple(recs[-1]["foot_point"]),
            "v_out": v_out, "v_in": v_in,
            "speed_out": math.hypot(*v_out), "speed_in": math.hypot(*v_in),
            "cls": recs[0]["class"], "n": len(recs),
        }
    return summary


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def build_candidates(summary, gap_frames, max_predict_dist, min_speed,
                     max_jump, speed_ratio):
    """Precision-oncelikli aday ciftler: (hata, A_tid, B_tid).

    Guvenlik kontrolleri (yanlis birlestirmeyi onlemek icin):
      1. A cikis hizi >= min_speed  (hareketli A)
      2. B giris hizi >= min_speed  (hareketli B; duran araca baglanma)
      3. Yon uyumu: v_out(A) . v_in(B) > 0  (ayni yone gidiyorlar)
      4. Hiz benzerligi: hizlar birbirine yakin (ani hizlanma/yavaslama supheli)
      5. Tahmin hatasi <= max_predict_dist
      6. Ham atlama <= max_jump  (dev sicramalari reddet; belirsizlik cok yuksek)
      7. 0 < gap <= gap_frames, ayni sinif
    """
    cands = []
    tracks = list(summary.values())
    for a in tracks:
        if a["speed_out"] < min_speed:
            continue
        for b in tracks:
            if a["tid"] == b["tid"] or a["cls"] != b["cls"]:
                continue
            if b["speed_in"] < min_speed:
                continue
            gap = b["f0"] - a["f1"]
            if not (0 < gap <= gap_frames):
                continue
            # 3) yon uyumu
            dot = a["v_out"][0] * b["v_in"][0] + a["v_out"][1] * b["v_in"][1]
            if dot <= 0:
                continue
            # 4) hiz benzerligi
            ratio = b["speed_in"] / a["speed_out"] if a["speed_out"] > 0 else 999
            if not (1.0 / speed_ratio <= ratio <= speed_ratio):
                continue
            # 6) ham atlama siniri
            jump = math.hypot(b["p0"][0] - a["p1"][0], b["p0"][1] - a["p1"][1])
            if jump > max_jump:
                continue
            # 5) tahmin hatasi
            px = a["p1"][0] + a["v_out"][0] * gap
            py = a["p1"][1] + a["v_out"][1] * gap
            err = math.hypot(b["p0"][0] - px, b["p0"][1] - py)
            if err <= max_predict_dist:
                cands.append((err, a["tid"], b["tid"]))
    return cands


def stitch(summary, gap_frames, max_predict_dist, min_speed, max_jump, speed_ratio):
    """Greedy + Union-Find. Her parca en fazla bir kez oncesine/sonrasina baglanir."""
    parent = {tid: tid for tid in summary}
    cands = sorted(build_candidates(summary, gap_frames, max_predict_dist, min_speed,
                                    max_jump, speed_ratio))
    used_as_next = set()   # bir B en fazla bir A'ya baglanir
    used_as_prev = set()   # bir A en fazla bir B'ye baglanir
    merges = 0
    for err, a, b in cands:
        if a in used_as_prev or b in used_as_next:
            continue
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[rb] = ra
            used_as_prev.add(a)
            used_as_next.add(b)
            merges += 1
    # global_id = kokun temsilcisi; okunur kucuk sayilara yeniden numaralandir
    roots = {}
    global_of = {}
    next_gid = 1
    for tid in sorted(summary):
        r = find(parent, tid)
        if r not in roots:
            roots[r] = next_gid
            next_gid += 1
        global_of[tid] = roots[r]
    return global_of, merges


def main():
    ap = argparse.ArgumentParser(description="Tek kamera ici track stitching (Faz 1.5).")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--gap-frames", type=int, default=45,
                    help="A bittikten sonra B'nin baslamasi icin izin verilen en fazla kare bosluk (1.5sn)")
    ap.add_argument("--max-predict-dist", type=float, default=50.0,
                    help="A'nin tahmini konumu ile B baslangici arasi izin verilen en fazla piksel")
    ap.add_argument("--min-speed", type=float, default=3.0,
                    help="Hem A hem B bu piksel/kare hizin uzerinde olmali (duran araclara baglanma)")
    ap.add_argument("--max-jump", type=float, default=200.0,
                    help="A son konumu ile B baslangici arasi izin verilen en fazla ham atlama (px)")
    ap.add_argument("--speed-ratio", type=float, default=2.5,
                    help="B/A hiz orani bu araligin (1/r .. r) icinde olmali")
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    in_path = args.input or (TRACKS_DIR / f"tracks_{args.camera}.jsonl")
    out_path = args.output or (TRACKS_DIR / f"tracks_stitched_{args.camera}.jsonl")
    map_path = TRACKS_DIR / f"stitch_map_{args.camera}.csv"
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.is_file():
        raise SystemExit(f"HATA: girdi yok: {in_path}")

    records_by_tid = defaultdict(list)
    all_records = []
    for rec in iter_jsonl(in_path):
        records_by_tid[int(rec["track_id"])].append(rec)
        all_records.append(rec)

    summary = summarize_tracks(records_by_tid)
    global_of, merges = stitch(summary, args.gap_frames, args.max_predict_dist,
                               args.min_speed, args.max_jump, args.speed_ratio)

    n_orig = len(summary)
    n_global = len(set(global_of.values()))
    print(f"Kamera {args.camera}: {n_orig} track -> {n_global} global_id  (birlesme: {merges})")
    print(f"  Fragmentation azalmasi: {n_orig - n_global} parca birlestirildi "
          f"({100*(n_orig-n_global)//n_orig}%)")

    # Cikti jsonl: track_id = global_id, orig_track_id saklanir
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in all_records:
            orig = int(rec["track_id"])
            rec = dict(rec)
            rec["orig_track_id"] = orig
            rec["track_id"] = global_of[orig]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Yazildi: {out_path}")

    # Harita CSV (denetim icin)
    with open(map_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["orig_track_id", "global_id", "class", "num_points", "speed_out_px_f"])
        for tid in sorted(summary):
            s = summary[tid]
            w.writerow([tid, global_of[tid], s["cls"], s["n"], round(s["speed_out"], 2)])
    print(f"Yazildi: {map_path}")


if __name__ == "__main__":
    main()

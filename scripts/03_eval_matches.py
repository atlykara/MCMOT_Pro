"""Faz 3 eslestirme basarimini dogrulama setine (ground truth) gore olcer.

KRITIK KURAL: confidence='manual' satirlari OTOMATIK basarima dahil edilmez.
Bu satirlar insan tarafindan girilen CEVAPLARDIR; bunlari basari olarak saymak
kendi verdigimiz cevabi kendimize dogrulatmak (self-validation) olur. Manuel
ciftler yalnizca:
  - recall PAYDASINA girer (otomatik hattin bulmasi GEREKEN gercek ciftlerdir),
  - ayri bir "sistem toplami" satirinda raporlanir.

Kullanim:
    python3 scripts/03_eval_matches.py
    python3 scripts/03_eval_matches.py --window 0-30     # sadece GT penceresi
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCH_DIR = PROJECT_ROOT / "outputs" / "matching"
GT_PATH = MATCH_DIR / "ground_truth.csv"

REPORTED = ("high", "medium")


def load_gt():
    """Doner: (dogru_ciftler, yanlis_ciftler) — her biri {(src,dst)} kumesi."""
    true_pairs, false_pairs = set(), set()
    if not GT_PATH.is_file():
        raise SystemExit(f"HATA: dogrulama seti yok: {GT_PATH}")
    with GT_PATH.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["src"].strip(), row["dst"].strip())
            (true_pairs if row["label"].strip().upper() == "T" else false_pairs).add(key)
    return true_pairs, false_pairs


def load_matches():
    rows = []
    for name in ("matches.csv", "matches_ambiguous.csv"):
        path = MATCH_DIR / name
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    return rows


def key_of(row):
    return (f"{row['src_camera']}#{row['src_track']}",
            f"{row['dst_camera']}#{row['dst_track']}")


def in_window(row, lo, hi):
    if lo is None:
        return True
    try:
        t = float(row["src_exit_t"])
    except (TypeError, ValueError):
        return True
    return lo <= t <= hi


def main():
    ap = argparse.ArgumentParser(description="Eslestirme basarimini GT'ye gore olcer.")
    ap.add_argument("--window", default=None,
                    help="Yalnizca bu zaman araligindaki kaynaklar, or. 0-30")
    args = ap.parse_args()
    lo = hi = None
    if args.window:
        lo, hi = (float(x) for x in args.window.split("-"))

    true_pairs, false_pairs = load_gt()
    all_rows = [r for r in load_matches() if in_window(r, lo, hi)]

    auto = [r for r in all_rows if r["confidence"] != "manual"]
    manual = [r for r in all_rows if r["confidence"] == "manual"]
    manual_keys = {key_of(r) for r in manual}

    auto_keys = {key_of(r) for r in auto}
    reported_keys = {key_of(r) for r in auto if r["confidence"] in REPORTED}

    # --- Kesinlik: yalnizca OTOMATIK uretilen, GT'de etiketli ciftler uzerinden
    tier = Counter()
    hit = Counter()
    for r in auto:
        k = key_of(r)
        if k in true_pairs or k in false_pairs:      # yalnizca etiketli olanlar
            tier[r["confidence"]] += 1
            if k in true_pairs:
                hit[r["confidence"]] += 1

    print("=" * 66)
    print("OTOMATIK HAT BASARIMI (manuel enjeksiyonlar haric)")
    print("=" * 66)
    print(f"{'katman':10}{'eslesme':>9}{'dogru':>8}{'kesinlik':>11}")
    for conf in ("high", "medium", "low"):
        if tier[conf]:
            print(f"{conf:10}{tier[conf]:>9}{hit[conf]:>8}{100*hit[conf]/tier[conf]:>10.0f}%")
    rep_n = tier["high"] + tier["medium"]
    rep_ok = hit["high"] + hit["medium"]
    if rep_n:
        print(f"{'RAPOR':10}{rep_n:>9}{rep_ok:>8}{100*rep_ok/rep_n:>10.0f}%   <- sistemin cikti kalitesi")

    # --- Kapsama: paydada TUM gercek ciftler (manuel olanlar dahil, cunku
    #     otomatik hattin ideal olarak onlari da bulmasi gerekir)
    gt_in_window = true_pairs
    if lo is not None:
        gt_in_window = {k for k in true_pairs
                        if any(key_of(r) == k for r in all_rows)} or true_pairs
    found_auto = gt_in_window & auto_keys
    found_rep = gt_in_window & reported_keys
    only_manual = gt_in_window & manual_keys - auto_keys

    print()
    print("=" * 66)
    print(f"KAPSAMA (payda: {len(gt_in_window)} gercek cift)")
    print("=" * 66)
    print(f"  otomatik hat bulmus (herhangi katman): {len(found_auto):>3}"
          f"  = {100*len(found_auto)//max(len(gt_in_window),1):>3}%")
    print(f"  otomatik hat raporlamis (high+med)   : {len(found_rep):>3}"
          f"  = {100*len(found_rep)//max(len(gt_in_window),1):>3}%")
    print(f"  YALNIZCA elle girilmis (otomatik kacirdi): {len(only_manual):>3}")
    missing = gt_in_window - auto_keys - manual_keys
    print(f"  hic bulunamamis (ne otomatik ne manuel)  : {len(missing):>3}")
    if missing:
        for k in sorted(missing)[:8]:
            print(f"      {k[0]} -> {k[1]}")

    # --- Sistem toplami (insan onayi dahil operasyonel cikti)
    print()
    print("=" * 66)
    print("SISTEM TOPLAMI (otomatik + insan onayi)")
    print("=" * 66)
    total_rep = len([r for r in auto if r["confidence"] in REPORTED]) + len(manual)
    covered = len(found_rep | only_manual)
    print(f"  raporlanan eslesme  : {total_rep}"
          f"  ({len([r for r in auto if r['confidence'] in REPORTED])} otomatik"
          f" + {len(manual)} elle)")
    print(f"  GT kapsamasi        : {covered}/{len(gt_in_window)}"
          f" = {100*covered//max(len(gt_in_window),1)}%")


if __name__ == "__main__":
    main()

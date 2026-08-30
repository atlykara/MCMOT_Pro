"""Faz 3: aday ciftlerden bire-bir handoff eslesmeleri ve GUVEN skoru uretir.

BU SCRIPT DE "KESIN KIMLIK" URETMEZ. Her eslesmeye bir guven seviyesi verir ve
belirsiz olanlarin alternatiflerini saklar. Amac: dogru oldugundan emin oldugumuz
yerlerde eslestir, emin olmadigimiz yerlerde belirsizligi durustce raporla.

GUVEN = pairwise skor + CIFT TARAFLI MARGIN.
    margin = kaynak ve hedef taraflarindaki en iyi/ikinci skor farkinin kucugu.
    - rakipsiz / net ayrisan (yuksek margin) -> yuksek guven (or. bus/truck)
    - birden cok yakin rakip (dusuk margin)  -> dusuk guven (or. car)

SECIM: Jonker-Volgenant ile toplam skoru en yuksek bire-bir atama.
    Bir arac bir kez gecer -> her track en fazla bir kez.

Girdi : outputs/matching/match_candidates.csv
Cikti : outputs/matching/matches.csv            (secilen bire-bir eslesmeler)
        outputs/matching/matches_ambiguous.csv  (dusuk guvenli/belirsiz olanlar)
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

try:
    import lap
except ImportError:  # Test/gelistirme ortami icin kontrollu geri dusus.
    lap = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCH_DIR = PROJECT_ROOT / "outputs" / "matching"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "matching.yaml"


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["score"] = float(r["score"])
        r["delta_t"] = float(r["delta_t"])
        r["class_match"] = int(r["class_match"])
        r["color_sim"] = float(r["color_sim"]) if r.get("color_sim") not in (None, "") else None
        r["size_sim"] = float(r["size_sim"]) if r.get("size_sim") not in (None, "") else None
    return rows


def compute_margins(cands):
    """Her kaynak ve hedef track icin en iyi/ikinci skor farki."""
    by_src = defaultdict(list)
    by_dst = defaultdict(list)
    for c in cands:
        by_src[(c["src_camera"], c["src_track"])].append(c)
        by_dst[(c["dst_camera"], c["dst_track"])].append(c)

    def margins(groups):
        result = {}
        for key, lst in groups.items():
            scores = sorted((c["score"] for c in lst), reverse=True)
            second = scores[1] if len(scores) > 1 else 0.0
            result[key] = scores[0] - second
        return result

    return by_src, margins(by_src), margins(by_dst)


def _greedy_pairs(cands, min_score):
    """lap paketi yoksa kullanilan deterministik geri dusus."""
    used_src, used_dst = set(), set()
    chosen = []
    for c in sorted(cands, key=lambda row: -row["score"]):
        src = (c["src_camera"], c["src_track"])
        dst = (c["dst_camera"], c["dst_track"])
        if c["score"] >= min_score and src not in used_src and dst not in used_dst:
            chosen.append(c)
            used_src.add(src)
            used_dst.add(dst)
    return chosen


def mutual_best_lock(cands):
    """Karsilikli-en-iyi ciftleri kilitle (kararli-eslestirme / Gale-Shapley mantigi).

    src'nin en iyi adayi d VE d'nin en iyi adayi ayni src ise -> kilitle.
    Amac: en guclu cift, global TOPLAM-skor optimizasyonu ugruna feda edilmesin.
    Ornek (GT M10/M11 swap'i): 85->141 karsilikli-en-iyi (0.80); onceden
    kilitlenirse optimizer onu 85->143 + 133->141'e bolemez.
    Iteratif: her kilit turundan sonra kalanlarda tekrar bak.
    """
    locked = []
    remaining = list(cands)
    while True:
        by_src, by_dst = defaultdict(list), defaultdict(list)
        for c in remaining:
            by_src[(c["src_camera"], c["src_track"])].append(c)
            by_dst[(c["dst_camera"], c["dst_track"])].append(c)
        best_src = {k: max(v, key=lambda c: c["score"]) for k, v in by_src.items()}
        best_dst = {k: max(v, key=lambda c: c["score"]) for k, v in by_dst.items()}
        used_s, used_d, newly = set(), set(), []
        for s, c in best_src.items():
            d = (c["dst_camera"], c["dst_track"])
            if best_dst.get(d) is c and s not in used_s and d not in used_d:
                newly.append(c)
                used_s.add(s)
                used_d.add(d)
        if not newly:
            break
        for c in newly:
            c["_locked"] = True   # guven yukseltmesi icin isaretle
        locked.extend(newly)
        remaining = [c for c in remaining
                     if (c["src_camera"], c["src_track"]) not in used_s
                     and (c["dst_camera"], c["dst_track"]) not in used_d]
    return locked, remaining


def monotonic_pairs(cands):
    """FIFO sira-koruyan (non-crossing) maksimum-skor bire-bir atama, yon basina DP.

    Kaynaklar cikis zamanina, hedefler giris zamanina gore siralanir. Eslesmeler
    caprazlanamaz: s_i, d_j eslesip s_i'>s_i, d_j' eslesirse d_j'>d_j olmali.
    Bu, gorunumden bagimsiz guclu bir yapisal kisit (camA'dan k. cikan araba
    camB'ye k. giren araba). DP: f[i][j]=max(f[i-1][j], f[i][j-1],
    f[i-1][j-1]+score(i,j)). Yavas populasyon + beyaz-arac belirsizligi icin
    zaman/renk'in cozemedigi sirayi kullanir.
    """
    chosen = []
    by_direction = defaultdict(list)
    for c in cands:
        by_direction[c["movement_label"]].append(c)
    for dcands in by_direction.values():
        src_t, dst_t = {}, {}
        for c in dcands:
            src_t[(c["src_camera"], c["src_track"])] = float(c["src_exit_t"])
            dst_t[(c["dst_camera"], c["dst_track"])] = float(c["dst_enter_t"])
        srcs = sorted(src_t, key=lambda k: src_t[k])
        dsts = sorted(dst_t, key=lambda k: dst_t[k])
        si = {k: i for i, k in enumerate(srcs)}
        di = {k: i for i, k in enumerate(dsts)}
        best = {}
        for c in dcands:
            i = si[(c["src_camera"], c["src_track"])]
            j = di[(c["dst_camera"], c["dst_track"])]
            if (i, j) not in best or c["score"] > best[(i, j)]["score"]:
                best[(i, j)] = c
        n, m = len(srcs), len(dsts)
        f = [[0.0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if f[i - 1][j] >= f[i][j - 1]:
                    f[i][j], bt[i][j] = f[i - 1][j], "up"
                else:
                    f[i][j], bt[i][j] = f[i][j - 1], "left"
                c = best.get((i - 1, j - 1))
                if c is not None and f[i - 1][j - 1] + c["score"] > f[i][j]:
                    f[i][j], bt[i][j] = f[i - 1][j - 1] + c["score"], ("match", c)
        i, j = n, m
        while i > 0 and j > 0:
            step = bt[i][j]
            if isinstance(step, tuple):
                chosen.append(step[1]); i -= 1; j -= 1
            elif step == "up":
                i -= 1
            else:
                j -= 1
    return chosen


def optimal_pairs(cands, min_score, mode="mutual_best"):
    """Atama modu:
    - mutual_best: karsilikli-en-iyi kilit + kalanda toplam-skor optimizasyonu (lapjv)
    - fifo: karsilikli-en-iyi kilit + kalanda FIFO sira-koruyan DP
    - fifo_pure: sadece FIFO sira-koruyan DP (kilit yok)
    """
    eligible = [c for c in cands if c["score"] >= min_score]
    if mode == "fifo_pure":
        return monotonic_pairs(eligible)

    locked, remaining = mutual_best_lock(eligible)
    if mode == "fifo":
        return locked + monotonic_pairs(remaining)

    if lap is None:
        return locked + _greedy_pairs(remaining, min_score)

    chosen = list(locked)
    by_direction = defaultdict(list)
    for c in remaining:
        by_direction[c["movement_label"]].append(c)

    for direction_cands in by_direction.values():
        srcs = sorted({(c["src_camera"], c["src_track"]) for c in direction_cands})
        dsts = sorted({(c["dst_camera"], c["dst_track"]) for c in direction_cands})
        src_index = {key: i for i, key in enumerate(srcs)}
        dst_index = {key: i for i, key in enumerate(dsts)}
        costs = np.full((len(srcs), len(dsts)), 1e6, dtype=float)
        candidate_of = {}

        for c in direction_cands:
            src = (c["src_camera"], c["src_track"])
            dst = (c["dst_camera"], c["dst_track"])
            i, j = src_index[src], dst_index[dst]
            cost = 1.0 - c["score"]
            if cost < costs[i, j]:
                costs[i, j] = cost
                candidate_of[(i, j)] = c

        _, assignment, _ = lap.lapjv(
            costs,
            extend_cost=True,
            cost_limit=(1.0 - min_score) + 1e-9,
        )
        for i, j in enumerate(assignment):
            if j >= 0 and (i, int(j)) in candidate_of:
                chosen.append(candidate_of[(i, int(j))])
    return chosen


def confidence_level(candidate, margin, unique=False):
    """Skor, cift tarafli margin ve temel tutarliliklardan guven uretir.

    unique=True: bu cift hem kaynagi hem hedefi icin TEK secenek (mutual-unique).
    Kafa karistiracak rakip olmadigindan, fiziksel kapilari geciyorsa yuksek
    guven verilir; time_score'un merkeze uzakligi burada cezalandirilmaz.
    """
    time_score = float(candidate.get("time_score") or 0.0)
    # NOT: size_sim ARTIK SERT VETO DEGIL. Iki kamera koridora cok farkli
    # aci/olcekten baktigi icin ayni aracin bbox boyutu dogal olarak farkli
    # gorunur; boyut sert kapi yapilinca gercek eslesmeler de low'a duserdi.
    # Boyut sinyali skora yumusak katki olarak zaten giriyor (score_candidate).
    # Fiziksel kapilar (sinif uyumu, zaman-band) korunuyor.
    if not candidate["class_match"] or time_score <= 0.0:
        return "low"
    if unique:
        # Rakipsiz cift: zamansal belirsizlik yok. Ama HIGH demek icin gorunum
        # dogrulamasi da istiyoruz. color_sim filtreden gecmisse (>=color_min)
        # renk teyit edilmis sayilir -> high. color_sim EKSIKSE (ozellik
        # cikarilmamis) dogrulanamaz -> medium'da birak (M000012 gibi koyu<->beyaz
        # yanlis eslesmeler sirf zamanla high'a cikmasin).
        if candidate.get("color_sim") is None:
            return "medium"
        return "high" if candidate["score"] >= 0.55 else "medium"
    score = candidate["score"]
    if score >= 0.80 and margin >= 0.20:
        return "high"
    if score >= 0.65 and margin >= 0.10:
        return "medium"
    return "low"


MANUAL_PATH = PROJECT_ROOT / "configs" / "manual_matches.csv"


def load_manual_matches(path=MANUAL_PATH):
    """Insan onayli eslesmeleri okur (yorum satirlari '#' ile atlanir).

    Doner: [{src_camera, src_track, dst list[(cam, track)], note}]
    Ayni src birden fazla satirda gecerse hedefler tek bir alias grubunda
    toplanir; bu, takip kimligi degisimi (id-switch) olan araclar icindir:
    camB#186 ve camB#202 ayni araca ait iki farkli track kimligidir.
    """
    if not path.is_file():
        return []
    groups = {}
    order = []
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if not row.get("src_track"):
            continue
        key = (row["src_camera"].strip(), row["src_track"].strip())
        if key not in groups:
            groups[key] = {"src_camera": key[0], "src_track": key[1],
                           "dsts": [], "note": (row.get("note") or "").strip()}
            order.append(key)
        groups[key]["dsts"].append((row["dst_camera"].strip(), row["dst_track"].strip()))
    return [groups[k] for k in order]


def apply_manual_matches(matches, manual, mapped):
    """Elle onayli ciftleri ciktiya enjekte eder; cakisan otomatik ciftleri duser.

    Insan onayi otomatik atamayi ezer: manuel listede gecen bir track baska bir
    otomatik eslesmede kullanilmissa o otomatik eslesme kaldirilir.
    """
    if not manual:
        return matches, []

    # Otomatik hat ayni cifti zaten dogru uretmisse manuel enjeksiyona gerek
    # yok; boylece hattaki gercek iyilesme manuel satirlarin altinda gizlenmez.
    auto_pairs = {(m["src_camera"], str(m["src_track"]),
                   m["dst_camera"], str(m["dst_track"])) for m in matches}
    pending = []
    for g in manual:
        dsts = [d for d in g["dsts"]
                if (g["src_camera"], g["src_track"], d[0], d[1]) not in auto_pairs]
        if dsts:
            pending.append({**g, "dsts": dsts})
    manual = pending
    if not manual:
        return matches, []

    claimed = set()
    for g in manual:
        claimed.add((g["src_camera"], g["src_track"]))
        claimed.update(g["dsts"])

    kept, dropped = [], []
    for m in matches:
        src = (m["src_camera"], str(m["src_track"]))
        dst = (m["dst_camera"], str(m["dst_track"]))
        (dropped if (src in claimed or dst in claimed) else kept).append(m)

    injected = []
    for index, g in enumerate(manual, start=1):
        src_row = mapped.get((g["src_camera"], g["src_track"])) or {}
        for alias_no, (dcam, dtrack) in enumerate(g["dsts"]):
            dst_row = mapped.get((dcam, dtrack)) or {}
            try:
                exit_t = float(src_row.get("exit_timestamp") or 0.0)
                enter_t = float(dst_row.get("enter_timestamp") or 0.0)
                delta = round(enter_t - exit_t, 3)
            except (TypeError, ValueError):
                exit_t = enter_t = delta = ""
            injected.append({
                "match_id": f"MAN{index:04d}",
                "movement_label": src_row.get("movement_label", ""),
                "src_camera": g["src_camera"], "src_track": g["src_track"],
                "src_class": src_row.get("class", ""),
                "dst_camera": dcam, "dst_track": dtrack,
                "dst_class": dst_row.get("class", ""),
                "src_exit_t": exit_t, "dst_enter_t": enter_t,
                "delta_t": delta, "time_score": "",
                "score": "", "margin": "",
                "confidence": "manual",
                "n_candidates": "",
                "alternatives": ("alias" if alias_no else "") ,
            })
    return kept + injected, dropped


def load_mapped_rows():
    """zone_tracks_mapped_<cam>.csv satirlarini (kamera, track) anahtariyla verir."""
    result = {}
    zones_dir = PROJECT_ROOT / "outputs" / "zones"
    for cam in ("camA", "camB"):
        path = zones_dir / f"zone_tracks_mapped_{cam}.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                result[(cam, row["track_id"])] = row
    return result


def assign(cands, min_score, color_min, mode="mutual_best"):
    """Optimal bire-bir eslestirme + cift tarafli guven marji."""
    eligible = [
        c for c in cands
        if c["score"] >= min_score
        and (c["color_sim"] is None or c["color_sim"] >= color_min)
    ]
    by_src, src_margin, dst_margin = compute_margins(eligible)
    by_dst = defaultdict(list)
    for c in eligible:
        by_dst[(c["dst_camera"], c["dst_track"])].append(c)
    selected = optimal_pairs(eligible, min_score, mode)
    selected.sort(key=lambda c: (c["src_exit_t"], c["src_camera"], c["src_track"]))
    matches = []
    for index, c in enumerate(selected, start=1):
        src = (c["src_camera"], c["src_track"])
        dst = (c["dst_camera"], c["dst_track"])
        margin = min(src_margin[src], dst_margin[dst])
        # mutual-unique: kaynagin tek adayi VE hedefin tek adayi bu cift.
        # NOT: mutual-best-locked ciftleri de buraya dahil etmeyi denedik ama
        # seyrek grafikde cogu cift mutual-best oldugu icin herkesi high'a
        # terfi ettirip precision'i %100->%83'e dusurdu. Bu yuzden kilit
        # yalnizca ATAMAYI duzeltir (swap'i cozer), guveni yukseltmez.
        unique = len(by_src[src]) == 1 and len(by_dst[dst]) == 1
        # bu kaynagin alternatifleri (secilen disindaki adaylar)
        alts = [a for a in by_src[src] if a is not c]
        alts.sort(key=lambda a: -a["score"])
        alt_str = ";".join(f"{a['dst_camera']}#{a['dst_track']}({a['score']:.2f})" for a in alts[:3])
        matches.append({
            "match_id": f"M{index:06d}",
            "movement_label": c["movement_label"],
            "src_camera": c["src_camera"], "src_track": c["src_track"], "src_class": c["src_class"],
            "dst_camera": c["dst_camera"], "dst_track": c["dst_track"], "dst_class": c["dst_class"],
            "src_exit_t": c["src_exit_t"], "dst_enter_t": c["dst_enter_t"],
            "delta_t": c["delta_t"], "time_score": c.get("time_score", ""),
            "score": round(c["score"], 4),
            "margin": round(margin, 4),
            "confidence": confidence_level(c, margin, unique),
            "n_candidates": len(by_src[src]),
            "alternatives": alt_str,
        })
    return matches


def load_config(path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main():
    ap = argparse.ArgumentParser(description="Faz 3: bire-bir handoff eslesmeleri + guven.")
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--min-score", type=float, default=None,
                    help="Bu skorun altindaki adaylar hic eslestirilmez (gurultu esigi)")
    ap.add_argument("--color-min", type=float, default=None,
                    help="'Esi yok' esigi: renk benzerligi bunun altindaysa eslestirme (kesin farkli arac)")
    args = ap.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    _cfg = load_config(config_path)
    scoring = (_cfg.get("scoring") or {})
    min_score = args.min_score if args.min_score is not None else float(scoring.get("min_score", 0.3))
    color_min = args.color_min if args.color_min is not None else float(scoring.get("color_min", 0.45))
    mode = str(scoring.get("assignment_mode", "mutual_best"))

    in_path = args.input or (MATCH_DIR / "match_candidates.csv")
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.is_file():
        raise SystemExit(f"HATA: {in_path} yok. Once 03_build_match_candidates.py calistir.")

    cands = load_candidates(in_path)
    matches = assign(cands, min_score, color_min, mode)

    # Insan onayli eslesmeleri enjekte et (otomatik hattin uretemedikleri).
    manual = load_manual_matches()
    dropped = []
    if manual:
        matches, dropped = apply_manual_matches(matches, manual, load_mapped_rows())
        matches.sort(key=lambda m: (float(m["src_exit_t"] or 0), m["src_camera"]))

    # yuksek/orta guveni ana dosyaya, dusuk guveni belirsiz dosyaya
    fields = ["match_id", "movement_label", "src_camera", "src_track", "src_class",
              "dst_camera", "dst_track", "dst_class", "src_exit_t", "dst_enter_t",
              "delta_t", "time_score", "score", "margin",
              "confidence", "n_candidates", "alternatives"]
    main_rows = [m for m in matches if m["confidence"] in ("high", "medium", "manual")]
    amb_rows = [m for m in matches if m["confidence"] == "low"]

    for name, rows in [("matches.csv", main_rows), ("matches_ambiguous.csv", amb_rows)]:
        p = MATCH_DIR / name
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    from collections import Counter
    by_conf = Counter(m["confidence"] for m in matches)
    by_conf_cls = Counter((m["confidence"], m["src_class"]) for m in matches)
    print(f"Toplam bire-bir eslesme: {len(matches)}")
    print(f"  guven dagilimi: {dict(by_conf)}")
    if manual:
        n_manual = sum(1 for m in matches if m["confidence"] == "manual")
        print(f"  elle onayli enjekte: {n_manual} satir ({len(manual)} arac)")
        if dropped:
            print(f"  cakisan otomatik eslesme dusuruldu: {len(dropped)} "
                  f"({', '.join(m['match_id'] for m in dropped[:5])})")
    print(f"  -> matches.csv (high+medium): {len(main_rows)}")
    print(f"  -> matches_ambiguous.csv (low): {len(amb_rows)}")
    print("  guven x sinif:")
    for (conf, cls), n in sorted(by_conf_cls.items()):
        print(f"     {conf:6} {cls:5}: {n}")


if __name__ == "__main__":
    main()

"""Faz 3 saat kaymasi tahmini: iki kamera arasindaki zaman farkini CSV'den olcer.

ANLAM:
    ortak_zaman = video_zamani + offset_seconds[camera_id]

Yani her kameranin kendi videosuna gore urettigi timestamp degerine, o kameranin
offset degeri EKLENEREK ortak zaman eksenine tasinir. Referans kameranin offset
degeri tanim geregi 0.0'dir.

Bu script videolari ACMAZ, kare okumaz. Tum olcumler
outputs/zones/zone_tracks_mapped_<cam>.csv uzerinden yapilir.

Uc yontem de calistirilir ve ucunun sonucu da raporlanir:
    A) olay orani capraz korelasyonu   (zayif kanit; tepe belirginligi olculur)
    B) handoff tutarlilik taramasi     (problemin kendisini olcer; secilen yontem)
    C) ortusme sinamasi                (|delta_t| < 1 s orani -> overlap mi handoff mu)

Olcum basarisiz olursa offset 0.0 yazilir ama method "assumed_zero",
confidence "low" olur ve rapora ENGEL olarak islenir. cameras.yaml
DEGISTIRILMEZ; senkron bilgisi ayri dosyada (configs/time_sync.yaml) yasar.

Ornek:
    python scripts/03_check_sync.py --config configs/cameras.yaml --zones-dir outputs/zones
    python scripts/03_check_sync.py --search-range 10.0 --bin-seconds 0.5
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")          # pencere acilmaz; dosyaya yazar
import matplotlib.pyplot as plt   # noqa: E402  (backend secildikten sonra import edilir)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
DEFAULT_ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
DEFAULT_OUT_CONFIG = PROJECT_ROOT / "configs" / "time_sync.yaml"
DEFAULT_OUT_REPORT = PROJECT_ROOT / "outputs" / "sync" / "sync_report.md"
DEFAULT_OUT_PLOT = PROJECT_ROOT / "outputs" / "sync" / "event_rate_xcorr.png"

# Aday havuzlari: (hareket etiketi, kaynak (kamera, bolge), hedef (kamera, bolge))
# Kaynak tarafta bolgeden CIKIS (t_exit), hedef tarafta bolgeye GIRIS (t_enter) kullanilir.
DIRECTION_PAIRS = (
    ("camA_to_camB", ("camA", "camA_exit"), ("camB", "camB_entry")),
    ("camB_to_camA", ("camB", "camB_exit"), ("camA", "camA_entry")),
)

MAX_TRANSIT_S = 30.0        # kor bolge gecisi icin makul kabul edilen ust sinir
NEAR_ZERO_S = 1.0           # ortusme sinamasi esigi: |delta_t| < 1 s
OVERLAP_RATIO_MIN = 0.30    # |delta_t| < 1 s orani bunun uzerindeyse "overlap"
HANDOFF_RATIO_MIN = 0.60    # makul pozitif gecis orani bunun uzerindeyse "handoff"
PEAK_RATIO_MIN = 1.3        # A yonteminde tepe / ikinci tepe orani bunun altindaysa zayif
PEAK_EXCLUSION_S = 2.0      # ikinci tepe aranirken ana tepenin cevresinde dislanan yaricap
PLATEAU_SPAN_MAX_S = 2.0    # B yonteminde esit skorlu offset araligi bundan genisse zayif
MIN_IQR_SAMPLES = 4         # IQR hesabi icin gereken en az makul gecis sayisi
MODE_LIFT_MIN = 1.3         # mod onerisinin sans seviyesinden ayrismasi icin gereken kat

# --- yon bazli korelogram (yontem D) ---
# Tum (kaynak, hedef) ciftlerinin delta_t histogrami, iki olay dizisi bagimsizmis
# gibi beklenen arka planla karsilastirilir. En yakin komsu eslestirmesinden farkli
# olarak hicbir cift kurgusu varsaymaz; bu yuzden gecis suresini olcebilir.
CORR_MIN_Z = 4.0            # tepe bu z degerinin altindaysa "belirlenemedi"
CORR_MIN_EXPECTED = 2.0     # beklenen cift sayisi bunun altindaki kovalar degerlendirilmez
CORR_OVERLAP_S = 1.0        # tepe konumu |lag| < bu deger ise ortusme belirtisi
ANCHOR_AUTO = "auto"

REQUIRED_COLUMNS = ("camera_id", "zone_id", "t_enter", "t_exit", "movement_label")


# --------------------------------------------------------------------------
# Okuma yardimcilari
# --------------------------------------------------------------------------

def rel(path: Path) -> str:
    """Rapor icin proje kokune gore yol metni."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def to_float(value):
    """Metni float'a cevirir; olmuyorsa None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_cameras(config_path: Path):
    """cameras.yaml icindeki kamera kimliklerini sirasiyla dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cameras = [str(entry["camera_id"]) for entry in (data.get("cameras") or [])
               if isinstance(entry, dict) and entry.get("camera_id")]
    if not cameras:
        raise SystemExit(f"HATA: cameras.yaml icinde kamera tanimi yok: {config_path}")
    return cameras


def load_visits(zones_dir: Path, camera_id: str):
    """zone_tracks_mapped_<cam>.csv satirlarini okur; (satirlar, hata) dondurur."""
    path = zones_dir / f"zone_tracks_mapped_{camera_id}.csv"
    if not path.is_file():
        return [], f"dosya yok: {rel(path)}"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        return [], f"okunamadi: {exc}"
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        return [], f"beklenen kolonlar eksik: {missing}"
    return rows, None


def zone_times(rows, zone_id: str, movement: str, field: str) -> np.ndarray:
    """Belirli bolge ve hareket etiketindeki ziyaretlerin zaman degerlerini dondurur."""
    values = []
    for row in rows:
        if row.get("zone_id") != zone_id or row.get("movement_label") != movement:
            continue
        value = to_float(row.get(field))
        if value is not None:
            values.append(value)
    return np.array(sorted(values), dtype=float)


def all_enter_times(rows) -> np.ndarray:
    """Kameradaki tum ziyaretlerin t_enter degerleri (olay orani serisi icin)."""
    values = [to_float(row.get("t_enter")) for row in rows]
    return np.array(sorted(value for value in values if value is not None), dtype=float)


# --------------------------------------------------------------------------
# Yontem A - olay orani capraz korelasyonu
# --------------------------------------------------------------------------

def normalized_series(times: np.ndarray, edges: np.ndarray):
    """Zaman damgalarindan normalize edilmis (ortalama 0, std 1) kova serisi uretir."""
    counts, _ = np.histogram(times, bins=edges)
    counts = counts.astype(float)
    std = counts.std()
    if std <= 0.0:
        return None, counts
    return (counts - counts.mean()) / std, counts


def method_a_xcorr(times_ref: np.ndarray, times_other: np.ndarray,
                   search_range: float, bin_seconds: float, step: float):
    """Capraz korelasyon egrisini ve tepe belirginligini olcer."""
    result = {"available": False, "reason": None, "lags": None, "corr": None,
              "best_offset": None, "best_corr": None, "second_corr": None,
              "second_offset": None, "peak_ratio": None, "n_bins": None,
              "n_bins_data": None}
    if times_ref.size == 0 or times_other.size == 0:
        result["reason"] = "en az bir kamerada t_enter degeri yok"
        return result

    low = min(times_ref.min(), times_other.min()) - search_range
    high = max(times_ref.max(), times_other.max()) + search_range
    edges = np.arange(np.floor(low), np.ceil(high) + bin_seconds, bin_seconds)
    if edges.size < 3:
        result["reason"] = "kova sayisi yetersiz"
        return result

    ref_series, _ = normalized_series(times_ref, edges)
    if ref_series is None:
        result["reason"] = "referans seride degisim yok (std = 0)"
        return result

    lags = np.round(np.arange(-search_range, search_range + step / 2.0, step), 6)
    corr = np.full(lags.size, np.nan, dtype=float)
    for index, lag in enumerate(lags):
        other_series, _ = normalized_series(times_other + lag, edges)
        if other_series is None:
            continue
        corr[index] = float(np.mean(ref_series * other_series))

    if np.all(np.isnan(corr)):
        result["reason"] = "hicbir kaymada korelasyon hesaplanamadi"
        return result

    best_index = int(np.nanargmax(corr))
    best_offset = float(lags[best_index])
    best_corr = float(corr[best_index])

    # Ikinci tepe: ana tepenin PEAK_EXCLUSION_S yaricapinin disindaki en yuksek yerel tepe.
    second_corr = None
    second_offset = None
    for index in range(1, lags.size - 1):
        value = corr[index]
        if np.isnan(value):
            continue
        if abs(lags[index] - best_offset) <= PEAK_EXCLUSION_S:
            continue
        left = corr[index - 1]
        right = corr[index + 1]
        if np.isnan(left) or np.isnan(right):
            continue
        if value >= left and value >= right:
            if second_corr is None or value > second_corr:
                second_corr = float(value)
                second_offset = float(lags[index])

    peak_ratio = None
    if second_corr is not None and second_corr > 0.0 and best_corr > 0.0:
        peak_ratio = best_corr / second_corr

    data_bins = int(np.ceil((max(times_ref.max(), times_other.max())
                             - min(times_ref.min(), times_other.min())) / bin_seconds))
    result.update({
        "available": True, "lags": lags, "corr": corr, "edges": edges,
        "best_offset": best_offset, "best_corr": best_corr,
        "second_corr": second_corr, "second_offset": second_offset,
        "peak_ratio": peak_ratio, "n_bins": int(edges.size - 1),
        "n_bins_data": data_bins,
    })
    return result


# --------------------------------------------------------------------------
# Yontem B - handoff tutarlilik taramasi
# --------------------------------------------------------------------------

def match_deltas(src_times: np.ndarray, dst_times: np.ndarray,
                 off_src: float, off_dst: float) -> np.ndarray:
    """Kaynak ve hedef ziyaretleri |delta_t| kucukten buyuge bire-bir eslestirir.

    Eslestirme acgozludur ve bir kaynak ziyaret yalnizca bir hedefe baglanir.
    Bu, ayni hedefin onlarca kaynaga sayilmasini engeller. Donen dizi ortak
    zamandaki delta_t degerleridir:
        delta_t = (t_hedef + off_hedef) - (t_kaynak + off_kaynak)
    """
    if src_times.size == 0 or dst_times.size == 0:
        return np.array([], dtype=float)
    matrix = (dst_times[None, :] + off_dst) - (src_times[:, None] + off_src)
    n_cols = matrix.shape[1]
    order = np.argsort(np.abs(matrix), axis=None)
    used_src, used_dst = set(), set()
    deltas = []
    for flat in order:
        row, col = divmod(int(flat), n_cols)
        if row in used_src or col in used_dst:
            continue
        used_src.add(row)
        used_dst.add(col)
        deltas.append(float(matrix[row, col]))
        if len(used_src) == matrix.shape[0] or len(used_dst) == n_cols:
            break
    return np.array(deltas, dtype=float)


def delta_stats(deltas: np.ndarray) -> dict:
    """delta_t dagiliminin ozet istatistikleri."""
    if deltas.size == 0:
        return {"n": 0}
    q25, q75 = np.percentile(deltas, [25, 75])
    return {
        "n": int(deltas.size),
        "min": float(deltas.min()),
        "p10": float(np.percentile(deltas, 10)),
        "median": float(np.median(deltas)),
        "p90": float(np.percentile(deltas, 90)),
        "max": float(deltas.max()),
        "iqr": float(q75 - q25),
    }


def evaluate_offset(pools, offset_other: float, reference: str, other: str) -> dict:
    """Tek bir offset adayini iki yonde de degerlendirir."""
    offsets = {reference: 0.0, other: offset_other}
    directions = {}
    plausible_total = 0
    matched_total = 0
    near_zero_total = 0
    iqr_values = []
    for movement, (src_cam, src_zone), (dst_cam, dst_zone) in DIRECTION_PAIRS:
        pool = pools[movement]
        deltas = match_deltas(pool["src"], pool["dst"], offsets[src_cam], offsets[dst_cam])
        plausible = deltas[(deltas > 0.0) & (deltas < MAX_TRANSIT_S)]
        near_zero = deltas[np.abs(deltas) < NEAR_ZERO_S]
        stats = delta_stats(plausible)
        directions[movement] = {
            "matched": int(deltas.size),
            "plausible": int(plausible.size),
            "near_zero": int(near_zero.size),
            "deltas": deltas,
            "plausible_deltas": plausible,
            "stats": stats,
            "all_stats": delta_stats(deltas),
        }
        matched_total += int(deltas.size)
        plausible_total += int(plausible.size)
        near_zero_total += int(near_zero.size)
        if plausible.size >= MIN_IQR_SAMPLES:
            iqr_values.append(stats["iqr"])

    iqr_score = float(np.mean(iqr_values)) if iqr_values else float("inf")
    return {
        "offset": offset_other,
        "directions": directions,
        "matched": matched_total,
        "plausible": plausible_total,
        "near_zero": near_zero_total,
        "plausible_ratio": plausible_total / matched_total if matched_total else 0.0,
        "near_zero_ratio": near_zero_total / matched_total if matched_total else 0.0,
        "iqr_score": iqr_score,
    }


def method_b_handoff(pools, reference: str, other: str,
                     search_range: float, step: float):
    """Offset adaylarini tarar; en cok makul gecis ve en dar IQR verenini secer."""
    result = {"available": False, "reason": None, "candidates": [], "best": None,
              "plateau_count": None, "plateau_span": None}
    if pools["camA_to_camB"]["src"].size == 0 and pools["camB_to_camA"]["src"].size == 0:
        result["reason"] = "iki yonde de kaynak ziyaret yok"
        return result
    if pools["camA_to_camB"]["dst"].size == 0 and pools["camB_to_camA"]["dst"].size == 0:
        result["reason"] = "iki yonde de hedef ziyaret yok"
        return result

    offsets = np.round(np.arange(-search_range, search_range + step / 2.0, step), 6)
    candidates = [evaluate_offset(pools, float(offset), reference, other) for offset in offsets]
    if not candidates:
        result["reason"] = "offset adayi uretilemedi"
        return result

    # Once en cok makul gecis, esitlikte en dar IQR, sonra sifira en yakin offset.
    ranked = sorted(candidates,
                    key=lambda item: (-item["plausible"], item["iqr_score"], abs(item["offset"])))
    best = ranked[0]

    max_plausible = best["plausible"]
    if max_plausible == 0:
        result["reason"] = "hicbir offset makul (0 < delta_t < 30 s) gecis uretmedi"
        result["candidates"] = ranked
        return result

    tied = [item["offset"] for item in candidates if item["plausible"] == max_plausible]
    result.update({
        "available": True, "candidates": ranked, "best": best,
        "plateau_count": len(tied),
        "plateau_span": float(max(tied) - min(tied)) if tied else 0.0,
        "offsets": offsets,
        "plausible_curve": np.array([item["plausible"] for item in candidates], dtype=float),
    })
    return result


# --------------------------------------------------------------------------
# Yontem C - ortusme sinamasi ve guven derecesi
# --------------------------------------------------------------------------

def method_c_overlap(evaluation) -> dict:
    """Secilen offset'te |delta_t| < 1 s oranindan mod onerisi uretir."""
    near_ratio = evaluation["near_zero_ratio"]
    plausible_ratio = evaluation["plausible_ratio"]
    if near_ratio > OVERLAP_RATIO_MIN:
        mode = "overlap"
        reason = (f"|delta_t| < {NEAR_ZERO_S:.1f} s orani %{100 * near_ratio:.1f} > "
                  f"%{100 * OVERLAP_RATIO_MIN:.0f} esigi")
    elif plausible_ratio >= HANDOFF_RATIO_MIN:
        mode = "handoff"
        reason = (f"makul pozitif gecis orani %{100 * plausible_ratio:.1f} >= "
                  f"%{100 * HANDOFF_RATIO_MIN:.0f} esigi ve |delta_t| < {NEAR_ZERO_S:.1f} s "
                  f"orani %{100 * near_ratio:.1f} esigin altinda")
    else:
        mode = "unknown"
        reason = (f"ne |delta_t| < {NEAR_ZERO_S:.1f} s orani (%{100 * near_ratio:.1f}) "
                  f"ne de makul gecis orani (%{100 * plausible_ratio:.1f}) esigi gecti")
    return {"mode": mode, "reason": reason,
            "near_ratio": near_ratio, "plausible_ratio": plausible_ratio}


def correlogram(src: np.ndarray, dst: np.ndarray, lag_low: float, lag_high: float,
                bin_seconds: float):
    """Yon bazli capraz korelogram: tum ciftlerin delta_t histogrami + beklenen arka plan.

    Beklenen arka plan, iki olay dizisi birbirinden BAGIMSIZ ve kendi zaman
    araliklarinda duzgun dagilmis kabul edilerek analitik hesaplanir:
        beklenen(lag) = rate_kaynak * rate_hedef * kova_genisligi * ortusen_sure(lag)
    Gozlenen sayinin beklenenden ne kadar saptigi z = (gozlenen - beklenen)/sqrt(beklenen)
    ile olculur. Buradaki tepe, cift kurgusu varsayilmadan olculmus bir yigilmadir.
    """
    result = {"available": False, "reason": None}
    if src.size < 2 or dst.size < 2:
        result["reason"] = f"yetersiz ziyaret (kaynak {src.size}, hedef {dst.size})"
        return result

    deltas = (dst[None, :] - src[:, None]).ravel()
    edges = np.arange(lag_low, lag_high + bin_seconds, bin_seconds)
    if edges.size < 3:
        result["reason"] = "kova sayisi yetersiz"
        return result
    counts, _ = np.histogram(deltas, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    src_lo, src_hi = float(src.min()), float(src.max())
    dst_lo, dst_hi = float(dst.min()), float(dst.max())
    span_src = max(src_hi - src_lo, 1e-9)
    span_dst = max(dst_hi - dst_lo, 1e-9)
    rate_src = src.size / span_src
    rate_dst = dst.size / span_dst
    # kaynak araligi lag kadar kaydirilinca hedef araligiyla ortusen sure
    overlap = np.maximum(0.0, np.minimum(dst_hi, src_hi + centers)
                         - np.maximum(dst_lo, src_lo + centers))
    expected = rate_src * rate_dst * bin_seconds * overlap

    valid = expected >= CORR_MIN_EXPECTED
    if not np.any(valid):
        result["reason"] = "beklenen cift sayisi her kovada esigin altinda"
        return result
    z_scores = np.full(centers.size, np.nan)
    z_scores[valid] = (counts[valid] - expected[valid]) / np.sqrt(expected[valid])

    peak_index = int(np.nanargmax(z_scores))
    peak_lag = float(centers[peak_index])
    peak_z = float(z_scores[peak_index])

    # ikinci tepe: ana tepeden en az 2 kova uzaktaki en yuksek z
    far = np.abs(centers - peak_lag) > max(2.0 * bin_seconds, 1.0)
    second_z = None
    second_lag = None
    if np.any(far & valid):
        masked = np.where(far & valid, z_scores, -np.inf)
        second_index = int(np.argmax(masked))
        if np.isfinite(masked[second_index]):
            second_z = float(z_scores[second_index])
            second_lag = float(centers[second_index])

    # tepe cevresindeki fazlalik: arka plandan artan cift sayisi
    excess = float(counts[peak_index] - expected[peak_index])

    # Kova genisliginden daha ince tahmin: tepe kovasindaki delta'larin medyani.
    # Kovada arka plan ciftleri de bulundugu icin medyan (ortalama degil) kullanilir.
    in_peak = deltas[(deltas >= peak_lag - bin_seconds / 2.0)
                     & (deltas < peak_lag + bin_seconds / 2.0)]
    peak_refined = float(np.median(in_peak)) if in_peak.size else peak_lag
    result.update({
        "available": True, "centers": centers, "counts": counts.astype(float),
        "expected": expected, "z": z_scores, "valid": valid,
        "peak_lag": peak_lag, "peak_z": peak_z, "peak_refined": peak_refined,
        "peak_bin_n": int(in_peak.size),
        "peak_count": float(counts[peak_index]), "peak_expected": float(expected[peak_index]),
        "peak_excess": excess,
        "second_z": second_z, "second_lag": second_lag,
        "n_pairs": int(deltas.size), "bin_seconds": bin_seconds,
        "significant": peak_z >= CORR_MIN_Z,
    })
    return result


def correlogram_verdict(corr, lag_shift: float = 0.0, is_anchor: bool = False) -> dict:
    """Korelogram tepesinden yon bazli mod ve gecis suresi yorumu uretir.

    Capa OLMAYAN yonlerde tepe konumu, capa yon icin yapilan gecis suresi
    kabuluyle birlikte kayar; bu yuzden o yonlerin modu kosulludur.
    """
    if not corr["available"]:
        return {"mode": "unknown", "text": f"olculemedi ({corr['reason']})", "lag": None,
                "conditional": False}
    lag = corr["peak_refined"] + lag_shift
    conditional = not is_anchor
    suffix = ("" if is_anchor else
              " — DIKKAT: bu konum capa yon icin yapilan gecis suresi kabulune baglidir; "
              "kabul degisirse bu tepe de birebir kayar, dolayisiyla bu yonun modu "
              "olculmus degil KOSULLUDUR")
    if not corr["significant"]:
        return {"mode": "unknown", "lag": lag, "conditional": conditional,
                "text": (f"tepe {lag:+.1f} s'de ama z = {corr['peak_z']:.1f} "
                         f"({CORR_MIN_Z:.0f} esiginin altinda); arka plandan ayrismiyor")}
    # Negatif gecis suresi = arac kaynaktan cikmadan hedefte gorunuyor = ortusme.
    # Bu yuzden esik mutlak degere degil, isaretli degere uygulanir.
    if lag < CORR_OVERLAP_S:
        mode = "overlap"
        if lag < -CORR_OVERLAP_S:
            detail = (f"arac kaynak bolgeden cikmadan {abs(lag):.2f} s once hedef bolgede "
                      "gorunuyor")
        else:
            detail = "ayni arac iki kamerada neredeyse ayni anda gorunuyor"
        text = (f"tepe {lag:+.2f} s'de (z = {corr['peak_z']:.1f}): {detail}, "
                f"yani bu yonde ortusme var{suffix}")
    else:
        mode = "handoff"
        text = (f"tepe {lag:+.2f} s'de (z = {corr['peak_z']:.1f}), sifirdan "
                f"{CORR_OVERLAP_S:.1f} s'den uzak: bu yonde araclar arada "
                f"yaklasik {lag:.1f} s gorunmuyor demektir{suffix}")
    return {"mode": mode, "lag": lag, "text": text, "conditional": conditional}


def identifiability(raw_corr, other: str):
    """Saat farkinin gercek gecis suresinden ayrilip ayrilamadigini olcer.

    Her yon icin ortak zamandaki gercek gecis suresi:
        tau_yon = peak_ham_yon + (off_hedef - off_kaynak)
    Iki yonun biri +off_other, digeri -off_other tasidigi icin TOPLAMLARINDA
    offset sadelesir:
        tau_ileri + tau_geri = peak_ham_ileri + peak_ham_geri
    Yani iki gecis suresinin TOPLAMI offset bilinmeden olculebilir; ikisinin
    birbirinden nasil ayrildigi ise olculemez. Bir yonun gecis suresi disaridan
    verilmedikce offset tek basina belirlenemez.
    """
    usable = {movement: item for movement, item in raw_corr.items()
              if item["corr"]["available"] and item["corr"]["significant"]}
    if len(usable) < 2:
        return None
    forward = None   # hedefi 'other' olan yon: tau = peak_ham + off_other
    peaks = {}
    for movement, item in usable.items():
        peaks[movement] = float(item["corr"]["peak_refined"])
        if item["dst_cam"] == other:
            forward = movement
    if forward is None:
        return None
    backward = next(movement for movement in usable if movement != forward)
    return {
        "forward": forward, "backward": backward,
        "peak_forward": peaks[forward], "peak_backward": peaks[backward],
        "tau_sum": peaks[forward] + peaks[backward],
    }


def chance_baseline(method_b, chosen_offset: float):
    """Secilen offset'in oranlarini uzak offset'lerdeki "sans seviyesi" ile karsilastirir.

    Yogun olay dizilerinde en yakin komsu eslestirmesi, iki kamera hic ilgisiz olsa
    bile kucuk |delta_t| uretir. Bu nedenle secilen offset'teki oranin, ondan uzak
    offset'lerde olculen tipik oranin kac kati oldugu da raporlanir.
    """
    candidates = method_b.get("candidates") or []
    far = [item for item in candidates
           if abs(item["offset"] - chosen_offset) > PEAK_EXCLUSION_S]
    if not far:
        return None
    near_ratios = np.array([item["near_zero_ratio"] for item in far], dtype=float)
    plausible_ratios = np.array([item["plausible_ratio"] for item in far], dtype=float)
    return {
        "n_offsets": len(far),
        "near_median": float(np.median(near_ratios)),
        "near_max": float(near_ratios.max()),
        "plausible_median": float(np.median(plausible_ratios)),
        "plausible_max": float(plausible_ratios.max()),
    }


def decide_confidence(method_a, method_b) -> tuple:
    """Iki yontemin uyumundan guven derecesi uretir; (derece, gerekce) dondurur."""
    if not method_b["available"]:
        return "low", "handoff taramasi olcum uretemedi"

    notes = []
    strong_peak = bool(method_a["available"] and method_a["peak_ratio"] is not None
                       and method_a["peak_ratio"] >= PEAK_RATIO_MIN)
    if method_a["available"]:
        if method_a["peak_ratio"] is None:
            notes.append("A: ikinci tepe bulunamadi, tepe belirginligi olculemedi")
        else:
            notes.append(f"A: tepe/ikinci tepe = {method_a['peak_ratio']:.2f} "
                         f"({'belirgin' if strong_peak else f'esik {PEAK_RATIO_MIN} altinda'})")
    else:
        notes.append(f"A: olcum yok ({method_a['reason']})")

    narrow = method_b["plateau_span"] <= PLATEAU_SPAN_MAX_S
    notes.append(f"B: en yuksek makul gecis sayisini veren offset araligi "
                 f"{method_b['plateau_span']:.1f} s genisliginde "
                 f"({'dar' if narrow else f'esik {PLATEAU_SPAN_MAX_S:.1f} s uzeri'})")

    agreement = None
    if method_a["available"]:
        agreement = abs(method_b["best"]["offset"] - method_a["best_offset"])
        notes.append(f"A ile B arasindaki fark {agreement:.2f} s")

    if strong_peak and narrow and agreement is not None and agreement <= 1.0:
        level = "high"
    elif (strong_peak or narrow) and agreement is not None and agreement <= 3.0:
        level = "medium"
    else:
        level = "low"
    return level, "; ".join(notes)


# --------------------------------------------------------------------------
# Grafik
# --------------------------------------------------------------------------

def draw_correlogram_plot(path: Path, direction_results, offset_other: float,
                          other: str) -> str:
    """Her yon icin gozlenen delta_t histogramini ve beklenen arka plani cizer."""
    usable = [item for item in direction_results if item["corr"]["available"]]
    if not usable:
        return "cizilmedi (hicbir yonde korelogram hesaplanamadi)"

    figure, axes = plt.subplots(len(usable), 1, figsize=(11, 4.0 * len(usable)))
    if len(usable) == 1:
        axes = [axes]
    for axis, item in zip(axes, usable):
        corr = item["corr"]
        centers = corr["centers"] + item["lag_shift"]
        axis.bar(centers, corr["counts"], width=corr["bin_seconds"] * 0.9,
                 color="#4C72B0", label="gözlenen çift sayısı")
        axis.plot(centers, corr["expected"], color="#C44E52", linewidth=2,
                  label="bağımsızlık varsayımı altında beklenen (şans seviyesi)")
        peak_lag = corr["peak_lag"] + item["lag_shift"]
        axis.axvline(peak_lag, color="green", linestyle="--",
                     label=f"tepe: {peak_lag:+.1f} s (z = {corr['peak_z']:.1f})")
        axis.axvline(0.0, color="gray", linewidth=1, alpha=0.6)
        axis.set_title(f"{item['movement']}  ({item['src_label']} → {item['dst_label']})  "
                       f"— {item['verdict']['mode']}")
        axis.set_xlabel(f"delta_t (saniye, ortak zaman — {other} offset "
                        f"{offset_other:+.2f} s uygulanmış)")
        axis.set_ylabel("çift sayısı")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return f"yazildi: {rel(path)}"


def draw_plot(path: Path, method_a, reference: str, other: str,
              times_ref: np.ndarray, times_other: np.ndarray,
              bin_seconds: float, chosen_offset: float) -> str:
    """Ust panel olay orani serileri, alt panel korelasyon egrisi. Durum metni dondurur."""
    if not method_a["available"]:
        return f"cizilmedi ({method_a['reason']})"

    edges = method_a["edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts_ref, _ = np.histogram(times_ref, bins=edges)
    counts_other, _ = np.histogram(times_other + chosen_offset, bins=edges)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5))

    axes[0].step(centers, counts_ref, where="mid", label=f"{reference} (referans)")
    axes[0].step(centers, counts_other, where="mid",
                 label=f"{other} (offset {chosen_offset:+.1f} s uygulanmis)")
    axes[0].set_title(f"Olay oranı — {bin_seconds:g} s'lik kovalarda bölgeye giriş sayısı")
    axes[0].set_xlabel("ortak zaman (saniye)")
    axes[0].set_ylabel(f"giriş sayısı / {bin_seconds:g} s")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(method_a["lags"], method_a["corr"], label="normalize çapraz korelasyon")
    axes[1].plot([method_a["best_offset"]], [method_a["best_corr"]], "o", color="red",
                 label=f"tepe: {method_a['best_offset']:+.1f} s (r = {method_a['best_corr']:.3f})")
    if method_a["second_offset"] is not None:
        axes[1].plot([method_a["second_offset"]], [method_a["second_corr"]], "s", color="orange",
                     label=f"ikinci tepe: {method_a['second_offset']:+.1f} s "
                           f"(r = {method_a['second_corr']:.3f})")
    axes[1].axvline(chosen_offset, color="green", linestyle="--",
                    label=f"seçilen offset: {chosen_offset:+.1f} s")
    axes[1].set_title("Kaymaya karşı korelasyon (yöntem A — zayıf kanıt)")
    axes[1].set_xlabel(f"{other} kaması / offset (saniye)")
    axes[1].set_ylabel("normalize korelasyon")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return f"yazildi: {rel(path)}"


# --------------------------------------------------------------------------
# Cikti dosyalari
# --------------------------------------------------------------------------

def format_stats_row(label: str, stats: dict) -> str:
    """delta_t ozet tablosu satiri."""
    if not stats or stats.get("n", 0) == 0:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    return (f"| {label} | {stats['n']} | {stats['min']:.2f} | {stats['p10']:.2f} | "
            f"{stats['median']:.2f} | {stats['p90']:.2f} | {stats['max']:.2f} | "
            f"{stats['iqr']:.2f} |")


def write_time_sync(path: Path, reference: str, other: str, offset: float,
                    method: str, confidence: str, search_range: float,
                    mode_suggestion: str, notes: str) -> None:
    """configs/time_sync.yaml dosyasini yazar (cameras.yaml'a dokunulmaz)."""
    # Cift tirnakli YAML skalerinde ters bolu kacis dizisi baslatir; Windows yollari
    # notes icine girdiginde dosya gecersiz olur. Once kacislar guvenli hale getirilir.
    safe_notes = notes.replace("\\", "\\\\").replace('"', "'")
    lines = [
        "# MCMOT_Pro — Faz 3 saat kaymasi (scripts/03_check_sync.py tarafindan uretildi)",
        "#",
        "# ANLAM:",
        "#     ortak_zaman = video_zamani + offset_seconds[camera_id]",
        "#",
        "# Referans kameranin offset degeri tanim geregi 0.0'dir. Degerler olcumdur,",
        "# kesin kimlik veya kesin senkron iddiasi degildir.",
        "",
        f'reference_camera: "{reference}"',
        "offset_seconds:",
        f"  {reference}: 0.0",
        f"  {other}: {offset:.2f}",
        f'method: "{method}"',
        f'confidence: "{confidence}"',
        f"search_range_s: [{-search_range:.1f}, {search_range:.1f}]",
        f'mode_suggestion: "{mode_suggestion}"',
        f'notes: "{safe_notes}"',
        "",
    ]
    text = "\n".join(lines)
    yaml.safe_load(text)   # yazmadan once gecerliligini dogrula
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_report(context) -> str:
    """sync_report.md metnini uretir."""
    method_a = context["method_a"]
    method_b = context["method_b"]
    method_c = context["method_c"]
    reference = context["reference"]
    other = context["other"]
    blockers = context["blockers"]

    lines = [
        "# Faz 3 - Saat Kaymasi Raporu",
        "",
        f"Bu rapor `scripts/03_check_sync.py` tarafindan {context['stamp']:%Y-%m-%d %H:%M:%S} "
        "tarihinde uretildi. Olcum yalnizca "
        "`zone_tracks_mapped_<cam>.csv` uzerinden yapilmistir; video acilmamistir.",
        "",
        f"- Referans kamera: `{reference}` (offset tanim geregi 0.0)",
        f"- Olculen kamera: `{other}`",
        f"- Arama araligi: {-context['search_range']:.1f} .. {+context['search_range']:.1f} s",
        f"- Kova genisligi (yontem A): {context['bin_seconds']:g} s, adim {context['xcorr_step']:g} s",
        f"- Tarama adimi (yontem B): {context['handoff_step']:g} s",
        f"- Anlam: `ortak_zaman = video_zamani + offset_seconds[camera_id]`",
        "",
    ]

    if blockers:
        lines += [f"## ENGEL ({len(blockers)})", ""]
        lines += [f"- {item}" for item in blockers]
        lines += ["", "Bu maddeler giderilmeden zaman tabanli eslestirme skorlamasi "
                      "guvenilir sayilmamalidir.", ""]

    # 1 - uc yontem yan yana
    lines += ["## 1. Uc yontemin sonucu", "",
              "| yontem | ne olcer | en iyi offset (s) | olcut | guven isareti |",
              "|---|---|---|---|---|"]
    if method_a["available"]:
        peak_text = (f"{method_a['peak_ratio']:.2f}x ikinci tepe"
                     if method_a["peak_ratio"] is not None else "ikinci tepe bulunamadi")
        lines.append(f"| A - olay orani capraz korelasyonu | iki kameranin olay yogunlugu "
                     f"benzerligi | {method_a['best_offset']:+.1f} | "
                     f"r = {method_a['best_corr']:.3f} ({method_a['n_bins']} kova, "
                     f"verinin kapsadigi ~{method_a['n_bins_data']} kova) | {peak_text} |")
    else:
        lines.append(f"| A - olay orani capraz korelasyonu | iki kameranin olay yogunlugu "
                     f"benzerligi | olculemedi | {method_a['reason']} | - |")

    if method_b["available"]:
        best = method_b["best"]
        lines.append(f"| B - handoff tutarliligi | gercek gecis delta_t'lerinin makullugu | "
                     f"{best['offset']:+.1f} | makul gecis {best['plausible']}/{best['matched']} "
                     f"(%{100 * best['plausible_ratio']:.1f}), ortalama IQR "
                     f"{best['iqr_score']:.2f} s | esit skorlu offset araligi "
                     f"{method_b['plateau_span']:.1f} s ({method_b['plateau_count']} aday) |")
    else:
        lines.append(f"| B - handoff tutarliligi | gercek gecis delta_t'lerinin makullugu | "
                     f"olculemedi | {method_b['reason']} | - |")

    if method_c:
        lines.append(f"| C - ortusme sinamasi | ayni anda gorunme belirtisi | "
                     f"(secilen offset'te) | \\|delta_t\\| < {NEAR_ZERO_S:.1f} s orani "
                     f"%{100 * method_c['near_ratio']:.1f} | mode_suggestion: "
                     f"`{method_c['mode']}` |")
    else:
        lines.append("| C - ortusme sinamasi | ayni anda gorunme belirtisi | olculemedi | "
                     "secilecek offset yok | mode_suggestion: `unknown` |")

    lines += ["",
              "Yontem A tek basina zayif kanittir: seri kisadir ve tepe komsu tepelerden "
              "yeterince ayrilmayabilir. Yontem B problemin kendisini (gercek gecis "
              "delta_t'lerini) olctugu icin secim onun sonucuna dayandirilmistir.",
              ""]

    if method_b["available"]:
        lines += ["### B yonteminin en iyi 5 adayi", "",
                  "| offset (s) | eslesen cift | makul gecis | oran | ortalama IQR (s) | "
                  f"\\|delta_t\\| < {NEAR_ZERO_S:.1f} s |",
                  "|---|---|---|---|---|---|"]
        for item in method_b["candidates"][:5]:
            iqr_text = "-" if item["iqr_score"] == float("inf") else f"{item['iqr_score']:.2f}"
            lines.append(f"| {item['offset']:+.1f} | {item['matched']} | {item['plausible']} | "
                         f"%{100 * item['plausible_ratio']:.1f} | {iqr_text} | "
                         f"{item['near_zero']} (%{100 * item['near_zero_ratio']:.1f}) |")
        lines.append("")

    # 2 - yon bazli korelogram
    lines += ["## 2. Yon bazli korelogram (yontem D — ana olcum)", "",
              "Bu olcum hicbir cift kurgusu varsaymaz: iki yonun TUM (kaynak, hedef) "
              "ciftlerinin `delta_t` degerleri histograma dokulur ve iki olay dizisi "
              "birbirinden bagimsizmis gibi beklenen sayiyla karsilastirilir. Arka planin "
              f"uzerine cikan tepe, o yonde gercek bir yigilma oldugunu gosterir "
              f"(z = (gozlenen - beklenen) / sqrt(beklenen), esik z >= {CORR_MIN_Z:.0f}).",
              "",
              f"- Kova genisligi: {context['corr_bin']:g} s, incelenen aralik "
              f"{-context['search_range']:.0f} .. {context['transit_max']:.0f} s",
              f"- Capa yon: `{context['anchor']}` — {context['anchor_note']}",
              f"- Capa yonde kabul edilen gercek gecis suresi: "
              f"{context['assumed_tau']:.1f} s (offset bu kabulun uzerine kurulur)",
              "",
              "| yon | cift | tepe (ortak zaman) | tepe kovasi: gozlenen / beklenen | z | "
              "ikinci tepe | yon bazli mod |",
              "|---|---|---|---|---|---|---|"]
    for item in context["direction_results"]:
        corr = item["corr"]
        label = f"{item['movement']}{' (capa)' if item['is_anchor'] else ''}"
        if not corr["available"]:
            lines.append(f"| {label} | - | olculemedi | - | - | - | `unknown` |")
            continue
        second = ("-" if corr["second_z"] is None
                  else f"{corr['second_lag'] + item['lag_shift']:+.1f} s "
                       f"(z = {corr['second_z']:.1f})")
        lines.append(f"| {label} | {corr['n_pairs']} | "
                     f"{corr['peak_refined'] + item['lag_shift']:+.2f} s "
                     f"(kova {corr['peak_lag'] + item['lag_shift']:+.2f}) | "
                     f"{corr['peak_count']:.0f} / {corr['peak_expected']:.1f} | "
                     f"{corr['peak_z']:.1f} | {second} | `{item['verdict']['mode']}`"
                     + ("" if item["is_anchor"] else " (kosullu)") + " |")
    lines.append("")
    for item in context["direction_results"]:
        lines.append(f"- **{item['movement']}** ({item['src_label']} -> {item['dst_label']}): "
                     f"{item['verdict']['text']}.")
    lines.append("")
    ident = context["identifiability"]
    if ident:
        lines += ["### 2b. Saat farki ile gecis suresi birbirinden ayrilabilir mi?", "",
                  "Her yon icin ortak zamandaki gercek gecis suresi `tau`, ham korelogram "
                  "tepesine offset EKLENEREK bulunur. Iki yon offset'i ters isaretle tasidigi "
                  "icin toplamlarinda offset sadelesir:",
                  "",
                  f"- `tau_{ident['forward']} = {ident['peak_forward']:+.3f} + offset[{other}]`",
                  f"- `tau_{ident['backward']} = {ident['peak_backward']:+.3f} - offset[{other}]`",
                  f"- **`tau_{ident['forward']} + tau_{ident['backward']} = "
                  f"{ident['tau_sum']:+.3f} s`** — bu toplam offset'ten BAGIMSIZ olarak "
                  "olculmustur.",
                  "",
                  "Sonuc: bu veriden iki gecis suresinin TOPLAMI olculebilir, ama saat "
                  "farkinin nerede bitip gecis suresinin nerede basladigi olculemez. "
                  "Bir yonun gecis suresi disaridan verilmedikce offset tek basina "
                  "belirlenemez. Asagidaki tablo, olasi kabuller icin sonucu gosterir "
                  f"(`--assumed-anchor-transit` ile capa yonun degeri verilir):",
                  "",
                  f"| {ident['forward']} gecis suresi (kabul) | offset[{other}] | "
                  f"{ident['backward']} gecis suresi (sonuc) | anlami |",
                  "|---|---|---|---|"]
        for tau_forward in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
            off = tau_forward - ident["peak_forward"]
            tau_backward = ident["peak_backward"] - off
            meaning = ("`" + ident["backward"] + "` yonunde ortusme (arac kaynaktan "
                       "cikmadan hedefte gorunuyor)") if tau_backward < 0 else \
                      ("`" + ident["backward"] + "` yonunde de bosluk var")
            lines.append(f"| {tau_forward:.2f} s | {off:+.2f} s | {tau_backward:+.2f} s | "
                         f"{meaning} |")
        lines += ["",
                  f"Su anda raporlanan offset ({context['offset']:+.2f} s), capa yon "
                  f"`{context['anchor']}` icin {context['assumed_tau']:.2f} s gecis suresi "
                  "KABUL EDILEREK uretilmistir; bu kabul olculmemistir.",
                  ""]

    if context["mixed_modes"]:
        lines += ["> UYARI: iki yon ayni davranmiyor. Bir yonde ortusme, diger yonde kor "
                  "bolge gecisi olcumu vardir. Faz 3 eslestirmesinde iki yon icin AYNI zaman "
                  "penceresi kullanilamaz; her yonun kendi delta_t beklentisi olmalidir.", ""]

    # 3 - secilen offset
    lines += ["## 3. Secilen offset ve gerekce", "",
              f"- `offset_seconds[{reference}]` = 0.0 (referans)",
              f"- `offset_seconds[{other}]` = **{context['offset']:.2f} s**",
              f"- `method` = `{context['method']}`",
              f"- `confidence` = `{context['confidence']}`",
              "",
              f"Gerekce: {context['selection_reason']}",
              "",
              f"Guven derecesi dayanagi: {context['confidence_reason']}.",
              ""]

    # 4 - delta_t dagilimi
    lines += [f"## 4. Secilen offset'te ({context['offset']:+.2f} s) delta_t dagilimi", "",
              "`delta_t = (hedef bolgeye giris + offset_hedef) - (kaynak bolgeden cikis + "
              "offset_kaynak)`. Ciftler `|delta_t|` kucukten buyuge acgozlu ve bire-bir "
              "eslestirilmistir; bu eslestirme bir kimlik iddiasi degil, dagilimi olcmek "
              "icin kurulmus gecici bir kabuldur.",
              ""]
    if context["evaluation"]:
        lines += ["Makul araliktaki (0 < delta_t < "
                  f"{MAX_TRANSIT_S:.0f} s) gecisler:", "",
                  "| yon | n | min | p10 | medyan | p90 | maks | IQR |",
                  "|---|---|---|---|---|---|---|---|"]
        for movement, (src_cam, src_zone), (dst_cam, dst_zone) in DIRECTION_PAIRS:
            data = context["evaluation"]["directions"][movement]
            lines.append(format_stats_row(f"{movement} ({src_cam}/{src_zone} -> {dst_cam}/{dst_zone})",
                                          data["stats"]))
        lines += ["",
                  "Filtresiz (tum eslesen ciftler, makul araligin disindakiler dahil):", "",
                  "| yon | n | min | p10 | medyan | p90 | maks | IQR |",
                  "|---|---|---|---|---|---|---|---|"]
        for movement, (src_cam, src_zone), (dst_cam, dst_zone) in DIRECTION_PAIRS:
            data = context["evaluation"]["directions"][movement]
            lines.append(format_stats_row(f"{movement} ({src_cam}/{src_zone} -> {dst_cam}/{dst_zone})",
                                          data["all_stats"]))
        lines += ["",
                  "| yon | eslesen cift | makul gecis | "
                  f"\\|delta_t\\| < {NEAR_ZERO_S:.1f} s | negatif delta_t |",
                  "|---|---|---|---|---|"]
        for movement, _src, _dst in DIRECTION_PAIRS:
            data = context["evaluation"]["directions"][movement]
            negative = int(np.sum(data["deltas"] < 0.0)) if data["deltas"].size else 0
            matched = data["matched"] or 1
            lines.append(f"| {movement} | {data['matched']} | {data['plausible']} "
                         f"(%{100 * data['plausible'] / matched:.1f}) | {data['near_zero']} "
                         f"(%{100 * data['near_zero'] / matched:.1f}) | {negative} "
                         f"(%{100 * negative / matched:.1f}) |")
        lines.append("")
    else:
        lines += ["Secilecek offset uretilemedigi icin dagilim olculememistir.", ""]

    # 5 - mod onerisi
    lines += ["## 5. mode_suggestion ve dayanagi", "",
              f"- **`mode_suggestion` = `{context['mode_suggestion']}`** — "
              f"{context['mode_source']}.",
              "",
              "Asagidaki karar kurali, en yakin komsu eslestirmesine dayanan C yontemine "
              "aittir ve iki yonu birlikte degerlendirir; yon bazli sonuc icin 2. bolume "
              "bakiniz.",
              "",
              "Karar kurali:",
              f"- `|delta_t| < {NEAR_ZERO_S:.1f} s` orani > %{100 * OVERLAP_RATIO_MIN:.0f} "
              "-> `overlap`",
              f"- makul pozitif gecis orani >= %{100 * HANDOFF_RATIO_MIN:.0f} -> `handoff`",
              "- hicbiri net degilse -> `unknown`",
              ""]
    if method_c:
        lines += [f"- Olculen `|delta_t| < {NEAR_ZERO_S:.1f} s` orani: "
                  f"%{100 * method_c['near_ratio']:.1f}",
                  f"- Olculen makul pozitif gecis orani: %{100 * method_c['plausible_ratio']:.1f}",
                  f"- C yonteminin karar kuralinin ciktisi: `{method_c['mode']}` — "
                  f"{method_c['reason']}.",
                  ""]
        baseline = context["baseline"]
        if baseline:
            lines += ["### Karar kuralinin sans seviyesinden ayrisma olcusu", "",
                      "Olay dizileri yogundur; en yakin komsu eslestirmesi iki kamera hic "
                      "ilgisiz olsa bile kucuk `|delta_t|` uretebilir. Asagidaki satirlar, "
                      f"secilen offset'ten {PEAK_EXCLUSION_S:.0f} s'den uzak "
                      f"{baseline['n_offsets']} offset adayinda olculen tipik oranlardir "
                      "(bu offset'lerde gercek bir eslesme beklenmez, yani sans seviyesidir):",
                      "",
                      "| olcut | secilen offset | uzak offset'lerin medyani | uzak offset'lerin "
                      "en yuksegi | ayrisma (kat) |",
                      "|---|---|---|---|---|"]
            near_lift = (method_c["near_ratio"] / baseline["near_median"]
                         if baseline["near_median"] > 0 else None)
            plaus_lift = (method_c["plausible_ratio"] / baseline["plausible_median"]
                          if baseline["plausible_median"] > 0 else None)
            lines.append(f"| `\\|delta_t\\| < {NEAR_ZERO_S:.1f} s` orani | "
                         f"%{100 * method_c['near_ratio']:.1f} | "
                         f"%{100 * baseline['near_median']:.1f} | "
                         f"%{100 * baseline['near_max']:.1f} | "
                         + (f"{near_lift:.2f}x |" if near_lift else "olculemedi |"))
            lines.append(f"| makul pozitif gecis orani | "
                         f"%{100 * method_c['plausible_ratio']:.1f} | "
                         f"%{100 * baseline['plausible_median']:.1f} | "
                         f"%{100 * baseline['plausible_max']:.1f} | "
                         + (f"{plaus_lift:.2f}x |" if plaus_lift else "olculemedi |"))
            lines.append("")
            if near_lift is not None and near_lift < MODE_LIFT_MIN:
                lines += [f"> UYARI: secilen offset'teki oran, sans seviyesinin yalnizca "
                          f"{near_lift:.2f} katidir ({MODE_LIFT_MIN} esiginin altinda). "
                          f"`mode_suggestion = {method_c['mode']}` karar kuralinin ciktisidir, "
                          "ancak bu veri ile ortusme ile rastlanti birbirinden "
                          "AYRISTIRILAMAMAKTADIR. Kameralarin ortusup ortusmedigi bu olcume "
                          "dayanarak kararlastirilmamalidir.", ""]
            else:
                lines += [f"Secilen offset'teki oran sans seviyesinin {near_lift:.2f} katidir "
                          f"({MODE_LIFT_MIN} esigi uzerinde); mod onerisi bu veride sans "
                          "seviyesinden ayrisiyor.", ""]
    else:
        lines += ["- **mode_suggestion = `unknown`** — olcum uretilemedi.", ""]

    # 6 - kalan belirsizlik
    lines += ["## 6. Kalan belirsizlik", "", context["uncertainty"], ""]

    lines += ["## Uretilen dosyalar", "",
              f"- `{rel(context['out_config'])}`",
              f"- `{rel(context['out_report'])}`",
              f"- Olay orani / korelasyon grafigi: {context['plot_status']}",
              f"- Yon bazli korelogram grafigi: {context['corr_plot_status']}",
              ""]
    return "\n".join(lines)


def build_uncertainty(context) -> str:
    """Kalan belirsizlik paragrafini olculen sayilarla kurar."""
    method_a = context["method_a"]
    method_b = context["method_b"]
    parts = []
    if context["method"] == "assumed_zero":
        parts.append("Bu tahmin bir olcume dayanmiyor: offset degeri 0.0 olarak varsayilmistir "
                     f"({context['selection_reason']}).")
    else:
        parts.append(f"Bu offset degeri ({context['offset']:+.2f} s) bir tahmindir, olculmus "
                     "bir saat farki degildir; iki kameranin gercek zaman damgalarini "
                     "karsilastiran bagimsiz bir kaynak (ortak saat, ortak olay, senkron "
                     "cekim kaydi) yoktur.")
    if method_b["available"]:
        best = method_b["best"]
        parts.append(f"Secim, {best['matched']} eslesen ciftin {best['plausible']} tanesinin "
                     f"0 ile {MAX_TRANSIT_S:.0f} s arasinda kalmasina dayanir; ancak bu pencere "
                     f"genistir ve {method_b['plateau_count']} farkli offset adayi "
                     f"({method_b['plateau_span']:.1f} s genisliginde bir aralik) ayni sayida "
                     "makul gecis uretmektedir, yani veri tek bir offset'i tek basina secmeye "
                     "yetmemektedir.")
    if method_a["available"]:
        if method_a["peak_ratio"] is None:
            parts.append(f"Capraz korelasyon tepesi {method_a['best_offset']:+.1f} s'de "
                         f"(r = {method_a['best_corr']:.3f}) olusmustur; karsilastirilacak "
                         "ikinci tepe bulunamadigi icin tepenin belirginligi olculememistir.")
        else:
            parts.append(f"Capraz korelasyon tepesi {method_a['best_offset']:+.1f} s'de "
                         f"(r = {method_a['best_corr']:.3f}) olup ikinci tepenin yalnizca "
                         f"{method_a['peak_ratio']:.2f} kati kadardir; "
                         f"{method_a['n_bins_data']} kovalik bir seride bu fark rastlantiyla "
                         "da olusabilir.")
    ident = context.get("identifiability")
    if ident:
        parts.append(f"En temel sinir sudur: bu veriden iki yonun gecis suresi TOPLAMI "
                     f"({ident['tau_sum']:+.3f} s) olculebilir, ama saat farkinin nerede "
                     "bitip gecis suresinin nerede basladigi olculemez; ikisi delta_t "
                     "icinde toplanmis halde gorunur. Bu nedenle raporlanan offset, bir "
                     "yonun gecis suresi hakkindaki kabulun dogrudan tasiyicisidir.")
    anchor = context.get("anchor")
    if anchor:
        parts.append(f"Offset, '{anchor}' yonundeki korelogram tepesinin ortak zamanda "
                     f"{context['assumed_tau']:.1f} s'ye oturtulmasiyla bulunmustur; bu kabul "
                     "yanlissa offset de ayni miktarda kayar, cunku saat farki ile o yondeki "
                     "gercek gecis suresi bu veriyle birbirinden ayrilamaz (ikisi de delta_t "
                     "icinde toplanmis halde gorulur).")
    for item in context.get("direction_results", []):
        if item["is_anchor"] or not item["corr"]["available"]:
            continue
        verdict = item["verdict"]
        if verdict["mode"] == "unknown":
            parts.append(f"Diger yonde ({item['movement']}) gecis suresi olculememistir: "
                         f"{verdict['text']}; bu yon icin zaman penceresi veriden degil "
                         "disaridan verilmelidir.")
        else:
            parts.append(f"Diger yonde ({item['movement']}) olculen "
                         f"{verdict['lag']:+.1f} s'lik tepe, capa yondeki kabule baglidir; "
                         "capa kabulu degisirse bu deger de birebir kayar.")
    baseline = context.get("baseline")
    method_c = context.get("method_c")
    if baseline and method_c and baseline["near_median"] > 0:
        lift = method_c["near_ratio"] / baseline["near_median"]
        parts.append(f"Mod onerisi de zayif temellidir: secilen offset'teki "
                     f"|delta_t| < {NEAR_ZERO_S:.1f} s orani "
                     f"(%{100 * method_c['near_ratio']:.1f}), gercek eslesme beklenmeyen uzak "
                     f"offset'lerdeki medyan oranin (%{100 * baseline['near_median']:.1f}) "
                     f"{lift:.2f} katidir; olay dizileri yogun oldugu icin en yakin komsu "
                     "eslestirmesi ilgisiz iki kamerada bile kucuk delta_t uretir.")
    parts.append("Eslestirmede kullanilan cift kurgusu da bir kabuldur: her kaynak ziyaret "
                 "en yakin hedefe baglanmistir, oysa kor bolgede baska yola sapan, duran veya "
                 "takibi kopan araclar bu ciftleri yanlis kurabilir; bu nedenle dagilim "
                 "istatistikleri gercek gecis suresinin degil, kabul edilen eslestirmenin "
                 "istatistikleridir.")
    parts.append("Sonuc olasiliksaldir ve Faz 3 skorlamasinda tek basina kesin kabul "
                 "edilmemelidir; offset degeri degistirilirse eslestirme sonuclari da degisir.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Ana akis
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 3: iki kamera arasindaki saat kaymasini CSV'den tahmin eder (video acmaz)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMERAS_CONFIG,
                        help="kamera config yolu (yalnizca okunur)")
    parser.add_argument("--zones-dir", type=Path, default=DEFAULT_ZONES_DIR,
                        help="zone_tracks_mapped_<cam>.csv dizini")
    parser.add_argument("--out-config", type=Path, default=DEFAULT_OUT_CONFIG,
                        help="uretilecek senkron config yolu")
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT_REPORT,
                        help="uretilecek rapor yolu")
    parser.add_argument("--out-plot", type=Path, default=DEFAULT_OUT_PLOT,
                        help="uretilecek grafik yolu")
    parser.add_argument("--search-range", type=float, default=20.0,
                        help="offset arama yaricapi (saniye)")
    parser.add_argument("--bin-seconds", type=float, default=1.0,
                        help="olay orani kova genisligi (saniye)")
    parser.add_argument("--xcorr-step", type=float, default=0.1,
                        help="yontem A tarama adimi (saniye)")
    parser.add_argument("--handoff-step", type=float, default=0.5,
                        help="yontem B tarama adimi (saniye)")
    parser.add_argument("--reference", default=None,
                        help="referans kamera (varsayilan: cameras.yaml'daki ilk kamera)")
    parser.add_argument("--corr-bin", type=float, default=0.5,
                        help="yon bazli korelogram kova genisligi (saniye)")
    parser.add_argument("--transit-max", type=float, default=30.0,
                        help="korelogramda incelenen en buyuk pozitif gecis suresi (saniye)")
    parser.add_argument("--anchor-direction", default=ANCHOR_AUTO,
                        help="offset'in turetilecegi capa yon: auto, camA_to_camB veya "
                             "camB_to_camA. Ortustugu BILINEN yon verilirse offset o yonden "
                             "turetilir")
    parser.add_argument("--assumed-anchor-transit", type=float, default=0.0,
                        help="capa yonde kabul edilen gercek gecis suresi (saniye); ortusen "
                             "bir yonde 0.0 kabulu offset'e bu kadar yanlilik tasir")
    parser.add_argument("--out-corr-plot", type=Path,
                        default=PROJECT_ROOT / "outputs" / "sync" / "direction_correlogram.png",
                        help="yon bazli korelogram grafigi yolu")
    args = parser.parse_args()

    for attribute in ("config", "zones_dir", "out_config", "out_report", "out_plot",
                      "out_corr_plot"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, PROJECT_ROOT / path)

    cameras = load_cameras(args.config)
    reference = args.reference or cameras[0]
    if reference not in cameras:
        raise SystemExit(f"HATA: referans kamera '{reference}' cameras.yaml icinde yok.")
    others = [camera_id for camera_id in cameras if camera_id != reference]
    blockers = []
    if len(others) != 1:
        raise SystemExit("HATA: bu script tam olarak iki kamera icin yazilmistir; "
                         f"cameras.yaml icinde {len(cameras)} kamera var.")
    other = others[0]

    visits = {}
    for camera_id in cameras:
        rows, error = load_visits(args.zones_dir, camera_id)
        visits[camera_id] = rows
        if error:
            blockers.append(f"{camera_id}: ziyaret ozeti okunamadi ({error}).")

    # Aday havuzlari: kaynak tarafta cikis zamani, hedef tarafta giris zamani.
    pools = {}
    for movement, (src_cam, src_zone), (dst_cam, dst_zone) in DIRECTION_PAIRS:
        src = zone_times(visits.get(src_cam, []), src_zone, movement, "t_exit")
        dst = zone_times(visits.get(dst_cam, []), dst_zone, movement, "t_enter")
        pools[movement] = {"src": src, "dst": dst,
                           "src_label": f"{src_cam}/{src_zone}", "dst_label": f"{dst_cam}/{dst_zone}"}
        if src.size == 0 or dst.size == 0:
            blockers.append(f"{movement}: aday havuzu bos (kaynak {src.size}, hedef {dst.size}).")

    times_ref = all_enter_times(visits.get(reference, []))
    times_other = all_enter_times(visits.get(other, []))

    method_a = method_a_xcorr(times_ref, times_other, args.search_range,
                              args.bin_seconds, args.xcorr_step)
    method_b = method_b_handoff(pools, reference, other, args.search_range, args.handoff_step)

    # --- Yontem D: yon bazli korelogram (video zamaninda, offset uygulanmadan) ---
    raw_corr = {}
    for movement, (src_cam, _src_zone), (dst_cam, _dst_zone) in DIRECTION_PAIRS:
        pool = pools[movement]
        raw_corr[movement] = {
            "movement": movement, "src_cam": src_cam, "dst_cam": dst_cam,
            "src_label": pool["src_label"], "dst_label": pool["dst_label"],
            "corr": correlogram(pool["src"], pool["dst"],
                                -args.search_range, args.transit_max, args.corr_bin),
        }

    # Capa yon: offset'i hangi yonun tepesinden turetecegiz?
    anchor = args.anchor_direction
    anchor_auto_note = ""
    if anchor == ANCHOR_AUTO:
        scored = [(item["corr"]["peak_z"], movement) for movement, item in raw_corr.items()
                  if item["corr"]["available"]]
        if scored:
            anchor = max(scored)[1]
            anchor_auto_note = (f"capa yon otomatik secildi: korelogram tepesi en belirgin olan "
                                f"yon '{anchor}' (z = {max(scored)[0]:.1f})")
        else:
            anchor = None
            anchor_auto_note = "capa yon secilemedi: hicbir yonde korelogram hesaplanamadi"
    elif anchor not in raw_corr:
        raise SystemExit(f"HATA: --anchor-direction '{anchor}' tanimli degil. "
                         f"Secenekler: {ANCHOR_AUTO}, " + ", ".join(raw_corr))
    else:
        anchor_auto_note = (f"capa yon kullanici tarafindan verildi: '{anchor}' "
                            "(bu bir gozlem bildirimidir, bu script tarafindan olculmemistir)")

    anchor_corr = raw_corr[anchor]["corr"] if anchor else {"available": False}
    if anchor and anchor_corr["available"] and anchor_corr["significant"]:
        item = raw_corr[anchor]
        peak_raw = float(anchor_corr["peak_refined"])
        tau = args.assumed_anchor_transit
        # ortak zamanda delta_t'nin tau olmasi istenir: off_hedef - off_kaynak = tau - peak_raw
        if item["dst_cam"] == other:
            offset = tau - peak_raw
        else:
            offset = peak_raw - tau
        method = "handoff_consistency"
        evaluation = evaluate_offset(pools, offset, reference, other)
        selection_reason = (
            f"D yontemi (yon bazli korelogram) secildi. Capa yon '{anchor}' "
            f"({item['src_label']} -> {item['dst_label']}): {anchor_corr['n_pairs']} ciftin "
            f"delta_t histograminda tepe {peak_raw:+.2f} s'de olusmus, bu kovada beklenen "
            f"{anchor_corr['peak_expected']:.1f} cifte karsilik {anchor_corr['peak_count']:.0f} "
            f"cift gorulmustur (z = {anchor_corr['peak_z']:.1f}). Bu yonde gercek gecis "
            f"suresi {tau:.1f} s kabul edildigi icin offset = {offset:+.2f} s bulunmustur. "
            f"{anchor_auto_note}")
    elif method_b["available"]:
        offset = float(method_b["best"]["offset"])
        method = "handoff_consistency"
        evaluation = method_b["best"]
        if anchor:
            blockers.append(
                f"Capa yon '{anchor}' korelogramindan offset turetilemedi "
                f"({anchor_corr.get('reason') or 'tepe z esigin altinda'}); offset daha zayif "
                "olan en yakin komsu taramasina dayaniyor.")
        selection_reason = (
            f"B yontemi (handoff tutarliligi) secildi: {evaluation['matched']} eslesen ciftin "
            f"{evaluation['plausible']} tanesi (%{100 * evaluation['plausible_ratio']:.1f}) "
            f"0 ile {MAX_TRANSIT_S:.0f} s arasinda kaldi ve esit skorlu adaylar icinde "
            f"ortalama IQR'si en dar olan ({evaluation['iqr_score']:.2f} s) offset budur. "
            "Bu yontem dogrudan kameralar arasi gecis suresini olctugu icin A yonteminin "
            "korelasyon tepesine tercih edilmistir")
    elif method_a["available"]:
        offset = float(method_a["best_offset"])
        method = "event_rate_xcorr"
        evaluation = evaluate_offset(pools, offset, reference, other)
        selection_reason = (
            f"B yontemi olcum uretemedi ({method_b['reason']}); bu nedenle A yonteminin "
            f"korelasyon tepesi ({offset:+.1f} s, r = {method_a['best_corr']:.3f}) kullanildi")
        blockers.append("Handoff taramasi olcum uretemedi; offset zayif kanit olan "
                        "capraz korelasyona dayaniyor.")
    else:
        offset = 0.0
        method = "assumed_zero"
        evaluation = None
        selection_reason = (f"hicbir yontem olcum uretemedi (A: {method_a['reason']}; "
                            f"B: {method_b['reason']}); offset olculmeden 0.0 varsayildi")
        blockers.append("Saat kaymasi OLCULEMEDI; offset 0.0 varsayimdir, senkron oldugu "
                        "anlamina gelmez.")

    # Yon bazli sonuclari ortak zamana tasi: her yon icin ayri mod ve gecis suresi.
    offsets_map = {reference: 0.0, other: offset}
    direction_results = []
    for movement, (src_cam, _src_zone), (dst_cam, _dst_zone) in DIRECTION_PAIRS:
        item = dict(raw_corr[movement])
        item["lag_shift"] = offsets_map[dst_cam] - offsets_map[src_cam]
        item["is_anchor"] = (movement == anchor)
        item["verdict"] = correlogram_verdict(item["corr"], item["lag_shift"],
                                              is_anchor=(movement == anchor))
        direction_results.append(item)

    method_c = method_c_overlap(evaluation) if evaluation else None
    mode_suggestion = method_c["mode"] if method_c else "unknown"
    # Yonler farkli davraniyorsa tek bir mod etiketi yetersiz kalir; bunu isaretle.
    direction_modes = {item["movement"]: item["verdict"]["mode"] for item in direction_results}
    distinct_modes = {mode for mode in direction_modes.values() if mode != "unknown"}
    mixed_modes = len(distinct_modes) > 1
    if mixed_modes:
        blockers.append(
            "Iki yon ayni davranmiyor: "
            + "; ".join(f"{movement} -> {mode}" for movement, mode in direction_modes.items())
            + ". time_sync.yaml semasindaki tek `mode_suggestion` alani bu ayrimi "
              "tasiyamaz; yon bazli sonuc yalnizca raporda ve notes alanindadir.")

    if mixed_modes:
        # Tek etiket iki yonu birden yanlis temsil eder; tuketiciyi rapora yonlendir.
        mode_suggestion = "unknown"
        mode_source = ("yonler farkli davrandigi icin tek etiket verilmedi (olcum "
                       "basarisizligi degil): "
                       + "; ".join(f"{movement} -> {mode}"
                                   for movement, mode in direction_modes.items()))
    elif distinct_modes:
        mode_suggestion = distinct_modes.pop()
        mode_source = "iki yonun korelogrami da ayni modu verdi"
    elif method_c:
        mode_source = ("korelogram sonuc vermedi; deger C yonteminin karar kuralindan "
                       f"alindi ({method_c['reason']})")
    else:
        mode_source = "hicbir yontem mod olcumu uretemedi"
    baseline = chance_baseline(method_b, offset) if method_c else None
    mode_lift = None
    if baseline and method_c and baseline["near_median"] > 0:
        mode_lift = method_c["near_ratio"] / baseline["near_median"]
        if mode_lift < MODE_LIFT_MIN:
            blockers.append(
                f"mode_suggestion ('{mode_suggestion}') sans seviyesinden ayrismiyor: "
                f"secilen offset'teki |delta_t| < {NEAR_ZERO_S:.1f} s orani, uzak "
                f"offset'lerdeki medyan oranin yalnizca {mode_lift:.2f} katidir "
                f"({MODE_LIFT_MIN} esiginin altinda). Ortusme/handoff karari bu olcume "
                "dayandirilamaz.")

    ident = identifiability(raw_corr, other)
    if method == "assumed_zero":
        confidence, confidence_reason = "low", "olcum uretilemedi"
    else:
        confidence, confidence_reason = decide_confidence(method_a, method_b)
    # Iki yonun tepesi de sifira yakinsa saat farki ile gecis suresi ayrilamaz:
    # offset, capa yon icin yapilan kabulun uzerine kurulmus olur.
    if ident and args.assumed_anchor_transit == 0.0:
        confidence = "low"
        confidence_reason = (
            f"{confidence_reason}; ANCAK offset, capa yon icin varsayilan 0.0 s gecis "
            f"suresi kabulune dayanmaktadir. Veriden yalnizca iki yonun gecis suresi "
            f"TOPLAMI ({ident['tau_sum']:+.3f} s) olculebilmistir")
        blockers.append(
            f"Saat farki ile gecis suresi bu veriden ayrilamiyor: olculebilen tek buyukluk "
            f"iki yonun gecis suresi toplami ({ident['tau_sum']:+.3f} s). Raporlanan offset "
            f"({offset:+.2f} s), '{anchor}' yonunde {args.assumed_anchor_transit:.2f} s gecis "
            "suresi KABUL EDILEREK uretilmistir. Bir yonun gecis suresi disaridan "
            "verilirse (--assumed-anchor-transit) offset dogrudan belirlenir.")

    plot_status = draw_plot(args.out_plot, method_a, reference, other,
                            times_ref, times_other, args.bin_seconds, offset)
    corr_plot_status = draw_correlogram_plot(args.out_corr_plot, direction_results,
                                             offset, other)

    context = {
        "stamp": datetime.now(), "reference": reference, "other": other,
        "search_range": args.search_range, "bin_seconds": args.bin_seconds,
        "xcorr_step": args.xcorr_step, "handoff_step": args.handoff_step,
        "method_a": method_a, "method_b": method_b, "method_c": method_c,
        "offset": offset, "method": method, "confidence": confidence,
        "confidence_reason": confidence_reason, "selection_reason": selection_reason,
        "evaluation": evaluation, "blockers": blockers,
        "baseline": baseline, "mode_lift": mode_lift,
        "direction_results": direction_results, "anchor": anchor,
        "anchor_note": anchor_auto_note, "mode_source": mode_source,
        "mode_suggestion": mode_suggestion, "mixed_modes": mixed_modes,
        "assumed_tau": args.assumed_anchor_transit,
        "corr_bin": args.corr_bin, "transit_max": args.transit_max,
        "corr_plot_status": corr_plot_status, "identifiability": ident,
        "out_config": args.out_config, "out_report": args.out_report,
        "plot_status": plot_status,
    }
    context["uncertainty"] = build_uncertainty(context)

    lift_note = ""
    if mode_lift is not None and mode_lift < MODE_LIFT_MIN:
        lift_note = (f" DIKKAT: mode_suggestion sans seviyesinin yalnizca {mode_lift:.2f} "
                     f"katidir ({MODE_LIFT_MIN} esigi altinda), yani ortusme ile rastlanti "
                     "bu veriyle ayristirilamamistir.")
    direction_note = " ".join(
        f"{item['movement']}: {item['verdict']['mode']}"
        + (f" (tepe {item['verdict']['lag']:+.1f} s)" if item["verdict"]["lag"] is not None
           else "") + "."
        for item in direction_results)
    notes = (f"{selection_reason}. Yon bazli sonuc: {direction_note} "
             f"mode_suggestion '{mode_suggestion}': {mode_source}.{lift_note} "
             f"Ayrinti icin {rel(args.out_report)} dosyasina bakiniz. "
             "Deger olcumdur, kesin senkron iddiasi degildir.").replace('"', "'")
    write_time_sync(args.out_config, reference, other, offset, method, confidence,
                    args.search_range, mode_suggestion, notes)

    report_text = build_report(context)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_text, encoding="utf-8")

    print(f"Referans      : {reference} (offset 0.0)")
    print(f"Olculen       : {other} offset = {offset:+.2f} s "
          f"[method={method}, confidence={confidence}, mode={mode_suggestion}]")
    print(f"Config        : {args.out_config}")
    print(f"Rapor         : {args.out_report}")
    print(f"Grafik        : {plot_status}")
    print(f"Korelogram    : {corr_plot_status}")
    for item in direction_results:
        lag = item["verdict"]["lag"]
        lag_text = f"{lag:+.2f} s" if lag is not None else "olculemedi"
        print(f"  {item['movement']:<14} tepe {lag_text:<12} -> {item['verdict']['mode']}"
              + ("  (capa)" if item["is_anchor"] else ""))
    if blockers:
        print(f"ENGEL         : {len(blockers)} madde (raporun basinda listelendi)")


if __name__ == "__main__":
    main()

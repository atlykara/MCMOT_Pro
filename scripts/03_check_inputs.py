"""Faz 3 girdi envanteri: eslestirme oncesi girdi dosyalarinin gercek durumu.

Script SALT OKUR. Hicbir girdi dosyasini degistirmez, silmez veya yeniden
yazmaz; tek yazdigi dosya --out ile verilen markdown raporudur.

Rapor yalnizca diskte olculen degerleri icerir. Okunamayan veya hesaplanamayan
bir buyukluk tahmin edilmez; "olculemedi" olarak yazilir.

Ornek:
    python scripts/03_check_inputs.py --config configs/cameras.yaml --out docs/faz3_girdi_envanteri.md
    python scripts/03_check_inputs.py --max-records 5000
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "faz3_girdi_envanteri.md"
CONFIG_DIR = PROJECT_ROOT / "configs"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"

MISSING = "EKSIK"
NOT_MEASURED = "olculemedi"

# zone_tracks_mapped_<cam>.csv icin beklenen kolon sirasi (Faz 2 ciktisi).
EXPECTED_MAPPED_COLUMNS = (
    "camera_id", "zone_id", "track_id", "visit_index", "t_enter", "t_exit",
    "frame_enter", "frame_exit", "dwell_s", "n_frames_inside",
    "start_foot_x", "start_foot_y", "end_foot_x", "end_foot_y",
    "dx", "dy", "distance_px", "speed_px_s", "direction_label", "lane",
    "class_mode", "conf_mean", "exit_reason", "movement_label", "mapping_rule",
)

# tracks_<cam>.jsonl icinde brief'in Faz 3 gecis sartinin karsiligi olan alanlar.
BRIEF_TRACK_FIELDS = ("timestamp", "camera_id", "track_id", "foot_point")

# direction_mapping.yaml kural anahtari (02_apply_direction_mapping.py ile ayni).
KEY_FIELDS = ("camera_id", "zone_id", "lane", "direction_label")
RULE_DEFAULT = "default"

REAL_MOVEMENTS = ("camA_to_camB", "camB_to_camA")
AMBIGUOUS_DIRECTIONS = ("stationary", "up", "down")

# Veri kalitesi bayrak esikleri. Bu script salt okur; yeni config uretmez.
DWELL_LONG_S = 5.0            # bu suredan uzun ziyaret "durmus arac" adayi
WEAK_DISTANCE_PX = 40.0       # bu yer degistirmenin altinda yon kaniti zayif
TRACK_END_REASON = "track_end"

# Eslestirilecek aday havuzlari: (yon, kaynak (kamera, bolge), hedef (kamera, bolge))
DIRECTION_PAIRS = (
    ("camA_to_camB", ("camA", "camA_exit"), ("camB", "camB_entry")),
    ("camB_to_camA", ("camB", "camB_exit"), ("camA", "camA_entry")),
)

DEFAULT_HINTS_SAMPLE = 200
EXPECTED_HINTS = {
    "dominant_color": "3 elemanli liste",
    "size_class": "str",
    "aspect_ratio": "float",
}


# --------------------------------------------------------------------------
# Genel yardimcilar
# --------------------------------------------------------------------------

def human_size(num_bytes: int) -> str:
    """Bayt degerini okunabilir birime cevirir."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def count_lines(path: Path):
    """Dosyadaki satir sayisini dondurur; okunamazsa None."""
    try:
        total = 0
        last_char = b"\n"
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += chunk.count(b"\n")
                last_char = chunk[-1:]
        if last_char not in (b"\n", b""):
            total += 1   # son satir yeni satirla bitmiyorsa o da sayilir
        return total
    except OSError:
        return None


def file_info(path: Path) -> dict:
    """Tek dosya icin varlik/boyut/satir/tarih bilgisini toplar."""
    info = {"path": path, "exists": path.is_file()}
    if not info["exists"]:
        return info
    stat = path.stat()
    info["size_bytes"] = stat.st_size
    info["mtime"] = datetime.fromtimestamp(stat.st_mtime)
    info["lines"] = count_lines(path)
    return info


def rel(path: Path) -> str:
    """Rapor icin proje kokune gore yol metni."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def pct(count: int, total: int) -> str:
    """Yuzde metni; toplam 0 ise oran hesaplanmaz."""
    if not total:
        return NOT_MEASURED
    return f"%{100.0 * count / total:.1f}"


def to_float(value):
    """Metni float'a cevirir; olmuyorsa None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    """Metni int'e cevirir; olmuyorsa None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_yaml(path: Path):
    """YAML dosyasini okur; (veri, hata_mesaji) dondurur."""
    if not path.is_file():
        return None, f"dosya yok: {rel(path)}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle), None
    except (yaml.YAMLError, OSError) as exc:
        return None, f"okunamadi: {exc}"


def read_csv_rows(path: Path):
    """CSV'yi okur; (satirlar, kolon adlari, hata) dondurur."""
    if not path.is_file():
        return [], [], f"dosya yok: {rel(path)}"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        return rows, fieldnames, None
    except OSError as exc:
        return [], [], f"okunamadi: {exc}"


def type_name(value) -> str:
    """Rapor icin deger tipi adi (liste ise uzunluguyla birlikte)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return f"liste[{len(value)}]"
    if isinstance(value, dict):
        return "sozluk"
    if value is None:
        return "null"
    return type(value).__name__


# --------------------------------------------------------------------------
# Bolum 1 - dosya varligi
# --------------------------------------------------------------------------

def build_file_list(cameras, config_path: Path):
    """Faz 3'un okuyacagi dosyalarin listesini uretir."""
    items = [
        ("kamera config", config_path),
        ("bolge tanimi", CONFIG_DIR / "zone_params.yaml"),
        ("yon kural tablosu", CONFIG_DIR / "direction_mapping.yaml"),
    ]
    for camera_id in cameras:
        items.append((f"{camera_id} bolge poligonlari", CONFIG_DIR / f"zones_{camera_id}.yaml"))
    for camera_id in cameras:
        items.append((f"{camera_id} takip kayitlari", TRACKS_DIR / f"tracks_{camera_id}.jsonl"))
    for camera_id in cameras:
        items.append((f"{camera_id} bolge olaylari", ZONES_DIR / f"zone_events_{camera_id}.csv"))
        items.append((f"{camera_id} ziyaret ozeti", ZONES_DIR / f"zone_tracks_{camera_id}.csv"))
        items.append((f"{camera_id} etiketli ziyaret", ZONES_DIR / f"zone_tracks_mapped_{camera_id}.csv"))
    return items


def section_files(cameras, config_path: Path, blockers):
    """Bolum 1 metnini uretir; eksik dosyalari engel listesine ekler."""
    lines = ["## 1. Dosya varligi", "",
             "| dosya | rol | var mi | boyut | satir | son degisiklik |",
             "|---|---|---|---|---|---|"]
    missing = []
    for role, path in build_file_list(cameras, config_path):
        info = file_info(path)
        if not info["exists"]:
            missing.append(path)
            lines.append(f"| `{rel(path)}` | {role} | **{MISSING}** | - | - | - |")
            continue
        line_count = info["lines"] if info["lines"] is not None else NOT_MEASURED
        lines.append(
            f"| `{rel(path)}` | {role} | var | {human_size(info['size_bytes'])} "
            f"({info['size_bytes']} bayt) | {line_count} | "
            f"{info['mtime']:%Y-%m-%d %H:%M:%S} |"
        )
    for path in missing:
        blockers.append(f"Girdi dosyasi eksik: `{rel(path)}`")

    total = len(build_file_list(cameras, config_path))
    lines += ["",
              f"Bu tabloya gore beklenen {total} dosyadan {total - len(missing)} tanesi diskte, "
              f"{len(missing)} tanesi eksiktir."]
    return "\n".join(lines), missing


# --------------------------------------------------------------------------
# Bolum 2 - sema
# --------------------------------------------------------------------------

def section_schema(cameras, mapped_fields, mapped_errors, blockers):
    """Bolum 2: zone_tracks_mapped_<cam>.csv kolonlarini beklenenle karsilastirir."""
    lines = ["## 2. Sema - zone_tracks_mapped_<cam>.csv", "",
             f"Beklenen kolon sayisi: {len(EXPECTED_MAPPED_COLUMNS)}", ""]
    clean_cameras = 0
    for camera_id in cameras:
        path = ZONES_DIR / f"zone_tracks_mapped_{camera_id}.csv"
        lines.append(f"### {camera_id}")
        lines.append("")
        error = mapped_errors.get(camera_id)
        if error:
            lines += [f"- Okunamadi: {error}", ""]
            blockers.append(f"{camera_id}: `{rel(path)}` okunamadi ({error}).")
            continue

        fields = mapped_fields.get(camera_id, [])
        lines.append(f"- Gercek kolon sayisi: {len(fields)}")
        lines.append(f"- Gercek kolon listesi: `{fields}`")

        expected = list(EXPECTED_MAPPED_COLUMNS)
        missing = [name for name in expected if name not in fields]
        extra = [name for name in fields if name not in expected]
        common_expected = [name for name in expected if name in fields]
        common_actual = [name for name in fields if name in expected]
        misordered = []
        if common_expected != common_actual:
            for index, (want, got) in enumerate(zip(common_expected, common_actual)):
                if want != got:
                    misordered.append(f"konum {index}: beklenen `{want}`, gercek `{got}`")

        lines.append(f"- Eksik kolon ({len(missing)}): "
                     + (f"`{missing}`" if missing else "yok"))
        lines.append(f"- Fazla kolon ({len(extra)}): "
                     + (f"`{extra}`" if extra else "yok"))
        lines.append(f"- Sira farki ({len(misordered)}): "
                     + ("; ".join(misordered) if misordered else "yok"))
        lines.append("")

        if missing or extra or misordered:
            detail = []
            if missing:
                detail.append(f"eksik {missing}")
            if extra:
                detail.append(f"fazla {extra}")
            if misordered:
                detail.append(f"{len(misordered)} kolon farkli sirada")
            blockers.append(f"{camera_id}: `zone_tracks_mapped_{camera_id}.csv` semasi "
                            f"beklenenden farkli ({'; '.join(detail)}).")
        else:
            clean_cameras += 1

    lines.append(f"Bu tabloya gore {clean_cameras} / {len(cameras)} kamerada kolon adi, "
                 f"sayisi ve sirasi beklenen {len(EXPECTED_MAPPED_COLUMNS)} kolonla birebir aynidir.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Bolum 3 - aday havuzlari
# --------------------------------------------------------------------------

def section_pools(cameras, mapped_rows):
    """Bolum 3: zone_id x movement_label caprazi ve yon bazli aday havuzlari."""
    lines = ["## 3. Aday havuzlari", ""]
    counts = Counter()
    for camera_id in cameras:
        for row in mapped_rows.get(camera_id, []):
            counts[(camera_id, row.get("zone_id", ""), row.get("movement_label", ""))] += 1

    for camera_id in cameras:
        rows = mapped_rows.get(camera_id, [])
        lines += [f"### {camera_id} - zone_id x movement_label", ""]
        if not rows:
            lines += ["- Veri okunamadi veya bos.", ""]
            continue
        zone_ids = sorted({row.get("zone_id", "") for row in rows})
        movements = sorted({row.get("movement_label", "") for row in rows})
        header = "| zone_id | " + " | ".join(movements) + " | toplam |"
        lines.append(header)
        lines.append("|" + "---|" * (len(movements) + 2))
        for zone_id in zone_ids:
            cells = [str(counts[(camera_id, zone_id, movement)]) for movement in movements]
            row_total = sum(counts[(camera_id, zone_id, movement)] for movement in movements)
            lines.append(f"| {zone_id} | " + " | ".join(cells) + f" | {row_total} |")
        col_totals = [str(sum(counts[(camera_id, zone_id, movement)] for zone_id in zone_ids))
                      for movement in movements]
        lines.append("| **toplam** | " + " | ".join(col_totals) + f" | {len(rows)} |")
        lines.append("")

    lines += ["### Yon bazli aday havuzlari", ""]
    ceilings = []
    for movement, (src_cam, src_zone), (dst_cam, dst_zone) in DIRECTION_PAIRS:
        n_src = counts[(src_cam, src_zone, movement)]
        n_dst = counts[(dst_cam, dst_zone, movement)]
        ceiling = min(n_src, n_dst)
        ceilings.append((movement, n_src, n_dst, ceiling))
        lines.append(f"- Yon {movement}: {src_cam}/{src_zone} {n_src} ziyaret -> "
                     f"{dst_cam}/{dst_zone} {n_dst} ziyaret, teorik tavan min({n_src},{n_dst}) = {ceiling}")
    total_ceiling = sum(item[3] for item in ceilings)
    lines += ["",
              f"- Iki yonun teorik tavan toplami: {total_ceiling} eslesme",
              "",
              f"Bu tabloya gore uretilebilecek eslesme sayisi en fazla {total_ceiling}'dir; "
              f"bu sayi ust sinirdir, gercek eslesme sayisi bunun altinda kalir."]
    return "\n".join(lines), ceilings, total_ceiling


# --------------------------------------------------------------------------
# Bolum 4 - zaman
# --------------------------------------------------------------------------

def section_time(cameras, mapped_rows, camera_config):
    """Bolum 4: ziyaret zaman araliklari ve fps degerleri."""
    lines = ["## 4. Zaman", "",
             "| kamera | t_enter min | t_enter maks | t_exit maks | video suresi tahmini (sn) "
             "| fps (cameras.yaml) | clock_offset_seconds |",
             "|---|---|---|---|---|---|---|"]
    spans = {}
    for camera_id in cameras:
        rows = mapped_rows.get(camera_id, [])
        enters = [value for value in (to_float(row.get("t_enter")) for row in rows) if value is not None]
        exits = [value for value in (to_float(row.get("t_exit")) for row in rows) if value is not None]
        entry = camera_config.get(camera_id, {})
        fps = entry.get("fps", MISSING)
        offset = entry.get("clock_offset_seconds", MISSING)
        if enters and exits:
            spans[camera_id] = max(exits)
            lines.append(f"| {camera_id} | {min(enters):.3f} | {max(enters):.3f} | {max(exits):.3f} "
                         f"| {max(exits):.3f} | {fps} | {offset} |")
        else:
            lines.append(f"| {camera_id} | {NOT_MEASURED} | {NOT_MEASURED} | {NOT_MEASURED} "
                         f"| {NOT_MEASURED} | {fps} | {offset} |")

    offsets = {camera_id: camera_config.get(camera_id, {}).get("clock_offset_seconds")
               for camera_id in cameras}
    lines += ["",
              "> UYARI: timestamp degerleri kendi videosuna gore saniyedir; "
              "ortak saat henuz kurulmamistir.",
              ""]
    offset_text = ", ".join(f"{camera_id}={offsets[camera_id]}" for camera_id in cameras)
    lines.append(f"- `clock_offset_seconds` diskteki degerler: {offset_text}")
    if spans:
        span_text = ", ".join(f"{camera_id}={spans[camera_id]:.3f} sn" for camera_id in sorted(spans))
        lines.append(f"- Ziyaret verisinin kapsadigi sure (t_exit maks): {span_text}")
    lines += ["",
              "Bu tabloya gore iki kameranin zaman ekseni ayri ayri olculmustur; "
              "aralarindaki gercek saat farki bu veriden olculememektedir."]
    return "\n".join(lines), spans, offsets


# --------------------------------------------------------------------------
# Bolum 5 - veri kalitesi bayraklari
# --------------------------------------------------------------------------

def quality_flags(rows):
    """Tek kamera icin bayrak sayilarini dondurur."""
    flags = Counter()
    for row in rows:
        if row.get("exit_reason") == TRACK_END_REASON:
            flags["track_end"] += 1
        if row.get("movement_label") == "other":
            flags["movement_other"] += 1
        if (row.get("direction_label") in AMBIGUOUS_DIRECTIONS
                and row.get("movement_label") in REAL_MOVEMENTS):
            flags["ambiguous_direction"] += 1
        dwell = to_float(row.get("dwell_s"))
        if dwell is not None and dwell > DWELL_LONG_S:
            flags["long_dwell"] += 1
        visit_index = to_int(row.get("visit_index"))
        if visit_index is not None and visit_index > 0:
            flags["repeat_visit"] += 1
        distance = to_float(row.get("distance_px"))
        if distance is not None and distance < WEAK_DISTANCE_PX:
            flags["weak_distance"] += 1
    return flags


def section_quality(cameras, mapped_rows):
    """Bolum 5: her bayrak icin sayi ve yuzde."""
    definitions = [
        ("track_end", f'`exit_reason == "{TRACK_END_REASON}"` (t_exit guvenilmez)'),
        ("movement_other", '`movement_label == "other"` (yon atanamadi)'),
        ("ambiguous_direction",
         f'`direction_label` {list(AMBIGUOUS_DIRECTIONS)} icinde ama gercek movement_label almis'),
        ("long_dwell", f"`dwell_s > {DWELL_LONG_S}` (durmus arac adayi)"),
        ("repeat_visit", "`visit_index > 0` (ayni track bolgeye birden cok kez girmis)"),
        ("weak_distance", f"`distance_px < {WEAK_DISTANCE_PX:.0f}` (yon kaniti zayif)"),
    ]
    per_camera = {camera_id: quality_flags(mapped_rows.get(camera_id, [])) for camera_id in cameras}
    totals = {camera_id: len(mapped_rows.get(camera_id, [])) for camera_id in cameras}
    grand_total = sum(totals.values())

    lines = ["## 5. Veri kalitesi bayraklari", "",
             "Yuzdeler ilgili kameradaki ziyaret satiri sayisina goredir.", "",
             "| bayrak | " + " | ".join(f"{camera_id} (n={totals[camera_id]})" for camera_id in cameras)
             + f" | toplam (n={grand_total}) |",
             "|" + "---|" * (len(cameras) + 2)]
    for key, label in definitions:
        cells = []
        total_flag = 0
        for camera_id in cameras:
            count = per_camera[camera_id][key]
            total_flag += count
            cells.append(f"{count} ({pct(count, totals[camera_id])})")
        lines.append(f"| {label} | " + " | ".join(cells)
                     + f" | {total_flag} ({pct(total_flag, grand_total)}) |")

    worst_key, worst_label = max(
        definitions,
        key=lambda item: sum(per_camera[camera_id][item[0]] for camera_id in cameras),
    )
    worst_count = sum(per_camera[camera_id][worst_key] for camera_id in cameras)
    lines += ["",
              f"Bu tabloya gore en sik gorulen bayrak {worst_label}, "
              f"{grand_total} ziyaretin {worst_count} tanesinde ({pct(worst_count, grand_total)}) "
              "gorulmektedir."]
    return "\n".join(lines), per_camera, totals


# --------------------------------------------------------------------------
# Bolum 6 - direction_mapping denetimi
# --------------------------------------------------------------------------

def section_mapping_audit(cameras, mapped_rows, blockers):
    """Bolum 6: kural tablosunun gercek bolgelerle ve gercek veriyle uyumu."""
    lines = ["## 6. `direction_mapping.yaml` denetimi", ""]
    mapping_path = CONFIG_DIR / "direction_mapping.yaml"
    data, error = load_yaml(mapping_path)
    if error:
        lines += [f"- Kural tablosu okunamadi: {error}", "",
                  "Bu tabloya gore kural denetimi yapilamamistir."]
        blockers.append(f"`{rel(mapping_path)}` okunamadi: {error}")
        return "\n".join(lines)

    default_movement = (data or {}).get("default_movement", MISSING)
    rules = [rule for rule in ((data or {}).get("rules") or []) if isinstance(rule, dict)]
    lines.append(f"- `default_movement`: `{default_movement}`")
    lines.append(f"- Kural sayisi: {len(rules)}")

    # gercek bolge kimlikleri zones_<cam>.yaml'dan okunur
    real_zones = {}
    zone_errors = {}
    for camera_id in cameras:
        zones_path = CONFIG_DIR / f"zones_{camera_id}.yaml"
        zone_data, zone_error = load_yaml(zones_path)
        if zone_error:
            zone_errors[camera_id] = zone_error
            continue
        real_zones[camera_id] = [zone.get("zone_id") for zone in ((zone_data or {}).get("zones") or [])
                                 if isinstance(zone, dict) and zone.get("zone_id")]
    for camera_id, zone_error in zone_errors.items():
        lines.append(f"- UYARI: `zones_{camera_id}.yaml` okunamadi: {zone_error}")
        blockers.append(f"`zones_{camera_id}.yaml` okunamadi: {zone_error}")
    for camera_id in cameras:
        if camera_id in real_zones:
            lines.append(f"- `zones_{camera_id}.yaml` gercek zone_id listesi: `{real_zones[camera_id]}`")
    lines.append("")

    # 6a - olu kurallar
    dead = []
    for index, rule in enumerate(rules, start=1):
        camera_id = str(rule.get("camera_id", ""))
        zone_id = str(rule.get("zone_id", ""))
        if camera_id not in real_zones:
            reason = ("kamera cameras.yaml'da yok" if camera_id not in cameras
                      else f"zones_{camera_id}.yaml okunamadi")
            dead.append((index, camera_id, zone_id, reason))
        elif zone_id not in real_zones[camera_id]:
            dead.append((index, camera_id, zone_id, f"zone_id `zones_{camera_id}.yaml` icinde yok"))

    lines += ["### 6a. Olu kurallar (gercek bir bolgeye karsilik gelmeyen)", ""]
    if dead:
        lines += ["| kural # | camera_id | zone_id | neden |", "|---|---|---|---|"]
        for index, camera_id, zone_id, reason in dead:
            lines.append(f"| {index} | {camera_id} | {zone_id} | {reason} |")
    else:
        lines.append("Olu kural yok.")
    lines.append("")

    # 6b - hicbir kurala dusmeyen veri kombinasyonlari
    rule_keys = {tuple(str(rule.get(field, "")) for field in KEY_FIELDS) for rule in rules}
    unmatched = Counter()
    matched_total = 0
    default_column = Counter()
    for camera_id in cameras:
        for row in mapped_rows.get(camera_id, []):
            key = tuple(str(row.get(field, "")) for field in KEY_FIELDS)
            if key in rule_keys:
                matched_total += 1
            else:
                unmatched[key] += 1
            if row.get("mapping_rule") == RULE_DEFAULT:
                default_column[camera_id] += 1

    total_rows = sum(len(mapped_rows.get(camera_id, [])) for camera_id in cameras)
    unmatched_total = sum(unmatched.values())
    lines += ["### 6b. Hicbir kurala dusmeyen veri kombinasyonlari (default_movement'e duser)", ""]
    if unmatched:
        lines += ["| camera_id | zone_id | lane | direction_label | ziyaret |",
                  "|---|---|---|---|---|"]
        for key, count in sorted(unmatched.items(), key=lambda item: (-item[1], item[0])):
            lines.append("| " + " | ".join(key) + f" | {count} |")
    else:
        lines.append("Veride kural disinda kalan kombinasyon yok.")
    lines += ["",
              f"- Kurala dusen ziyaret: {matched_total} / {total_rows} ({pct(matched_total, total_rows)})",
              f"- Kurala dusmeyen ziyaret: {unmatched_total} / {total_rows} ({pct(unmatched_total, total_rows)})",
              f"- Capraz kontrol, CSV'deki `mapping_rule == \"{RULE_DEFAULT}\"` satirlari: "
              + (", ".join(f"{camera_id}={default_column[camera_id]}" for camera_id in cameras)
                 or "yok"),
              "",
              f"Bu tabloya gore kural tablosundaki {len(rules)} kuraldan {len(dead)} tanesi olu, "
              f"{total_rows} ziyaretin {unmatched_total} tanesi hicbir kurala dusmemektedir."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Bolum 7 - hints
# --------------------------------------------------------------------------

def scan_tracks(path: Path, hints_sample: int, max_records: int):
    """tracks_<cam>.jsonl dosyasini tarar; alan varligi ve hints tiplerini toplar."""
    result = {
        "exists": path.is_file(),
        "parsed": 0,
        "bad_lines": 0,
        "field_present": Counter(),
        "hints_records": 0,
        "hints_missing": 0,
        "hint_key_counts": Counter(),
        "hint_types": {},
        "track_ids": set(),
        "error": None,
    }
    if not result["exists"]:
        result["error"] = f"dosya yok: {rel(path)}"
        return result
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if max_records and result["parsed"] >= max_records:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    result["bad_lines"] += 1
                    continue
                result["parsed"] += 1
                for field in BRIEF_TRACK_FIELDS:
                    if field in record:
                        result["field_present"][field] += 1
                track_id = record.get("track_id")
                if track_id is not None:
                    result["track_ids"].add(str(track_id))
                # hints yalnizca ilk N kayit uzerinden incelenir
                if result["parsed"] <= hints_sample:
                    hints = record.get("hints")
                    if isinstance(hints, dict):
                        result["hints_records"] += 1
                        for key, value in hints.items():
                            result["hint_key_counts"][key] += 1
                            result["hint_types"].setdefault(key, Counter())[type_name(value)] += 1
                    else:
                        result["hints_missing"] += 1
    except OSError as exc:
        result["error"] = f"okunamadi: {exc}"
    return result


def section_hints(cameras, track_scans, hints_sample, blockers):
    """Bolum 7: hints alaninin anahtar ve tipleri."""
    lines = [f"## 7. `hints` alani (her kameranin ilk {hints_sample} kaydi)", "",
             "Beklenen: "
             + ", ".join(f"`{key}` ({spec})" for key, spec in EXPECTED_HINTS.items()) + "", ""]
    deviations = 0
    for camera_id in cameras:
        scan = track_scans.get(camera_id, {})
        lines += [f"### {camera_id}", ""]
        if scan.get("error"):
            lines += [f"- Okunamadi: {scan['error']}", ""]
            blockers.append(f"{camera_id}: `tracks_{camera_id}.jsonl` okunamadi ({scan['error']}).")
            deviations += 1
            continue
        sampled = min(scan["parsed"], hints_sample)
        lines.append(f"- Incelenen kayit: {sampled} (dosyada cozulen toplam: {scan['parsed']}, "
                     f"bozuk satir: {scan['bad_lines']})")
        lines.append(f"- `hints` sozlugu bulunan kayit: {scan['hints_records']} / {sampled}, "
                     f"bulunmayan: {scan['hints_missing']}")
        lines.append("")
        lines += ["| hints anahtari | bulundugu kayit | gorulen tipler | beklenen | uyum |",
                  "|---|---|---|---|---|"]
        seen_keys = list(scan["hint_types"].keys())
        for key in sorted(set(seen_keys) | set(EXPECTED_HINTS)):
            count = scan["hint_key_counts"][key]
            types = scan["hint_types"].get(key)
            types_text = ", ".join(f"{name} ({n})" for name, n in sorted(types.items())) if types else "-"
            expected = EXPECTED_HINTS.get(key, "(beklenmeyen anahtar)")
            ok = evaluate_hint(key, types, count, sampled)
            if ok != "evet":
                deviations += 1
            lines.append(f"| `{key}` | {count} | {types_text} | {expected} | {ok} |")
        lines.append("")

    lines.append(f"Bu tabloya gore ornekleme icinde beklenen hints semasindan "
                 f"{deviations} sapma bulunmustur.")
    return "\n".join(lines), deviations


def evaluate_hint(key, types, count, sampled) -> str:
    """Tek hints anahtarinin beklenen tiple uyumunu degerlendirir."""
    if key not in EXPECTED_HINTS:
        return "hayir (beklenmeyen anahtar)"
    if not types or count == 0:
        return "hayir (bulunamadi)"
    if count != sampled:
        return f"hayir (yalnizca {count}/{sampled} kayitta)"
    names = set(types.keys())
    if key == "dominant_color":
        return "evet" if names == {"liste[3]"} else f"hayir (gorulen: {sorted(names)})"
    if key == "size_class":
        return "evet" if names == {"str"} else f"hayir (gorulen: {sorted(names)})"
    if key == "aspect_ratio":
        return "evet" if names <= {"float", "int"} else f"hayir (gorulen: {sorted(names)})"
    return "evet"


# --------------------------------------------------------------------------
# Faz 3 gecis degerlendirmesi
# --------------------------------------------------------------------------

def section_verdict(cameras, mapped_rows, mapped_fields, track_scans, ceilings,
                    total_ceiling, quality_per_camera, quality_totals, blockers):
    """Rapor sonu: brief sarti, gercekci mertebe ve devam durumu."""
    lines = ["## FAZ 3'E GECIS DEGERLENDIRMESI", ""]

    # 1) brief sarti
    lines += ["### 1. Brief'in Faz 3 gecis sarti", "",
              "Sart: her arac kaydinda en az `timestamp`, `camera_id`, `track_id`, `zone_id` "
              "ve zemin/ayak noktasi (piksel) bulunmali.", "",
              "| kamera | timestamp | camera_id | track_id | foot_point | cozulen kayit |",
              "|---|---|---|---|---|---|"]
    track_ok = True
    for camera_id in cameras:
        scan = track_scans.get(camera_id, {})
        if scan.get("error"):
            lines.append(f"| {camera_id} | {NOT_MEASURED} | {NOT_MEASURED} | {NOT_MEASURED} "
                         f"| {NOT_MEASURED} | 0 |")
            track_ok = False
            continue
        parsed = scan["parsed"]
        cells = []
        for field in BRIEF_TRACK_FIELDS:
            count = scan["field_present"][field]
            cells.append(f"{count} ({pct(count, parsed)})")
            if count != parsed:
                track_ok = False
        lines.append(f"| {camera_id} | " + " | ".join(cells) + f" | {parsed} |")

    lines += ["", "`zone_id` alani takip kaydinda degil, ziyaret ozetindedir; "
                  "iki dosya `camera_id` + `track_id` uzerinden birlesir:", "",
              "| kamera | zone_id kolonu | foot kolonlari | ziyaret satiri | "
              "ziyaretteki track_id'lerin takip dosyasindaki karsiligi |",
              "|---|---|---|---|---|"]
    zone_ok = True
    for camera_id in cameras:
        fields = mapped_fields.get(camera_id, [])
        rows = mapped_rows.get(camera_id, [])
        has_zone = "zone_id" in fields
        foot_columns = [name for name in ("start_foot_x", "start_foot_y", "end_foot_x", "end_foot_y")
                        if name in fields]
        scan = track_scans.get(camera_id, {})
        track_ids = scan.get("track_ids", set())
        visit_ids = {str(row.get("track_id")) for row in rows}
        resolved = len(visit_ids & track_ids) if track_ids else 0
        if not has_zone or len(foot_columns) != 4 or (visit_ids and resolved != len(visit_ids)):
            zone_ok = False
        lines.append(f"| {camera_id} | {'var' if has_zone else MISSING} | "
                     f"{len(foot_columns)}/4 ({', '.join(foot_columns) or 'yok'}) | {len(rows)} | "
                     f"{resolved} / {len(visit_ids)} benzersiz track_id |")

    verdict_1 = ("Saglaniyor" if (track_ok and zone_ok)
                 else "Kismen saglaniyor / dogrulanamadi (yukaridaki sayilara bakiniz)")
    lines += ["",
              f"- **Sonuc: {verdict_1}.** Kanit kolonlari: takip kaydinda "
              f"`timestamp`, `camera_id`, `track_id`, `foot_point`; ziyaret ozetinde "
              f"`zone_id`, `start_foot_x/y`, `end_foot_x/y`, `t_enter`, `t_exit`.", ""]

    # 2) gercekci mertebe
    lines += ["### 2. Gercekci eslesme sayisi mertebesi", ""]
    total_rows = sum(quality_totals.values())
    other_total = sum(quality_per_camera[camera_id]["movement_other"] for camera_id in cameras)
    weak_total = sum(quality_per_camera[camera_id]["weak_distance"] for camera_id in cameras)
    track_end_total = sum(quality_per_camera[camera_id]["track_end"] for camera_id in cameras)
    for movement, n_src, n_dst, ceiling in ceilings:
        lines.append(f"- {movement}: kaynak {n_src}, hedef {n_dst}, teorik tavan {ceiling}")
    lines += [f"- Iki yonun teorik tavan toplami: **{total_ceiling}** eslesme",
              f"- Ayni veride yon atanamamis (`other`) ziyaret: {other_total} / {total_rows} "
              f"({pct(other_total, total_rows)})",
              f"- Yon kaniti zayif (`distance_px < {WEAK_DISTANCE_PX:.0f}`) ziyaret: "
              f"{weak_total} / {total_rows} ({pct(weak_total, total_rows)})",
              f"- `exit_reason == \"{TRACK_END_REASON}\"` ziyaret: {track_end_total} / {total_rows} "
              f"({pct(track_end_total, total_rows)})",
              "",
              f"- **Sonuc:** mertebe on-larla olculur, yuzlerle degil. Ust sinir {total_ceiling} "
              "eslesmedir; her kaynak ziyaretin hedefte karsiligi olacaginin garantisi yoktur "
              "(kor bolgede baska yola sapan, duran veya takibi kopan araclar bu sayiyi dusurur). "
              "Beklenen gercek sayi bu ust sinirin altindadir ve bu script ile olculemez; "
              "ancak Faz 3 eslestirmesi calistirildiktan sonra dogrulanabilir.", ""]

    # 3) devam durumu
    lines += ["### 3. Devam edilebilir mi?", ""]
    if blockers:
        lines.append(f"- Olculen ENGEL sayisi: **{len(blockers)}**. Bu engeller giderilmeden "
                     "Faz 3 eslestirmesine gecilmemelidir:")
        for item in blockers:
            lines.append(f"  - {item}")
    else:
        lines.append("- Bu script'in kontrol ettigi kalemlerde ENGEL olcumu yoktur "
                     "(eksik dosya, sema farki, okunamayan girdi bulunmadi).")
    lines += ["",
              "- Faz 3'e girmeden once acik kalan olculmus riskler:",
              f"  - Ortak saat kurulmamis: `clock_offset_seconds` her kamerada diskteki degeriyle "
              "raporun 4. bolumunde listelenmistir; kameralar arasi gercek zaman farki bu veriden "
              "olculememektedir. Transit suresi bir eslestirme sinyali olacaksa bu deger olculmelidir.",
              f"  - Yon atanamamis ziyaretler: {other_total} / {total_rows} "
              f"({pct(other_total, total_rows)}) satir `other` etiketli olup aday havuzuna girmez.",
              f"  - `{TRACK_END_REASON}` ile biten {track_end_total} ziyarette `t_exit` gercek bolge "
              "cikisini degil takibin kopus anini gosterir; zaman tabanli skorlamada bu satirlar "
              "ayri ele alinmalidir.",
              "",
              "Bu maddeler olcum sonucudur; Faz 3'e gecis karari kullaniciya aittir."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Rapor
# --------------------------------------------------------------------------

def load_camera_config(config_path: Path):
    """cameras.yaml icindeki kamera kayitlarini dondurur."""
    data, error = load_yaml(config_path)
    if error:
        raise SystemExit(f"HATA: kamera config okunamadi: {error}")
    entries = {}
    order = []
    for entry in ((data or {}).get("cameras") or []):
        if not isinstance(entry, dict) or not entry.get("camera_id"):
            continue
        camera_id = str(entry["camera_id"])
        entries[camera_id] = entry
        order.append(camera_id)
    if not order:
        raise SystemExit(f"HATA: cameras.yaml icinde kamera tanimi yok: {config_path}")
    return order, entries


def build_report(args) -> str:
    """Tum bolumleri sirayla uretir ve markdown metnini dondurur."""
    config_path = args.config
    cameras, camera_config = load_camera_config(config_path)
    blockers = []

    files_text, missing_files = section_files(cameras, config_path, blockers)

    mapped_rows = {}
    mapped_fields = {}
    mapped_errors = {}
    for camera_id in cameras:
        path = ZONES_DIR / f"zone_tracks_mapped_{camera_id}.csv"
        rows, fields, error = read_csv_rows(path)
        mapped_rows[camera_id] = rows
        mapped_fields[camera_id] = fields
        if error:
            mapped_errors[camera_id] = error

    schema_text = section_schema(cameras, mapped_fields, mapped_errors, blockers)
    pools_text, ceilings, total_ceiling = section_pools(cameras, mapped_rows)
    time_text, _spans, _offsets = section_time(cameras, mapped_rows, camera_config)
    quality_text, quality_per_camera, quality_totals = section_quality(cameras, mapped_rows)
    mapping_text = section_mapping_audit(cameras, mapped_rows, blockers)

    track_scans = {}
    for camera_id in cameras:
        track_scans[camera_id] = scan_tracks(
            TRACKS_DIR / f"tracks_{camera_id}.jsonl", args.hints_sample, args.max_records
        )
    hints_text, _deviations = section_hints(cameras, track_scans, args.hints_sample, blockers)

    verdict_text = section_verdict(cameras, mapped_rows, mapped_fields, track_scans, ceilings,
                                   total_ceiling, quality_per_camera, quality_totals, blockers)

    stamp = datetime.now()
    head = [
        "# Faz 3 - Girdi Envanteri",
        "",
        f"Bu rapor `scripts/03_check_inputs.py` tarafindan {stamp:%Y-%m-%d %H:%M:%S} tarihinde "
        "uretildi. Script salt okur; girdi dosyalari degistirilmemistir.",
        "",
        f"- Proje koku: `{PROJECT_ROOT}`",
        f"- Kamera config: `{rel(config_path)}`",
        f"- Kameralar: {', '.join(cameras)}",
        f"- hints ornegi: her kameranin ilk {args.hints_sample} kaydi"
        + (f"; takip taramasi en fazla {args.max_records} kayit" if args.max_records else ""),
        "",
    ]

    if blockers:
        head += [f"## ENGEL ({len(blockers)})", ""]
        head += [f"- {item}" for item in blockers]
        head += ["", "Yukaridaki maddeler giderilmeden Faz 3 eslestirmesine gecilmemelidir.", ""]
    else:
        head += ["## ENGEL (0)", "",
                 "Bu script'in kontrol ettigi kalemlerde engel olculmedi: "
                 "beklenen girdi dosyalarinin tamami diskte ve "
                 "`zone_tracks_mapped_<cam>.csv` semasi beklenen kolonlarla birebir ayni.", ""]

    body = [files_text, schema_text, pools_text, time_text, quality_text,
            mapping_text, hints_text, verdict_text]
    return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 3 girdi envanteri: eslestirme oncesi girdileri olcer ve raporlar (salt okur)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMERAS_CONFIG,
                        help="kamera config yolu (varsayilan configs/cameras.yaml)")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH,
                        help="rapor cikti yolu (varsayilan docs/faz3_girdi_envanteri.md)")
    parser.add_argument("--hints-sample", type=int, default=DEFAULT_HINTS_SAMPLE,
                        help=f"hints incelemesi icin kayit sayisi (varsayilan {DEFAULT_HINTS_SAMPLE})")
    parser.add_argument("--max-records", type=int, default=0,
                        help="tracks_<cam>.jsonl taramasinda en fazla kayit (0 = tumu)")
    args = parser.parse_args()

    for attribute in ("config", "out"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, PROJECT_ROOT / path)

    text = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Rapor yazildi : {args.out}")
    print(f"Satir sayisi  : {len(text.splitlines())}")


if __name__ == "__main__":
    main()

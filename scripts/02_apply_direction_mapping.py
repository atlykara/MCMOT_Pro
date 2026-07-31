"""Faz 2 fiziksel hareket etiketi: goruntu yonu + serit -> movement_label.

Iki modu vardir:
  --init-template : gercek veride bulunan (camera_id, zone_id, lane,
                    direction_label) kombinasyonlarini sayar ve
                    configs/direction_mapping.yaml sablonunu uretir.
  (varsayilan)    : doldurulmus kural tablosunu uygular ve
                    outputs/zones/zone_tracks_mapped_<cam>.csv yazar.

Kural tablosu KOD ICINDE tanimlanmaz; her sey YAML'dan gelir. movement degerleri
otomatik tahmin edilmez: tabloda "TODO" kalan satir varsa script calismayi
reddeder. Kameralar arasi eslestirme yapilmaz; burada yalnizca etiket yazilir.

Ornek:
    python scripts/02_apply_direction_mapping.py --init-template
    python scripts/02_apply_direction_mapping.py --camera all --verify
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
CONFIG_DIR = PROJECT_ROOT / "configs"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"
DEFAULT_MAPPING_PATH = CONFIG_DIR / "direction_mapping.yaml"

TODO_VALUE = "TODO"
DEFAULT_MOVEMENT_FALLBACK = "other"
# Bu uc etiket disinda yeni ad uretilmez.
ALLOWED_MOVEMENTS = ("camA_to_camB", "camB_to_camA", "other")

KEY_FIELDS = ("camera_id", "zone_id", "lane", "direction_label")
EXTRA_COLUMNS = ("movement_label", "mapping_rule")
RULE_DEFAULT = "default"

FLOW_IMBALANCE_FACTOR = 3.0   # bir yon digerinin bu katindan fazlaysa UYARI
DEFAULT_RATIO_WARN = 0.60     # default_movement orani bu esigi asarsa UYARI

TEMPLATE_HEADER = """# Görüntü yönü + şerit -> fiziksel hareket etiketi
# movement alanını elle doldur: camA_to_camB | camB_to_camA | other
# count alanı bilgi amaçlıdır, kod tarafından kullanılmaz.
"""


def load_camera_ids(config_path: Path):
    """cameras.yaml icindeki camera_id listesini dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cameras = [c.get("camera_id") for c in (data.get("cameras") or [])
               if isinstance(c, dict) and c.get("camera_id")]
    if not cameras:
        raise SystemExit(f"HATA: cameras.yaml icinde kamera tanimi yok: {config_path}")
    return cameras


def zone_tracks_path(camera_id: str) -> Path:
    return ZONES_DIR / f"zone_tracks_{camera_id}.csv"


def read_zone_tracks(path: Path):
    """zone_tracks CSV'sini okur; (satirlar, kolon adlari) dondurur."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [field for field in KEY_FIELDS if field not in fieldnames]
    if missing:
        raise SystemExit(f"HATA: {path} icinde beklenen kolonlar eksik: {missing}")
    return rows, fieldnames


# --------------------------------------------------------------------------
# PARCA 1 - sablon uretimi
# --------------------------------------------------------------------------

def collect_combinations(cameras):
    """Gercek veride bulunan anahtar kombinasyonlarini sayar."""
    counts: Counter = Counter()
    missing_files = []
    for camera_id in cameras:
        path = zone_tracks_path(camera_id)
        if not path.is_file():
            missing_files.append(path)
            continue
        rows, _ = read_zone_tracks(path)
        for row in rows:
            counts[tuple(row[field] for field in KEY_FIELDS)] += 1
    return counts, missing_files


def read_existing_mapping(path: Path):
    """Mevcut YAML'daki doldurulmus movement degerlerini korumak icin okur.

    donus: (default_movement, {anahtar: movement}, kural sirasi)
    """
    if not path.is_file():
        return None, {}, []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"HATA: mevcut {path} cozumlenemedi: {exc}")

    movements = {}
    order = []
    for rule in (data.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        key = tuple(str(rule.get(field, "")) for field in KEY_FIELDS)
        movements[key] = str(rule.get("movement") or TODO_VALUE)
        order.append(key)
    return data.get("default_movement"), movements, order


def build_template_text(entries, default_movement):
    """Sablon YAML metnini elle uretir (movement yaninda count yorumu ile)."""
    lines = [TEMPLATE_HEADER.rstrip("\n"),
             f'default_movement: "{default_movement}"',
             "rules:"]
    for (camera_id, zone_id, lane, direction_label), movement, count in entries:
        lines += [
            f'  - camera_id: "{camera_id}"',
            f'    zone_id: "{zone_id}"',
            f'    lane: "{lane}"',
            f'    direction_label: "{direction_label}"',
            f'    movement: "{movement}"        # count: {count}',
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def init_template(cameras, mapping_path: Path) -> None:
    """Kombinasyonlari sayar ve sablonu yazar; mevcut kararlar korunur."""
    counts, missing_files = collect_combinations(cameras)
    for path in missing_files:
        print(f"UYARI: ziyaret ozeti bulunamadi, atlandi: {path}")
    if not counts:
        raise SystemExit("HATA: hicbir zone_tracks_<cam>.csv okunamadi; "
                         "once 02_summarize_zone_tracks.py calistirin.")

    previous_default, previous_movements, previous_order = read_existing_mapping(mapping_path)
    default_movement = previous_default or DEFAULT_MOVEMENT_FALLBACK

    # once veride gecen kombinasyonlar (cok gecenden aza), sonra veride artik
    # gorulmeyen ama daha once doldurulmus kurallar (insan karari kaybolmasin)
    entries = []
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        entries.append((key, previous_movements.get(key, TODO_VALUE), count))
    for key in previous_order:
        if key not in counts:
            entries.append((key, previous_movements[key], 0))

    text = build_template_text(entries, default_movement)
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SystemExit(f"HATA: uretilen YAML gecersiz, yazilmadi: {exc}")

    if mapping_path.exists():
        backup = mapping_path.with_suffix(mapping_path.suffix + ".bak")
        backup.write_bytes(mapping_path.read_bytes())
        print(f"Yedek alindi  : {backup}")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(text, encoding="utf-8")

    todo_count = sum(1 for _, movement, _ in entries if movement == TODO_VALUE)
    stale = sum(1 for _, _, count in entries if count == 0)
    print(f"Yazildi       : {mapping_path}")
    if stale:
        print(f"Not           : veride artik gorulmeyen {stale} kural korundu (count: 0).")
    print(f"\n{len(entries)} kombinasyon bulundu, {todo_count} tanesi TODO")
    if todo_count:
        print("movement alanlarini elle doldurmadan uygulayici mod calismaz.")


# --------------------------------------------------------------------------
# PARCA 2 - uygulayici mod
# --------------------------------------------------------------------------

def load_mapping(path: Path):
    """Kural tablosunu okur ve dogrular; (default_movement, kural sozlugu) dondurur."""
    if not path.is_file():
        raise SystemExit(
            f"HATA: kural tablosu bulunamadi: {path}\n"
            "      Once sablonu uretin: "
            "python scripts/02_apply_direction_mapping.py --init-template"
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"HATA: {path} cozumlenemedi: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"HATA: {path} en ust duzeyi sozluk olmali.")

    default_movement = str(data.get("default_movement") or DEFAULT_MOVEMENT_FALLBACK)
    if default_movement not in ALLOWED_MOVEMENTS:
        raise SystemExit(f"HATA: {path}: default_movement '{default_movement}' gecersiz. "
                         f"Izin verilenler: {', '.join(ALLOWED_MOVEMENTS)}")

    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise SystemExit(f"HATA: {path} icinde 'rules' listesi yok veya bos.")

    mapping = {}
    todo = []
    invalid = []
    duplicates = []
    for index, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            invalid.append(f"rules[{index}]: sozluk olmali")
            continue
        key = tuple(str(rule.get(field, "")) for field in KEY_FIELDS)
        movement = str(rule.get("movement") or "")
        label = "/".join(key)
        if movement == TODO_VALUE or not movement:
            todo.append(label)
            continue
        if movement not in ALLOWED_MOVEMENTS:
            invalid.append(f"{label}: movement '{movement}' gecersiz "
                           f"(izin verilenler: {', '.join(ALLOWED_MOVEMENTS)})")
            continue
        if key in mapping:
            duplicates.append(label)
            continue
        mapping[key] = movement

    if todo:
        print(f"HATA: {path} icinde {len(todo)} satirin movement alani hala TODO. "
              "Uygulama yapilmadi.")
        print("Doldurulmasi gereken satirlar (camera_id/zone_id/lane/direction_label):")
        for label in todo:
            print(f"  - {label}")
        print(f"\nIzin verilen degerler: {', '.join(ALLOWED_MOVEMENTS)}")
        raise SystemExit(1)
    if invalid:
        print(f"HATA: {path} icinde gecersiz satirlar var:")
        for message in invalid:
            print(f"  - {message}")
        raise SystemExit(1)
    if duplicates:
        print(f"HATA: {path} icinde ayni anahtar birden fazla kez tanimli:")
        for label in duplicates:
            print(f"  - {label}")
        raise SystemExit(1)

    return default_movement, mapping


def apply_mapping(camera_id, mapping, default_movement):
    """Tek kamera icin etiketleri yazar; (satirlar, kolonlar, cikti yolu) dondurur."""
    source = zone_tracks_path(camera_id)
    if not source.is_file():
        print(f"HATA: ziyaret ozeti bulunamadi: {source}")
        return None

    rows, fieldnames = read_zone_tracks(source)
    out_fields = list(fieldnames) + [column for column in EXTRA_COLUMNS
                                     if column not in fieldnames]

    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        movement = mapping.get(key)
        if movement is None:
            row["movement_label"] = default_movement
            row["mapping_rule"] = RULE_DEFAULT
        else:
            row["movement_label"] = movement
            row["mapping_rule"] = "/".join(key)

    out_path = ZONES_DIR / f"zone_tracks_mapped_{camera_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {camera_id}: {len(rows)} satir -> {out_path}")
    return rows


def print_verification(per_camera, default_movement):
    """movement_label dagilimi, akis tutarliligi ve kural kullanimini yazar."""
    print("\n--- DOGRULAMA ---")

    print("\nmovement_label dagilimi (kamera bazinda):")
    for camera_id, rows in per_camera.items():
        counts = Counter(row["movement_label"] for row in rows)
        total = len(rows) or 1
        print(f"  {camera_id} ({len(rows)} satir):")
        for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {label:<14} {count:>6}  (%{100 * count / total:.1f})")

    print("\nAkis tutarliligi (esit olmak zorunda degil; buyuk sapma kural hatasina isaret eder):")
    for label in ("camA_to_camB", "camB_to_camA"):
        per_cam = {camera_id: sum(1 for row in rows if row["movement_label"] == label)
                   for camera_id, rows in per_camera.items()}
        detail = " | ".join(f"{camera_id}: {count}" for camera_id, count in per_cam.items())
        print(f"  {label:<14} {detail}")
        values = [count for count in per_cam.values()]
        if len(values) >= 2:
            high, low = max(values), min(values)
            if high > 0 and (low == 0 or high > FLOW_IMBALANCE_FACTOR * low):
                print(f"    UYARI: '{label}' sayilari kameralar arasinda "
                      f"{FLOW_IMBALANCE_FACTOR:g} kattan fazla farkli "
                      f"({high} vs {low}). Kural tablosu ters doldurulmus olabilir.")

    print(f"\ndefault_movement ('{default_movement}') kullanimi:")
    total_rows = sum(len(rows) for rows in per_camera.values())
    total_default = 0
    for camera_id, rows in per_camera.items():
        count = sum(1 for row in rows if row["mapping_rule"] == RULE_DEFAULT)
        total_default += count
        ratio = count / len(rows) if rows else 0.0
        print(f"  {camera_id}: {count} / {len(rows)} satir (%{100 * ratio:.1f})")
    overall = total_default / total_rows if total_rows else 0.0
    print(f"  toplam: {total_default} / {total_rows} satir (%{100 * overall:.1f})")
    if overall > DEFAULT_RATIO_WARN:
        print(f"    UYARI: satirlarin %{100 * overall:.1f}'i kural tablosuyla eslesmedi "
              f"(esik %{100 * DEFAULT_RATIO_WARN:.0f}). Kural tablosu eksik olabilir; "
              "--init-template ile yeniden uretin.")

    print("\nmapping_rule bazinda satir sayisi:")
    rule_counts: Counter = Counter()
    for rows in per_camera.values():
        rule_counts.update(row["mapping_rule"] for row in rows)
    for rule, count in sorted(rule_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {rule:<48} {count:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 2: goruntu yonu + serit kombinasyonlarina fiziksel hareket etiketi yazar."
    )
    parser.add_argument("--init-template", action="store_true",
                        help="veride gecen kombinasyonlardan kural tablosu sablonu uret ve dur")
    parser.add_argument("--camera", default="all",
                        help="kamera kimligi veya 'all' (varsayilan all)")
    parser.add_argument("--verify", action="store_true",
                        help="dagilim ve akis tutarliligi kontrollerini yazdir")
    parser.add_argument("--mapping", type=Path, default=None,
                        help=f"kural tablosu yolu (varsayilan {DEFAULT_MAPPING_PATH.name})")
    args = parser.parse_args()

    mapping_path = args.mapping if args.mapping is not None else DEFAULT_MAPPING_PATH
    if not mapping_path.is_absolute():
        mapping_path = PROJECT_ROOT / mapping_path

    all_cameras = load_camera_ids(DEFAULT_CAMERAS_CONFIG)
    if args.camera == "all":
        cameras = all_cameras
    elif args.camera in all_cameras:
        cameras = [args.camera]
    else:
        raise SystemExit(f"HATA: '{args.camera}' cameras.yaml icinde yok. "
                         f"Tanimli kameralar: {', '.join(all_cameras)}")

    if args.init_template:
        init_template(cameras, mapping_path)
        return

    default_movement, mapping = load_mapping(mapping_path)
    print(f"Kural tablosu : {mapping_path} ({len(mapping)} kural, "
          f"default_movement='{default_movement}')")

    per_camera = {}
    for camera_id in cameras:
        rows = apply_mapping(camera_id, mapping, default_movement)
        if rows is None:
            raise SystemExit("HATA: en az bir kamera islenemedi.")
        per_camera[camera_id] = rows

    if args.verify:
        print_verification(per_camera, default_movement)


if __name__ == "__main__":
    main()

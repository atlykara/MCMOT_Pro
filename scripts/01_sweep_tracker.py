"""Faz 1: tracker/model parametre karsilastirmasi (olcum, video uretmez).

Ayni video uzerinde farkli model / tracker / conf ayarlarini sirayla kosturur
ve her ayar icin ham metrikleri toplar:

  benzersiz_id  : toplam benzersiz track_id sayisi
  ort_arac      : kare basina ortalama arac sayisi (kayit / islenen kare)
  tahmini_switch: supheli ID degisimi sayisi (mcmot.track_quality ortak kurali)
  islem_fps     : isleme hizi (model yukleme haric, ilk kare sonrasi olculur)

Cikti: outputs/tracks/sweep_<camera>.csv + konsolda tablo.

Not: "en az benzersiz id" tek basina iyi degildir; arac kacirarak id azaltmak
kotudur. Bu script yorum yapmaz, ham sayilari verir.

Ornek:
    python scripts/01_sweep_tracker.py --camera camA \
        --source data/samples/camA_sample.mp4 --max-frames 300
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.detect_track import detect_and_track, select_device  # noqa: E402
from mcmot.track_quality import add_record, find_suspect_switches  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"

CSV_COLUMNS = ["ayar", "model", "tracker", "conf",
               "benzersiz_id", "ort_arac", "tahmini_switch", "islem_fps"]

# (ayar_adi, model, tracker, conf)
# (a) conf esigi kiyasi, (b) tracker kiyasi (a_conf_035 bytetrack karsiligidir),
# (c) hiz referansi icin kucuk model.
SWEEP_RUNS = (
    ("a_conf_025", "yolo11m.pt", "bytetrack.yaml", 0.25),
    ("a_conf_035", "yolo11m.pt", "bytetrack.yaml", 0.35),
    ("a_conf_050", "yolo11m.pt", "bytetrack.yaml", 0.50),
    ("b_botsort_035", "yolo11m.pt", "botsort.yaml", 0.35),
    ("c_yolo11s_035", "yolo11s.pt", "bytetrack.yaml", 0.35),
)


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_camera_entry(config_path: Path, camera_id: str) -> dict:
    """cameras.yaml'daki kamera girdisini dondurur; yoksa bos sozluk."""
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for entry in config.get("cameras") or []:
        if entry.get("camera_id") == camera_id:
            return entry
    return {}


def probe_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps or 0)


def tracker_config_path(tracker: str) -> Path:
    """Tracker yaml'ini cozer: once verilen yol, sonra ultralytics paketi."""
    path = Path(tracker)
    if path.is_file():
        return path
    candidate = PROJECT_ROOT / tracker
    if candidate.is_file():
        return candidate
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg" / "trackers" / path.name


def assert_no_reid(tracker: str) -> None:
    """Proje yasagi: gorunum/ReID kolu acik bir tracker ile kosulmaz."""
    cfg_path = tracker_config_path(tracker)
    if not cfg_path.is_file():
        raise SystemExit(f"HATA: tracker config bulunamadi, ReID kontrolu yapilamadi: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("with_reid"):
        raise SystemExit(
            f"HATA: {cfg_path} icinde with_reid acik. Proje kurali geregi ReID/gorunum "
            "modeli kullanilamaz; sweep durduruldu."
        )


def run_setting(name: str, model: str, tracker: str, conf: float,
                video_path: Path, camera_id: str, fps: float,
                max_frames, device: str) -> dict:
    """Tek ayari kosturur ve metrik satirini dondurur (dosyaya yazmaz)."""
    assert_no_reid(tracker)

    summaries = {}
    record_count = 0
    frames_seen = 0
    timed_frames = 0
    t0 = None

    for frame_result in detect_and_track(
        video_path=video_path,
        camera_id=camera_id,
        fps=fps,
        model_name=model,
        conf=conf,
        max_frames=max_frames,
        device=device,
        tracker=tracker,
    ):
        if t0 is None:
            # ilk kare model yukleme + isinma maliyetini tasir, olcume katilmaz
            t0 = time.perf_counter()
        else:
            timed_frames += 1
        frames_seen += 1
        for record in frame_result.records:
            record_count += 1
            add_record(summaries, record)

    elapsed = (time.perf_counter() - t0) if t0 is not None else 0.0
    islem_fps = timed_frames / elapsed if elapsed > 0 and timed_frames else 0.0
    ort_arac = record_count / frames_seen if frames_seen else 0.0
    switches = find_suspect_switches(summaries)

    return {
        "ayar": name,
        "model": model,
        "tracker": tracker,
        "conf": f"{conf:.2f}",
        "benzersiz_id": len(summaries),
        "ort_arac": f"{ort_arac:.2f}",
        "tahmini_switch": len(switches),
        "islem_fps": f"{islem_fps:.1f}",
    }


def print_table(rows: list) -> None:
    """Satirlari konsolda hizalanmis tablo olarak yazar."""
    header = CSV_COLUMNS
    table = [header] + [[str(r[c]) for c in header] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    sep = "-+-".join("-" * w for w in widths)
    for index, row in enumerate(table):
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracker/model parametre kiyasi (sadece metrik, video uretilmez)."
    )
    parser.add_argument("--camera", required=True, help="camera_id; cikti adinda kullanilir (or. camA)")
    parser.add_argument("--source", default=None,
                        help="video yolu; verilmezse cameras.yaml'daki video_path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="her ayarda en fazla islenecek kare sayisi (kisa deneme icin)")
    parser.add_argument("--fps", type=float, default=None,
                        help="video fps'i (verilmezse cameras.yaml, yoksa videodan okunur)")
    parser.add_argument("--only", default=None,
                        help="virgulle ayrilmis ayar adlari; yalnizca bunlar kosulur")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="kamera config dosyasi yolu")
    parser.add_argument("--out", type=Path, default=None,
                        help="cikti CSV yolu (varsayilan outputs/tracks/sweep_<camera>.csv)")
    args = parser.parse_args()

    entry = load_camera_entry(args.config, args.camera)

    if args.source is not None:
        video_path = resolve_path(args.source)
    else:
        raw = entry.get("video_path") or ""
        if not raw:
            raise SystemExit(f"HATA: '{args.camera}' icin video_path bos ve --source verilmedi.")
        video_path = resolve_path(raw)
    if not video_path.is_file():
        raise SystemExit(f"HATA: video dosyasi bulunamadi: {video_path}")

    fps = args.fps or float(entry.get("fps") or 0)
    if fps <= 0:
        fps = probe_fps(video_path)
        print(f"UYARI: fps config'te yok, videodan okundu: {fps:g}")
    if fps <= 0:
        raise SystemExit("HATA: fps belirlenemedi (--fps ile verin).")

    runs = list(SWEEP_RUNS)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        unknown = wanted - {r[0] for r in runs}
        if unknown:
            raise SystemExit(f"HATA: bilinmeyen ayar adi: {sorted(unknown)}")
        runs = [r for r in runs if r[0] in wanted]

    device = select_device()
    print(f"Device : {device}")
    print(f"Video  : {video_path}")
    print(f"FPS    : {fps:g} | kare siniri: {args.max_frames if args.max_frames else 'yok'}")
    print(f"Ayar   : {len(runs)} kosu -> {', '.join(r[0] for r in runs)}")
    print()

    rows = []
    for name, model, tracker, conf in runs:
        print(f"[{name}] model={model} tracker={tracker} conf={conf:.2f} kosuluyor...")
        row = run_setting(name, model, tracker, conf, video_path,
                          args.camera, fps, args.max_frames, device)
        rows.append(row)
        print(f"  benzersiz_id={row['benzersiz_id']} ort_arac={row['ort_arac']} "
              f"tahmini_switch={row['tahmini_switch']} islem_fps={row['islem_fps']}")

    out_path = args.out or (TRACKS_DIR / f"sweep_{args.camera}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print_table(rows)
    print()
    print("Not: dusuk benzersiz_id tek basina iyi degildir; ort_arac dusuyorsa arac kaciriliyor olabilir.")
    print("Tracker kiyasi (b): a_conf_035 (bytetrack) <-> b_botsort_035 (botsort, with_reid kapali).")
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()

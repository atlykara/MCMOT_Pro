"""Faz 1 bitis kontrolu: takip kalite raporu ve supheli ID degisimi analizi.

tracks_<camera>.jsonl dosyasini akis halinde okur; genel istatistikleri,
kisa omurlu track'leri ve supheli ID degisimi adaylarini
outputs/tracks/quality_<camera>.md dosyasina raporlar. Supheli degisimler
icin videodan ornek kareler keser.

Ornek:
    python scripts/01_track_quality.py --camera camA
"""

import argparse
import statistics
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcmot.io_utils import read_jsonl  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
SAMPLES_DIR = TRACKS_DIR / "switch_samples"

SHORT_TRACK_MAX_SEC = 1.5   # bundan kisa yasayan track'ler hayalet/bolunme adayi
SWITCH_MAX_GAP_SEC = 1.0    # eski track kaybolduktan sonra yeni dogum icin azami sure
SWITCH_MAX_DIST_PX = 75.0   # eski son foot_point ile yeni ilk foot_point arasi azami mesafe
MAX_SAMPLES = 5             # en fazla kac supheli degisim icin ornek kare kesilecek

OLD_COLOR = (0, 0, 255)     # eski track: kirmizi (BGR)
NEW_COLOR = (0, 200, 0)     # yeni track: yesil (BGR)


class TrackSummary:
    """Bir track_id'nin dogum/olum bilgisi (akis okumada guncellenir)."""

    __slots__ = ("track_id", "first_ts", "last_ts", "first_frame", "last_frame",
                 "first_foot", "last_foot", "first_bbox", "last_bbox", "count")

    def __init__(self, track_id: int, record: dict):
        self.track_id = track_id
        self.first_ts = self.last_ts = record["timestamp"]
        self.first_frame = self.last_frame = record["frame"]
        self.first_foot = self.last_foot = record["foot_point"]
        self.first_bbox = self.last_bbox = record["bbox_xyxy"]
        self.count = 1

    def update(self, record: dict) -> None:
        self.count += 1
        ts = record["timestamp"]
        if ts >= self.last_ts:
            self.last_ts = ts
            self.last_frame = record["frame"]
            self.last_foot = record["foot_point"]
            self.last_bbox = record["bbox_xyxy"]

    @property
    def lifetime(self) -> float:
        return self.last_ts - self.first_ts


def resolve_video_path(config_path: Path, camera_id: str, source: str) -> Path:
    """--source verilmisse onu, yoksa cameras.yaml'daki video_path'i cozer."""
    if source is not None:
        raw = source
    else:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        entry = next(
            (e for e in (config.get("cameras") or []) if e.get("camera_id") == camera_id),
            None,
        )
        raw = (entry or {}).get("video_path") or ""
        if not raw:
            return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / raw


def collect_summaries(jsonl_path: Path) -> tuple:
    """JSONL'i akis halinde okuyup track ozetlerini ve kare sayaclarini toplar."""
    summaries = {}
    record_count = 0
    min_frame = None
    max_frame = None
    for record in read_jsonl(jsonl_path):
        record_count += 1
        frame = record["frame"]
        min_frame = frame if min_frame is None else min(min_frame, frame)
        max_frame = frame if max_frame is None else max(max_frame, frame)
        tid = record["track_id"]
        summary = summaries.get(tid)
        if summary is None:
            summaries[tid] = TrackSummary(tid, record)
        else:
            summary.update(record)
    return summaries, record_count, min_frame, max_frame


def find_suspect_switches(summaries: dict) -> list:
    """Kaybolan track'in yakininda kisa sure icinde dogan yeni track'leri bulur.

    Donen liste: (eski_id, yeni_id, timestamp, mesafe_px) — timestamp yeni
    track'in dogum ani. Her yeni track en yakin eski track ile eslestirilir.
    """
    tracks = sorted(summaries.values(), key=lambda t: t.first_ts)
    switches = []
    for new in tracks:
        best = None
        for old in tracks:
            if old.track_id == new.track_id:
                continue
            gap = new.first_ts - old.last_ts
            if not (0.0 < gap <= SWITCH_MAX_GAP_SEC):
                continue
            dx = new.first_foot[0] - old.last_foot[0]
            dy = new.first_foot[1] - old.last_foot[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist >= SWITCH_MAX_DIST_PX:
                continue
            if best is None or dist < best[1]:
                best = (old, dist)
        if best is not None:
            old, dist = best
            switches.append((old.track_id, new.track_id, new.first_ts, dist))
    switches.sort(key=lambda s: s[2])
    return switches


def save_switch_samples(video_path: Path, camera_id: str, switches: list,
                        summaries: dict) -> list:
    """Supheli degisimler icin videodan ornek kare kaydeder; yollari dondurur."""
    import cv2

    saved = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"UYARI: video acilamadi, ornek kare kesilemedi: {video_path}")
        return saved

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for old_id, new_id, _ts, _dist in switches[:MAX_SAMPLES]:
            old = summaries[old_id]
            new = summaries[new_id]
            frame_index = new.first_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                print(f"UYARI: kare {frame_index} okunamadi ({old_id}->{new_id}), atlandi.")
                continue

            for summary, bbox, foot, color, label in (
                (old, old.last_bbox, old.last_foot, OLD_COLOR, f"eski id:{old_id}"),
                (new, new.first_bbox, new.first_foot, NEW_COLOR, f"yeni id:{new_id}"),
            ):
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                fx, fy = (int(v) for v in foot)
                cv2.circle(image, (fx, fy), 5, color, -1)

            out_path = SAMPLES_DIR / f"{camera_id}_{old_id}_{new_id}.jpg"
            cv2.imwrite(str(out_path), image)
            saved.append(out_path)
    finally:
        cap.release()
    return saved


def build_report(camera_id: str, summaries: dict, record_count: int,
                 min_frame, max_frame, switches: list, samples: list) -> str:
    lines = [f"# Takip Kalite Raporu — {camera_id}", ""]

    lifetimes = [t.lifetime for t in summaries.values()]
    frame_span = (max_frame - min_frame + 1) if max_frame is not None else 0
    avg_per_frame = record_count / frame_span if frame_span else 0.0

    lines += [
        "## Genel istatistik",
        "",
        f"- Toplam benzersiz track_id: **{len(summaries)}**",
        f"- Toplam kayit: {record_count} ({frame_span} kare araligi)",
        f"- Kare basina ortalama arac: **{avg_per_frame:.2f}**",
    ]
    if lifetimes:
        lines += [
            f"- Track yasam suresi ortalama: **{statistics.mean(lifetimes):.2f} sn**, "
            f"medyan: **{statistics.median(lifetimes):.2f} sn**",
        ]
    lines.append("")

    short = sorted(
        (t for t in summaries.values() if t.lifetime < SHORT_TRACK_MAX_SEC),
        key=lambda t: t.lifetime,
    )
    lines += [f"## Kisa omurlu track'ler (<{SHORT_TRACK_MAX_SEC} sn)", ""]
    if short:
        lines += ["| track_id | sure (sn) |", "|---|---|"]
        lines += [f"| {t.track_id} | {t.lifetime:.2f} |" for t in short]
        lines += ["", f"Toplam: {len(short)} / {len(summaries)} track "
                      "(muhtemel hayalet tespit veya track bolunmesi).", ""]
    else:
        lines += ["Kisa omurlu track yok.", ""]

    lines += [
        "## Supheli ID degisimi adaylari",
        "",
        f"Kural: bir track_id kaybolduktan sonra <= {SWITCH_MAX_GAP_SEC} sn icinde, "
        f"son foot_point'ine < {SWITCH_MAX_DIST_PX:.0f} px mesafede yeni track_id dogumu.",
        "",
    ]
    if switches:
        lines += ["| eski_id | yeni_id | timestamp (sn) | mesafe_px |", "|---|---|---|---|"]
        lines += [f"| {o} | {n} | {ts:.2f} | {d:.1f} |" for o, n, ts, d in switches]
        lines.append("")
    else:
        lines += ["Supheli ID degisimi adayi bulunamadi.", ""]

    lines += ["## Ornek kareler", ""]
    if samples:
        lines += [f"- `{p.relative_to(PROJECT_ROOT)}`" for p in samples]
    elif switches:
        lines += ["Ornek kare uretilemedi (video erisilemedi veya kare okunamadi)."]
    else:
        lines += ["Supheli degisim olmadigi icin ornek kare uretilmedi."]
    lines.append("")

    lines += ["## Yorum", "", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 1 takip kalite raporu ve supheli ID degisimi analizi."
    )
    parser.add_argument("--camera", required=True, help="cameras.yaml icindeki camera_id (or. camA)")
    parser.add_argument("--source", default=None,
                        help="ornek kareler icin video yolu; verilmezse cameras.yaml'daki video_path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="kamera config dosyasi yolu")
    args = parser.parse_args()

    jsonl_path = TRACKS_DIR / f"tracks_{args.camera}.jsonl"
    if not jsonl_path.is_file():
        raise SystemExit(f"HATA: takip dosyasi bulunamadi: {jsonl_path}")

    summaries, record_count, min_frame, max_frame = collect_summaries(jsonl_path)
    if not summaries:
        raise SystemExit(f"HATA: {jsonl_path} icinde kayit yok.")

    switches = find_suspect_switches(summaries)

    samples = []
    if switches:
        video_path = resolve_video_path(args.config, args.camera, args.source)
        if video_path is None or not video_path.is_file():
            print(f"UYARI: video bulunamadi ({video_path}); ornek kare kesilmeyecek.")
        else:
            samples = save_switch_samples(video_path, args.camera, switches, summaries)

    report = build_report(args.camera, summaries, record_count,
                          min_frame, max_frame, switches, samples)
    report_path = TRACKS_DIR / f"quality_{args.camera}.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Rapor: {report_path}")
    for p in samples:
        print(f"Ornek: {p}")


if __name__ == "__main__":
    main()

"""Faz 2 gorsel hazirlik: referans kare, hareket isi haritasi ve yon oklari.

Her kamera icin uc gorsel uretir:
  1. data/reference_frames/ref_<cam>.jpg   - videodan tek temsili kare
  2. outputs/zones/heatmap_<cam>.jpg       - foot_point yogunlugu (JET) + referans kare
  3. outputs/zones/flow_<cam>.jpg          - isi haritasi uzerine hucre bazli yon oklari

Script ROI/zone TANIMLAMAZ; sadece bolgeyi insanin cizmesi icin gorsel uretir.
tracks_<cam>.jsonl dosyalari yalnizca okunur, model yuklenmez.

Ornek:
    python scripts/02_extract_reference_frames.py --camera camA
    python scripts/02_extract_reference_frames.py --camera all --sigma 16
    python scripts/02_extract_reference_frames.py --camera camB --frame 1500
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERAS_CONFIG = PROJECT_ROOT / "configs" / "cameras.yaml"
DEFAULT_TRACKS_DIR = PROJECT_ROOT / "outputs" / "tracks"
REF_FRAMES_DIR = PROJECT_ROOT / "data" / "reference_frames"
ZONES_DIR = PROJECT_ROOT / "outputs" / "zones"

# --- gorsellestirme parametreleri ---
DEFAULT_SIGMA_PX = 12        # foot_point lekesinin gaussian sigmasi (px)
HEATMAP_WEIGHT = 0.55        # harmanlamada isi haritasinin agirligi
FRAME_WEIGHT = 0.45          # harmanlamada referans karenin agirligi
# Duran araclar tek noktada yuzlerce kayit biriktirdigi icin ham max ile
# normalize edildiginde serit yapisi bastirilir; varsayilan olarak yuksek bir
# yuzdelikte kirpilir. Literal min-max davranisi icin --norm max kullanilir.
DEFAULT_NORM_MODE = "percentile"
DEFAULT_NORM_PERCENTILE = 99.0
DEFAULT_GAMMA = 0.6          # <1 orta yogunluklu bolgeleri one cikarir
GRID_CELLS = 40              # goruntu bu sayida sutun x satir hucreye bolunur
MIN_DISPLACEMENT_PX = 40.0   # net yer degistirmesi bu esigin altindaki track'ler elenir
ARROW_MAX_LEN_PX = 70        # cizilen okun en fazla uzunlugu (px)
ARROW_COLOR = (255, 255, 255)        # ok govdesi: beyaz (BGR)
ARROW_OUTLINE_COLOR = (0, 0, 0)      # ok konturu: siyah (BGR)
JPEG_QUALITY = 95


# --------------------------------------------------------------------------
# config ve veri okuma
# --------------------------------------------------------------------------

def load_cameras(config_path):
    """cameras.yaml icindeki kamera girdilerini dondurur."""
    if not config_path.is_file():
        raise SystemExit(f"HATA: kamera config dosyasi bulunamadi: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise SystemExit(f"HATA: cameras.yaml okunamadi: {exc}")
    cameras = [c for c in (data.get("cameras") or []) if isinstance(c, dict)]
    if not cameras:
        raise SystemExit(f"HATA: cameras.yaml icinde 'cameras' listesi bos: {config_path}")
    return cameras


def resolve_path(raw):
    """Config'teki goreli yolu PROJECT_ROOT'a gore cozer."""
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_foot_points(jsonl_path):
    """JSONL'den foot_point listesi ve track basina ilk/son noktayi okur.

    Donen degerler:
        points  : [(x, y), ...] tum kayitlar
        tracks  : {track_id: {"first": (frame, x, y), "last": (frame, x, y)}}
        skipped : foot_point'i okunamayan kayit sayisi
    """
    points = []
    tracks = {}
    skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            foot = rec.get("foot_point")
            if not (isinstance(foot, list) and len(foot) == 2):
                skipped += 1
                continue
            try:
                x, y = float(foot[0]), float(foot[1])
            except (TypeError, ValueError):
                skipped += 1
                continue
            points.append((x, y))

            track_id = rec.get("track_id")
            frame = rec.get("frame")
            if track_id is None or not isinstance(frame, (int, float)):
                continue
            entry = tracks.get(track_id)
            sample = (float(frame), x, y)
            if entry is None:
                tracks[track_id] = {"first": sample, "last": sample}
            else:
                if sample[0] < entry["first"][0]:
                    entry["first"] = sample
                if sample[0] > entry["last"][0]:
                    entry["last"] = sample

    return points, tracks, skipped


# --------------------------------------------------------------------------
# gorsel uretimi
# --------------------------------------------------------------------------

def grab_reference_frame(video_path, frame_index, cfg_fps):
    """Videodan tek kare okur; (kare, kare_no, timestamp, toplam_kare) dondurur."""
    if video_path is None:
        raise RuntimeError("cameras.yaml icinde video_path tanimli degil (EKSIK).")
    if not video_path.is_file():
        raise RuntimeError(f"video dosyasi bulunamadi: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"video cv2 ile acilamadi (codec veya dosya sorunu): {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_index is None:
            if total <= 0:
                raise RuntimeError(
                    f"videonun toplam kare sayisi okunamadi: {video_path}. "
                    "--frame ile kare numarasi verin."
                )
            target = total // 2
        else:
            target = int(frame_index)
            if target < 0:
                raise RuntimeError(f"--frame negatif olamaz: {target}")
            if total > 0 and target >= total:
                raise RuntimeError(
                    f"--frame {target} videonun kare sayisindan buyuk veya esit "
                    f"(toplam {total} kare): {video_path}"
                )

        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"kare {target} okunamadi: {video_path}")

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            timestamp = pos_msec / 1000.0
        elif cfg_fps:
            timestamp = target / float(cfg_fps)
        else:
            timestamp = None
        return frame, target, timestamp, total
    finally:
        cap.release()


def build_heatmap(points, shape, sigma):
    """foot_point'lerden tek float32 dizi uzerinde yogunluk haritasi biriktirir.

    Her nokta icin ayri goruntu uretilmez: noktalar tek bir akumulatore eklenir,
    gaussian leke tek bir bulaniklastirma ile uygulanir (toplamla ayni sonuc).
    """
    height, width = shape[:2]
    acc = np.zeros((height, width), dtype=np.float32)

    inside = 0
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            acc[yi, xi] += 1.0
            inside += 1

    ksize = int(6 * sigma) | 1  # tek sayiya yuvarla
    ksize = max(ksize, 3)
    cv2.GaussianBlur(acc, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, dst=acc)
    return acc, inside


def colorize_and_blend(acc, frame, norm_mode, norm_percentile, gamma):
    """Yogunluk haritasini 0-255'e normalize eder, JET uygular, kare ile harmanlar.

    norm_mode='max'        : ham tepe degerine gore min-max normalizasyon.
    norm_mode='percentile' : pozitif degerlerin norm_percentile yuzdeligine gore
                             olcekler ve ustunu kirpar; tek noktada bekleyen
                             araclarin tepe degeri serit yapisini ezmesin diye.
    Donen: (harmanlanmis goruntu, olcek_ust_siniri, ham_tepe)
    """
    peak = float(acc.max())
    if peak <= 0.0:
        return None, None, None

    if norm_mode == "percentile":
        positive = acc[acc > 1e-6]
        vmax = float(np.percentile(positive, norm_percentile)) if positive.size else peak
        if vmax <= 0.0:
            vmax = peak
    else:
        vmax = peak

    scaled = np.clip(acc / vmax, 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    norm = (scaled * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(colored, HEATMAP_WEIGHT, frame, FRAME_WEIGHT, 0.0)
    return blended, vmax, peak


def build_flow_arrows(base_image, tracks, grid_cells):
    """Track basina net yer degistirmeyi hucrelere gruplayip ortalama oku cizer.

    Bir track, ilk ve son foot_point'inin orta noktasinin dustugu hucreye atanir.
    Donen degerler: (goruntu, kullanilan_track, ok_cizilen_hucre)
    """
    image = base_image.copy()
    height, width = image.shape[:2]
    cell_w = width / grid_cells
    cell_h = height / grid_cells

    # hucre -> [vektor toplami, adet]
    cells = {}
    used = 0
    for entry in tracks.values():
        _, x0, y0 = entry["first"]
        _, x1, y1 = entry["last"]
        dx, dy = x1 - x0, y1 - y0
        if (dx * dx + dy * dy) ** 0.5 <= MIN_DISPLACEMENT_PX:
            continue
        used += 1

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        col = min(int(mx / cell_w), grid_cells - 1) if width else 0
        row = min(int(my / cell_h), grid_cells - 1) if height else 0
        if col < 0 or row < 0:
            continue
        acc = cells.setdefault((row, col), [0.0, 0.0, 0])
        acc[0] += dx
        acc[1] += dy
        acc[2] += 1

    for (row, col), (sum_dx, sum_dy, count) in cells.items():
        mean_dx, mean_dy = sum_dx / count, sum_dy / count
        magnitude = (mean_dx ** 2 + mean_dy ** 2) ** 0.5
        if magnitude < 1e-6:
            continue
        length = min(magnitude, ARROW_MAX_LEN_PX)
        ux, uy = mean_dx / magnitude, mean_dy / magnitude

        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        start = (int(round(cx - ux * length / 2)), int(round(cy - uy * length / 2)))
        end = (int(round(cx + ux * length / 2)), int(round(cy + uy * length / 2)))

        # once kalin siyah kontur, sonra beyaz govde: isi haritasi uzerinde okunakli olsun
        cv2.arrowedLine(image, start, end, ARROW_OUTLINE_COLOR, 4, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(image, start, end, ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.35)

    return image, used, len(cells)


def save_image(path, image):
    """JPEG olarak kaydeder; basarisizlikta RuntimeError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"goruntu yazilamadi: {path}")


# --------------------------------------------------------------------------
# kamera akisi
# --------------------------------------------------------------------------

def process_camera(entry, tracks_dir, frame_index, sigma, grid_cells,
                   norm_mode, norm_percentile, gamma):
    """Tek kamera icin uc gorseli uretir; basarili olursa True dondurur."""
    camera_id = entry.get("camera_id")
    print(f"\n=== {camera_id} ===")

    video_path = resolve_path(entry.get("video_path"))
    cfg_fps = entry.get("fps") if isinstance(entry.get("fps"), (int, float)) else None

    try:
        frame, frame_no, timestamp, total = grab_reference_frame(video_path, frame_index, cfg_fps)
    except RuntimeError as exc:
        print(f"HATA: {camera_id} referans karesi alinamadi -> {exc}")
        return False

    height, width = frame.shape[:2]
    ts_text = f"{timestamp:.3f} sn" if timestamp is not None else "EKSIK"
    ref_path = REF_FRAMES_DIR / f"ref_{camera_id}.jpg"
    save_image(ref_path, frame)
    print(f"Referans kare : kare no {frame_no} / toplam {total}, timestamp {ts_text}, "
          f"cozunurluk {width}x{height}")
    print(f"                {ref_path}")

    jsonl_path = tracks_dir / f"tracks_{camera_id}.jsonl"
    if not jsonl_path.is_file():
        print(f"UYARI: takip dosyasi bulunamadi, isi haritasi ve yon oklari uretilmedi: {jsonl_path}")
        return False

    points, tracks, skipped = read_foot_points(jsonl_path)
    if skipped:
        print(f"UYARI: foot_point'i okunamayan {skipped} kayit atlandi.")
    if not points:
        print(f"UYARI: {jsonl_path} icinde gecerli foot_point yok; isi haritasi uretilmedi.")
        return False

    acc, inside = build_heatmap(points, frame.shape, sigma)
    outside = len(points) - inside
    print(f"foot_point    : toplam {len(points)}, kare icinde {inside}"
          + (f", kare disinda {outside} (atlandi)" if outside else ""))

    blended, vmax, peak = colorize_and_blend(acc, frame, norm_mode, norm_percentile, gamma)
    if blended is None:
        print("UYARI: yogunluk haritasi bos kaldi; isi haritasi uretilmedi.")
        return False

    heatmap_path = ZONES_DIR / f"heatmap_{camera_id}.jpg"
    save_image(heatmap_path, blended)
    norm_text = (f"p{norm_percentile:g} = {vmax:.4f} (ham tepe {peak:.4f})"
                 if norm_mode == "percentile" else f"max = {peak:.4f}")
    print(f"Isi haritasi  : sigma {sigma} px, harman {HEATMAP_WEIGHT}/{FRAME_WEIGHT}, "
          f"gamma {gamma:g}, olcek {norm_text}")
    print(f"                {heatmap_path}")

    flow_image, used_tracks, arrow_cells = build_flow_arrows(blended, tracks, grid_cells)
    flow_path = ZONES_DIR / f"flow_{camera_id}.jpg"
    save_image(flow_path, flow_image)
    print(f"Yon oklari    : {len(tracks)} track'ten {used_tracks} tanesi "
          f"> {MIN_DISPLACEMENT_PX:.0f} px yer degistirdi, "
          f"{grid_cells}x{grid_cells} izgarada {arrow_cells} hucrede ok cizildi")
    print(f"                {flow_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Faz 2 gorselleri: referans kare, foot_point isi haritasi ve yon oklari."
    )
    parser.add_argument("--camera", required=True,
                        help="cameras.yaml icindeki camera_id; tum kameralar icin 'all'")
    parser.add_argument("--frame", type=int, default=None,
                        help="referans kare numarasi; verilmezse videonun ortasi (toplam//2)")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_PX,
                        help=f"isi haritasi gaussian sigmasi, px (varsayilan {DEFAULT_SIGMA_PX})")
    parser.add_argument("--grid", type=int, default=GRID_CELLS,
                        help=f"yon oklari icin izgara boyutu, NxN hucre (varsayilan {GRID_CELLS})")
    parser.add_argument("--cameras-config", type=Path, default=DEFAULT_CAMERAS_CONFIG,
                        help="kamera config dosyasi yolu")
    parser.add_argument("--tracks-dir", type=Path, default=DEFAULT_TRACKS_DIR,
                        help="tracks_<cam>.jsonl dosyalarinin bulundugu dizin")
    parser.add_argument("--norm", choices=("percentile", "max"), default=DEFAULT_NORM_MODE,
                        help="isi haritasi olcekleme: yuzdelikte kirp (varsayilan) veya ham max")
    parser.add_argument("--norm-percentile", type=float, default=DEFAULT_NORM_PERCENTILE,
                        help=f"--norm percentile icin kirpma yuzdeligi (varsayilan {DEFAULT_NORM_PERCENTILE:g})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help=f"isi haritasi gamma; <1 orta yogunlugu one cikarir (varsayilan {DEFAULT_GAMMA})")
    args = parser.parse_args()

    if args.sigma <= 0:
        raise SystemExit("HATA: --sigma pozitif olmali.")
    if args.grid < 1:
        raise SystemExit("HATA: --grid en az 1 olmali.")
    if not 0 < args.norm_percentile <= 100:
        raise SystemExit("HATA: --norm-percentile 0 ile 100 arasinda olmali.")
    if args.gamma <= 0:
        raise SystemExit("HATA: --gamma pozitif olmali.")

    config_path = (args.cameras_config if args.cameras_config.is_absolute()
                   else PROJECT_ROOT / args.cameras_config)
    tracks_dir = (args.tracks_dir if args.tracks_dir.is_absolute()
                  else PROJECT_ROOT / args.tracks_dir)

    cameras = load_cameras(config_path)
    if args.camera == "all":
        selected = cameras
    else:
        selected = [c for c in cameras if c.get("camera_id") == args.camera]
        if not selected:
            known = ", ".join(str(c.get("camera_id")) for c in cameras)
            raise SystemExit(
                f"HATA: '{args.camera}' cameras.yaml icinde bulunamadi. Tanimli kameralar: {known}"
            )
    if args.frame is not None and len(selected) > 1:
        print(f"NOT: --frame {args.frame} secilen tum kameralara uygulanacak.")

    results = [process_camera(entry, tracks_dir, args.frame, args.sigma, args.grid,
                              args.norm, args.norm_percentile, args.gamma)
               for entry in selected]

    ok_count = sum(1 for r in results if r)
    print(f"\nTamamlanan kamera: {ok_count} / {len(results)}")
    if ok_count != len(results):
        raise SystemExit("HATA: en az bir kamera icin gorseller eksik uretildi (yukaridaki mesajlara bakin).")


if __name__ == "__main__":
    main()

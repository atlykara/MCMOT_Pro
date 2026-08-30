# Kayseri MOBESE Proje Kurulumu

Bu klasor, Kayseri MOBESE kamera kayitlariyla coklu kamera arac takip ve kameralar arasi eslestirme projesi icin temiz calisma kokudur.

## Standart video karari

Proje artik orijinal FPS degerleriyle degil, 20 FPS'e standardize edilmis calisma videolariyla ilerler.

```text
data/cams/camA_20fps.mp4
data/cams/camB_20fps.mp4
```

Bu karar iki nedenle alindi:

- camA ve camB farkli FPS degerlerine sahipti; ikisini 20 FPS'e cekmek zaman hesaplarini daha tutarli yapar.
- Daha dusuk FPS islem suresini azaltir ve deneyleri hizlandirir.

## Temiz klasor yapisi

```text
kayseri_mobese/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── cams/
│   ├── raw/
│   └── samples/
├── docs/
├── models/
├── outputs/
├── runs/
├── scripts/
├── src/
└── tests/
```

## Klasorlerin amaci

- `src/`: tekrar kullanilabilir Python kodlari
- `scripts/`: fazlari calistiran komut dosyalari
- `configs/`: kamera, zone, yon ve takip ayarlari
- `data/cams/`: 20 FPS calisma videolari
- `data/raw/`: gerekirse orijinal ham videolar
- `models/`: YOLO model agirliklari
- `outputs/`: uretilen takip, zone, eslestirme ve video ciktilari
- `docs/`: raporlar ve karar notlari
- `tests/`: cikti kontrati ve temel testler
- `runs/`: YOLO/Ultralytics gecici ciktilari

## Kurulum

```bash
cd ~/Desktop/kayseri_mobese
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install lapx pytest
```

## Kamera ayari

`configs/cameras.yaml` standart olarak 20 FPS videolari gosterir:

```yaml
cameras:
  - camera_id: camA
    video_path: "data/cams/camA_20fps.mp4"
    fps: 20

  - camera_id: camB
    video_path: "data/cams/camB_20fps.mp4"
    fps: 20
```

Bu nedenle Faz 1 calistirirken `--source` vermek zorunlu degildir.

## Calistirma sirasi

### Faz 1: Tespit ve tek kamera takip

```bash
python scripts/01_run_detect_track.py \
  --camera camA \
  --model models/yolo11n.pt \
  --conf 0.25 \
  --imgsz 960
```

```bash
python scripts/01_run_detect_track.py \
  --camera camB \
  --model models/yolo11n.pt \
  --conf 0.25 \
  --imgsz 960
```

### Faz 2: Zone ve yon atama

```bash
python scripts/02_preview_zones.py --camera camA
python scripts/02_preview_zones.py --camera camB
python scripts/02_assign_zones.py --camera camA
python scripts/02_assign_zones.py --camera camB
python scripts/02_summarize_zone_tracks.py --camera camA
python scripts/02_summarize_zone_tracks.py --camera camB
python scripts/02_apply_direction_mapping.py --camera camA
python scripts/02_apply_direction_mapping.py --camera camB
```

### Faz 3: Kameralar arasi aday eslestirme

```bash
python scripts/03_extract_track_features.py --camera camA
python scripts/03_extract_track_features.py --camera camB
python scripts/03_build_match_candidates.py
python scripts/03_assign_matches.py
```


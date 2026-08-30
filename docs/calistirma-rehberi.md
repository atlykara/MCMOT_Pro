# Çalıştırma Rehberi

Hattın tamamını sıfırdan çalıştırmak için gereken sıra ve dikkat edilmesi gereken
noktalar. Kurulum adımları için [kurulum.md](kurulum.md) dosyasına bakın.

## Ortam

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision    # CUDA sürümü için pytorch.org talimatları
python -m pip install -r requirements.txt
```

Doğrulama:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; YOLO('models/yolo11s.pt'); print('YOLO hazir')"
```

## İki pratik ayrıntı

1. `src/` otomatik olarak import yoluna girmez. Scriptlerin bir kısmı `sys.path`
   düzenlemesini kendi içinde yapar, testler yapmaz. Bu yüzden komutları
   `PYTHONPATH=src` ile çalıştırmak en güvenli yoldur.
2. `02_preview_zones.py` için `--image` zorunludur. Öncesinde
   `02_extract_reference_frames.py` çalıştırılıp referans kare üretilmelidir.

## Tam zincir

```bash
export PYTHONPATH=src

# Testler
python -m pytest -q

# Faz 1 — tespit + takip (GPU'da kamera başına birkaç dakika)
python scripts/01_run_detect_track.py --camera camA --model models/yolo11s.pt --conf 0.25 --imgsz 960 --no-video
python scripts/01_run_detect_track.py --camera camB --model models/yolo11s.pt --conf 0.25 --imgsz 960 --no-video
python scripts/01_track_quality.py --camera camA
python scripts/01_track_quality.py --camera camB

# Faz 2 — ROI / zone / yön
for c in camA camB; do
  python scripts/02_extract_reference_frames.py --camera $c
  python scripts/02_preview_zones.py --camera $c --image outputs/reference_frames/$c/${c}_t30s.jpg
  python scripts/02_assign_zones.py --camera $c
  python scripts/02_summarize_zone_tracks.py --camera $c
  python scripts/02_apply_direction_mapping.py --camera $c
done

# Faz 3 — eşleştirme
python scripts/03_extract_track_features.py --camera camA
python scripts/03_extract_track_features.py --camera camB
python scripts/03_build_match_candidates.py
python scripts/03_assign_matches.py
python scripts/03_eval_matches.py

# İzleme konsolu (video yazar; canlı pencere için --show ekleyin)
python scripts/03_run_fifo_console.py --duration 30
```

## Referans ölçüm (RTX 4080 Laptop, CUDA 12.8)

| Adım | Ölçüm |
|---|---|
| Faz 1 camA | 5427 kare, 97213 kayıt, 23.4 FPS |
| Faz 1 camB | 4949 kare, 71580 kayıt, 24.6 FPS |
| Faz 2 | camA 518, camB 451 yönlü track |
| Faz 3 | 593 aday çift → 388 bire-bir eşleşme |

## Önemli: `ground_truth.csv` track kimlik uzayına bağlıdır

`outputs/matching/ground_truth.csv` içindeki çiftler (`camA#1 -> camB#14` gibi)
belirli bir Faz 1 koşusunun ByteTrack kimliklerine aittir. Faz 1 yeniden
çalıştırıldığında ultralytics sürümü, model veya parametre farkı tespit sırasını
değiştirir; ByteTrack farklı kimlikler atar ve `03_eval_matches.py` sıfıra düşer.

Ölçülen örnek: aynı video ve aynı model, farklı ultralytics sürümü —

| | Orijinal koşu | Yeni koşu |
|---|---|---|
| camA benzersiz track | 1692 | 1883 |
| `camA#1` | `car` | `truck` |

Bu nedenle doğrulama seti ile ölçüm yapılacaksa **Faz 1 çıktısı korunmalı, yalnızca
Faz 2–3 yeniden çalıştırılmalıdır.** Faz 1 yenilenirse doğrulama seti yeniden
etiketlenmelidir.

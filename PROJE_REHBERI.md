# MCMOT_Pro — Proje Başlangıç Rehberi ve AI Prompt Kütüphanesi

> **Sürüm:** v1.0 · 22 Temmuz 2026
> **Kaynak:** `SOOS_Coklu_Kamera_Otopark_Arac_Takip_Fazli_Gorev_Plani_v1.html` (PV-02 Görev Brifi)
> **Hedef Senaryo:** **Senaryo A — Akıllı Otopark**
> **Ekip:** Efe & Atalay · 10 iş günü MVP

Bu rehber yaşayan bir dokümandır; ilerledikçe birlikte ekleme/çıkarma yapacağız.
Bölüm 9'daki promptlar Claude Code / Codex gibi AI kodlama araçlarına **kopyala-yapıştır** verilmek üzere yazılmıştır.

---

## İçindekiler

1. [Projenin Özü — 60 Saniyede](#1-projenin-özü)
2. [Mevcut Durum Envanteri (E:\MCMOT_Pro)](#2-mevcut-durum-envanteri)
3. [Teknoloji Seçimi ve Gerekçeleri](#3-teknoloji-seçimi)
4. [Ortam Kurulumu (Windows, adım adım)](#4-ortam-kurulumu)
5. [Önerilen Repo / Dosya Yapısı](#5-repo-yapısı)
6. [Veri Kontratları — Projenin Omurgası](#6-veri-kontratları)
7. [Faz Haritası ve Bitti Kriterleri](#7-faz-haritası)
8. [AI Araçlarıyla Çalışma Prensipleri + CLAUDE.md](#8-ai-ile-çalışma)
9. [PROMPT KÜTÜPHANESİ — Faz Faz Hazır Promptlar](#9-prompt-kütüphanesi)
10. [Git Düzeni ve Günlük Rapor Şablonu](#10-git-ve-rapor)
11. [Riskler ve Kaçınılacaklar](#11-riskler)
12. [İlk Gün Yapılacaklar Listesi](#12-ilk-gün)

---

## 1. Projenin Özü

**Tek cümle:** Bir otoparktaki 2-3 kameranın görüntüsünde araçları YOLO ile tespit edip, her kamerada takip ederek (**track_id**), farklı kameralarda görünen aynı aracı **tek ortak kimlik (`match_id`)** altında birleştirmek ve bundan otopark analitiği (doluluk, süre, uyarı) üretmek.

### Üç kimlik kavramı — asla karıştırma

| Kimlik | Kapsam | Kim üretir | Bu projede |
|---|---|---|---|
| `track_id` | Tek kamera içi, geçici | Tracker (ByteTrack vb.) | ✅ Faz 1'de üretilir |
| `match_id` | **Kameralar arası — projenin ana çıktısı** | Bizim kural/skor motorumuz (Re-ID'siz) | ✅ Faz 3'te üretilir |
| `reid_id` | Öğrenilmiş görünüm modeli kimliği | Gelecekteki Re-ID modülü | ❌ Üretilmez; sadece boş adaptör alanı bırakılır (Faz 5) |

### Değişmez kurallar (brife göre sabit)

- **Re-ID YASAK (Faz 3'te):** Öğrenilmiş görünüm embedding'i, derin Re-ID ağı, plaka tanıma **kullanılmaz**. Serbest olan: tek kareden hesaplanan ucuz ipuçları (baskın renk / basit renk histogramı, araç tipi/boyut sınıfı, kutu en-boy oranı).
- **Ana sinyal geometri + zamandır**; görünüm ipucu sadece opsiyonel yardımcı skordur.
- **"Kesin kimlik eşleştirmesi" ifadesi kullanılmaz** — bu olasılıksal bir tahmindir; yanlış eşleşme oranı ölçülür ve raporda açıkça yazılır.
- Girdi/çıktı kontratları (alan adları, dosya formatları) sabittir — bkz. Bölüm 6. Yöntem seçimi serbesttir ama kontrat değişmez.
- Faz 2 verisi güvenilir olmadan Faz 3'e geçilmez; Faz 0'da senaryo kararı Sezer'e bildirilmeden Faz 1'e geçilmez.

### İki eşleştirme durumu (Faz 3'ün kalbi)

1. **Durum 1 — Örtüşen kameralar (birincil, hedeflenen):** Araç aynı anda iki kamerada görünür → eşleştirme **uzamsal**: aynı timestamp'te aynı fiziksel zemin konumu (homografi / ortak zemin noktaları).
2. **Durum 2 — Ardışık / kör bölge (ikincil):** Araç A'dan çıkar, bir süre sonra B'de belirir → eşleştirme **zamansal**: çıkış bölgesi/yönü + giriş bölgesi/yönü + makul geçiş süresi penceresi.

---

## 2. Mevcut Durum Envanteri

`E:\MCMOT_Pro` klasöründe şu an:

| Varlık | İçerik | Senaryo A için değerlendirme |
|---|---|---|
| `SOOS_..._Gorev_Plani_v1.html` | Görev brifi | Tüm projenin anayasası — bu rehber ona göre yazıldı |
| `DATASET/AICity21_Track2_ReID` | ~840 jpg araç kırpıntısı (ReID veri seti) | ⚠️ **Senaryo A için uygun değil** — bunlar kamera videosu değil, araç kırpıntı görüntüleri; ayrıca Re-ID bu projede Faz 5'e ertelendi |
| `DATASET/AICity21-Track4-Anomaly-Detection` | ~250 mp4 trafik videosu | ⚠️ Tek kameralı otoyol videoları — otopark değil; olsa olsa Senaryo B yedeği için ham malzeme |
| `python_server_soket-.../` | Soket sunucu/istemci prototipi (server.py + cl1-3.py) | Görüntü işleme değil; Faz 3 handoff mimarisi için **ilham**, kullanmak zorunlu değil |

> **Sonuç:** Faz 0'ın ana işi hâlâ önümüzde: **çok kameralı, görüş alanı örtüşen otopark videosu bulmak.** Aday kaynaklar: NVIDIA AI City Challenge **Track 3 (MTMC — CityFlow)**, **PKLot / CNRPark+EXT** (doluluk için, ama tek kare/tek kamera ağırlıklı), **Roboflow Universe** otopark setleri, ve gerekirse iki açıdan kısa kanıtlama çekimi (final sayılmaz).

---

## 3. Teknoloji Seçimi

Brif yöntem seçimini bize bırakıyor; gerekçeli öneri şu (Faz 0/1 teslim notuna da yazılacak):

| Katman | Seçim | Gerekçe |
|---|---|---|
| Dil / sürüm | **Python 3.10 veya 3.11** | Ultralytics + PyTorch ekosistemiyle en sorunsuz aralık |
| Tespit | **Ultralytics YOLO** (`yolov8s` ile başla; gerekirse `yolo11s`) | COCO'da `car/truck/bus` hazır; eğitim gerekmeden başlanır; tek satırda tracker entegrasyonu |
| Takip (tek kamera) | **ByteTrack** (Ultralytics içinde `bytetrack.yaml`) — alternatif: BoT-SORT | Hızlı, ID switch'i az, ek model istemez. BoT-SORT'un ReID kolu **kapalı tutulur** (kural gereği) |
| Görüntü işleme | **OpenCV** (`opencv-python`) | Video okuma/yazma, homografi (`cv2.findHomography`), çizim |
| Geometri | **Shapely** | Park yeri poligonları, point-in-polygon, taşma (çoklu yer kaplama) kontrolü |
| Veri | **pandas + JSONL/CSV** | Kontrat dosyaları; MVP'de veritabanı gerekmez |
| Config | **PyYAML** | `zone_mapping.yaml`, kamera ve park yeri tanımları |
| Panel (Faz 4) | **Streamlit** (veya tek dosya HTML) | Yarım günde doluluk paneli |

**Bilinçli olarak kullanmadıklarımız:** derin Re-ID ağları (torchreid vb.), plaka tanıma (OCR), DeepSORT'un appearance kolu — hepsi brifin "Re-ID sayılır, Faz 5 sonrasına" sınırının öbür tarafında.

---

## 4. Ortam Kurulumu

Windows (E: sürücüsünde çalışıyoruz) için adım adım. Her iki donanım yolu da var; önce GPU kontrolü yap.

### 4.1 GPU var mı?

```bat
nvidia-smi
```

- Tablo geliyorsa → NVIDIA GPU var, **CUDA yolu**nu izle (sağ üstteki "CUDA Version" değerini not et).
- "not recognized" hatası → **CPU yolu** (yolov8n/s + frame atlama ile MVP yine rahat çalışır).

### 4.2 Sanal ortam

```bat
cd /d E:\MCMOT_Pro
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

> Conda kullanıyorsanız: `conda create -n mcmot python=3.11 -y && conda activate mcmot` — ikisi de olur, **tek kural: ikiniz de aynı yolu kullanın** ve `requirements.txt` tek gerçek kaynak olsun.

### 4.3 PyTorch

```bat
:: GPU (CUDA 12.x) — güncel komutu https://pytorch.org 'Get Started' sayfasından doğrulayın:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: CPU-only:
pip install torch torchvision
```

### 4.4 Proje bağımlılıkları

```bat
pip install ultralytics opencv-python shapely pandas pyyaml streamlit lapx
```

`requirements.txt` (repoya ilk commit'lerden biri):

```text
ultralytics>=8.2
opencv-python>=4.9
shapely>=2.0
pandas>=2.0
pyyaml>=6.0
streamlit>=1.35
lapx>=0.5        # ByteTrack atama çözücüsü
# torch/torchvision: pytorch.org'daki platforma uygun komutla ayrı kurulur (README'ye not düş)
```

### 4.5 Kurulum doğrulama (smoke test)

```bat
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
yolo predict model=yolov8s.pt source="https://ultralytics.com/images/bus.jpg"
```

İkinci komut `runs/detect/predict/` altına kutulu bir görsel yazıyorsa ortam hazır demektir.

---

## 5. Repo Yapısı

Önerilen yapı — her klasörün faz karşılığı var, kontrat dosyaları `outputs/` altında standart isimlerle birikir:

```text
E:\MCMOT_Pro\
├── .venv/                        # sanal ortam (git'e girmez)
├── CLAUDE.md                     # AI araçları için proje anayasası (bkz. Bölüm 8) — Codex için AGENTS.md kopyası
├── README.md                     # kurulum + çalıştırma + parametreler (Faz 5 teslimi)
├── requirements.txt
├── .gitignore                    # .venv/, data/raw/, outputs/videos/, *.pt, runs/
│
├── docs/
│   ├── SOOS_..._Gorev_Plani_v1.html      # görev brifi (buraya taşı)
│   ├── PROJE_REHBERI.md                  # bu dosya
│   ├── veri_degerlendirme_notu.md        # Faz 0 teslimi
│   ├── validation_note.md                # Faz 3 Gün 8 teslimi
│   └── karar_raporu.md                   # Faz 5 teslimi
│
├── data/
│   ├── raw/                      # dokunulmaz ham videolar/veri setleri (git'e girmez)
│   │   ├── camA/  camB/  (camC/)
│   └── samples/                  # 30-60 sn'lik kısa test kesitleri (hızlı iterasyon için)
│
├── configs/
│   ├── cameras.yaml              # kamera kimlikleri, video yolları, fps, saat ofsetleri
│   ├── zones_camA.yaml           # kamera A park yeri / ROI poligonları
│   ├── zones_camB.yaml
│   ├── zone_mapping.yaml         # Faz 3: kameralar arası bölge/homografi eşlemesi
│   └── matching.yaml             # eşleştirme eşikleri, zaman penceresi, skor ağırlıkları
│
├── src/mcmot/
│   ├── __init__.py
│   ├── io_utils.py               # JSONL/CSV okuma-yazma, kontrat şema doğrulama
│   ├── detect_track.py           # Faz 1: YOLO + ByteTrack → tracks.jsonl
│   ├── zones.py                  # Faz 2: foot point, point-in-polygon, park/ROI ataması
│   ├── geometry.py               # Faz 3: homografi, zemin düzlemi projeksiyonu
│   ├── matching.py               # Faz 3: aday üretme + skor + match_id atama
│   ├── analytics.py              # Faz 4: doluluk, süre, uyarılar, zaman serisi
│   ├── reid_adapter.py           # Faz 5: boş Re-ID adaptörü (mock ile test edilir)
│   └── viz.py                    # kutu/poligon/match çizimleri, yan yana görüntü
│
├── scripts/                      # her faza bir CLI giriş noktası
│   ├── 00_probe_video.py         # Faz 0: aday video teknik analizi
│   ├── 01_run_detect_track.py    # python scripts/01_... --camera camA
│   ├── 02_define_zones.py        # tıkla-çiz poligon aracı
│   ├── 02_assign_zones.py
│   ├── 03_check_sync.py
│   ├── 03_build_mapping.py
│   ├── 03_run_matching.py
│   ├── 03_validate_matches.py
│   ├── 04_run_analytics.py
│   └── 05_run_pipeline.py        # uçtan uca tek komut
│
├── outputs/                      # üretilen her şey (kontrat dosyaları)
│   ├── tracks/    tracks_camA.jsonl, tracks_camB.jsonl
│   ├── zones/     zone_events_camA.csv, occupancy_snapshot_camA.csv
│   ├── matching/  match_candidates.csv, matches.csv, global_tracks.csv
│   ├── analytics/ occupancy_timeseries.csv, vehicle_durations.csv, alerts.csv
│   └── videos/    annotated_camA.mp4, matching_sidebyside.mp4 (git'e girmez)
│
└── tests/
    ├── test_contracts.py         # çıktı dosyaları şemaya uyuyor mu
    ├── test_zones.py             # point-in-polygon birim testleri
    ├── test_matching.py          # sentetik veriyle eşleştirme testi
    └── test_reid_adapter.py      # mock Re-ID testi
```

**Neden böyle:** Efe (Kamera-A) ve Atalay (Kamera-B) Faz 1-2'de **aynı script'leri sadece `--camera` parametresiyle** paralel koşturur — kod ikilenmez, kontrat garanti aynı kalır. Faz 3 (`matching.py`) ikisinin çıktısını okuyan ortak iş paketidir; brifteki görev dağılımı önerisiyle birebir örtüşür.

---

## 6. Veri Kontratları

**Bu bölüm projenin omurgası.** Brif "girdi/çıktı kontratları sabittir" diyor; AI'lara verilen her prompt bu şemaları referans alacak. Alan adlarını değiştirmek = ekip arkadaşının kodunu kırmak.

### 6.1 `tracks_<cam>.jsonl` — Faz 1 çıktısı (kare başına bir satır/araç)

```json
{"timestamp": 12.4833, "frame": 299, "camera_id": "camA", "track_id": 17,
 "class": "car", "conf": 0.87,
 "bbox_xyxy": [412.1, 220.5, 588.9, 340.2],
 "foot_point": [500.5, 340.2],
 "hints": {"dominant_color": [128, 130, 135], "size_class": "medium", "aspect_ratio": 1.62}}
```

- `timestamp`: videonun başından saniye (float). Gerçek saat ofseti `cameras.yaml`'da tutulur, ham veriye işlenmez.
- `foot_point`: kutunun **alt-orta noktası** (zemin teması varsayımı) — Faz 2/3'ün tek konum referansı.
- `hints`: sadece serbest ipuçları (kaba renk, boyut sınıfı, en-boy oranı). **Embedding alanı yok ve açılmayacak.**

### 6.2 `zones_<cam>.yaml` — Faz 2 park yeri/ROI tanımı

```yaml
camera_id: camA
image_size: [1920, 1080]
parks:
  - park_id: A-01
    polygon: [[512, 400], [700, 400], [720, 520], [500, 520]]
  - park_id: A-02
    polygon: [[700, 400], [880, 405], [905, 520], [720, 520]]
rois:                      # senaryo B / geçiş bölgeleri için opsiyonel
  - zone_id: A-EXIT-EAST
    polygon: [[1700, 300], [1920, 300], [1920, 700], [1700, 700]]
```

### 6.3 `zone_events_<cam>.csv` — Faz 2 çıktısı (Faz 3'e geçiş şartındaki alanlar)

```csv
timestamp,camera_id,track_id,park_id,zone_id,foot_x,foot_y,event
101.20,camA,17,A-03,,500.5,340.2,enter
245.87,camA,17,A-03,,505.1,338.9,exit
```

> Brifteki Faz 3 geçiş şartı: her araç için `timestamp`, `camera_id`, `track_id`, `zone_id`/`park_id`, zemin noktası **güvenilir** kaydediliyor olmalı. Bu CSV güvenilir değilse Faz 3'e geçilmez.

### 6.4 `matches.csv` — Faz 3 ana çıktısı

```csv
match_id,camera_id,track_id,t_start,t_end,method,score,suspect
M0001,camA,17,101.20,245.87,overlap_spatial,0.91,0
M0001,camB,42,99.95,250.10,overlap_spatial,0.91,0
M0002,camA,23,300.40,340.00,handoff_temporal,0.64,1
```

- `method`: `overlap_spatial` (Durum 1) | `handoff_temporal` (Durum 2)
- `suspect=1`: eşik altı/şüpheli eşleşme — doğrulama örneklemine girer, analitikte ayrıca işaretlenir.
- Ara ürün `match_candidates.csv` tüm aday çiftleri skor bileşenleriyle saklar (denetlenebilirlik).

### 6.5 `global_tracks.csv` — Faz 4'ün girdisi

```csv
match_id,reid_id,camera_id,track_id,park_id,t_start,t_end
M0001,,camA,17,A-03,101.20,245.87
```

`reid_id` **her zaman var ama boş** — Faz 5 adaptörü ileride sadece bu kolonu doldurur; boşken hiçbir kod kırılmaz (fallback = `match_id`).

---

## 7. Faz Haritası

| Faz | Gün | Ağırlık | Çıktı | "Bitti" kriteri |
|---|---|---|---|---|
| **F0** Veri/karar | 1-2 | %10 | `veri_degerlendirme_notu.md` + seçilen veri + yöntem taslağı | ≥2 kameralı, aynı alanı gören kullanılabilir kaynak seçildi, gerekçeli; **Sezer'e bildirildi** |
| **F1** Tespit/takip | 3-4 | %15 | Annotated video + `tracks_<cam>.jsonl` | 2 kamerada araçların çoğu takipte; ID switch'ler loglu ve örnekli |
| **F2** Konum/bölge | 5 | %15 | `zones_*.yaml` + `zone_events_*.csv` + doluluk anlık görüntüsü | Her araç bir park/bölgeyle ilişkili; kameralar arası iş YOK |
| **F3** Birleştirme ⭐ | 6-8 | **%35** | `zone_mapping.yaml`, `matches.csv`, `validation_note.md` | (A) aynı araç = aynı `match_id`, zaman sırası mantıklı; (B) yanlış eşleşme oranı ölçülü ve raporda; (C) tekrarlanabilir |
| **F4** Analitik | 9 | %15 | doluluk haritası, süre, taşma uyarısı, zaman serisi + panel | Metrik kaynağı F2/F3 çıktıları; "yaklaşık" sınırı raporda |
| **F5** Test/teslim | 10 | %10 | Re-ID adaptörü, README, demo video, karar raporu | 2 kamerayla uçtan uca çalışır; Re-ID yokken çökmez |

Gün planı (F3): **Gün 6** bölge & geometri (`zone_mapping.yaml`) → **Gün 7** eşleştirme prototipi & skor (`matches.csv`) → **Gün 8** doğrulama & saat senkronu (`validation_note.md`).

---

## 8. AI ile Çalışma

### 8.1 Prensipler (promptların kalitesini bunlar belirliyor)

1. **Bir prompt = bir teslim edilebilir çıktı.** "Tüm sistemi yaz" değil; "`01_run_detect_track.py`'yi yaz, çıktısı şu şemada JSONL olsun" gibi.
2. **Kontratı prompta göm.** AI alan adı uydurmasın; şemayı (Bölüm 6) promptun içine yapıştır.
3. **Kabul kriterini prompta yaz.** "Bitti sayılır: şu komut şu dosyayı üretir, şu alanlar dolu olur."
4. **Yasakları açıkça söyle.** Her Faz 3 promptunda "Re-ID/embedding/plaka OCR kullanma" satırı var — AI'lar bu tür projelerde refleks olarak DeepSORT+ReID önerir.
5. **Küçük başlat, videoyla değil kesitle iterasyon yap.** `data/samples/` içindeki 30-60 sn kesitlerle geliştir, tam videoyu en son koştur.
6. **AI çıktısını körü körüne alma:** önce sample'da çalıştır, çıktı dosyasını `tests/test_contracts.py` ile doğrula, annotated videoyu gözle izle.

### 8.2 `CLAUDE.md` — repo köküne koy (Codex için aynısını `AGENTS.md` olarak kopyala)

Claude Code bu dosyayı her oturumda otomatik okur; her prompta bağlam tekrarlamaktan kurtarır:

```markdown
# MCMOT_Pro — Çoklu Kamera Otopark Araç Takibi (SOOS PV-02)

## Proje
2-3 otopark kamerasında YOLO ile araç tespiti + ByteTrack takibi; farklı kameralardaki
aynı aracı Re-ID KULLANMADAN (geometri + zaman + ucuz görünüm ipucu) tek `match_id`
altında birleştirme; üstüne otopark analitiği (doluluk, süre, taşma uyarısı).

## Kesin kurallar
- Re-ID YASAK: öğrenilmiş görünüm embedding'i, derin Re-ID ağı, plaka tanıma/OCR kullanma,
  önerme. Serbest: tek kareden kaba renk, boyut sınıfı, en-boy oranı.
- Veri kontratları sabittir: alan adlarını docs/PROJE_REHBERI.md Bölüm 6'daki şemalardan al,
  asla yeniden adlandırma. Ana dosyalar: tracks_<cam>.jsonl, zone_events_<cam>.csv,
  matches.csv, global_tracks.csv (reid_id kolonu her zaman var, şimdilik boş).
- Konum referansı: bbox alt-orta noktası = foot_point. Tüm bölge/eşleştirme mantığı buna dayanır.
- "Kesin eşleştirme" iddiası yok: matching çıktılarında score ve suspect alanları zorunlu.

## Teknik
- Python 3.11, venv: .venv | Ultralytics YOLO (yolov8s) + bytetrack.yaml | OpenCV, Shapely,
  pandas, PyYAML, Streamlit.
- Windows'ta çalışıyoruz; yolları pathlib ile yaz, os.path string birleştirme yapma.
- Her script scripts/ altında CLI (argparse), çekirdek mantık src/mcmot/ altında modül.
- Videolar büyük: geliştirmede data/samples/ kesitlerini kullan, --max-frames parametresi ekle.

## Test
- pytest tests/ | Yeni çıktı formatı eklersen tests/test_contracts.py'ye şema testi ekle.
```

---

## 9. PROMPT KÜTÜPHANESİ

Kullanım: sırayla ilerle; her promptu çalıştırmadan önce `<...>` içindeki yerleri kendi değerlerinle doldur. Promptlar Türkçe; Claude Code ve Codex Türkçe komutlarla sorunsuz çalışır. Her prompt tek oturumda bitecek boyutta tasarlandı.

### FAZ 0 — Veri ve karar

#### P0.1 — Proje iskeletini kur

```text
E:\MCMOT_Pro içinde çok kameralı otopark araç takip projesi (MCMOT) için repo iskeleti kur.

Yapılacaklar:
1. Şu klasör ağacını oluştur (boş klasörlere .gitkeep koy):
   docs/, data/raw/camA, data/raw/camB, data/samples/, configs/, src/mcmot/, scripts/,
   outputs/tracks, outputs/zones, outputs/matching, outputs/analytics, outputs/videos, tests/
2. src/mcmot/__init__.py ve şu boş modül dosyalarını oluştur (her birine bir satır docstring):
   io_utils.py, detect_track.py, zones.py, geometry.py, matching.py, analytics.py,
   reid_adapter.py, viz.py
3. requirements.txt: ultralytics>=8.2, opencv-python>=4.9, shapely>=2.0, pandas>=2.0,
   pyyaml>=6.0, streamlit>=1.35, lapx>=0.5 (torch'un pytorch.org'dan ayrı kurulduğunu
   yorum satırıyla belirt).
4. .gitignore: .venv/, __pycache__/, data/raw/, outputs/videos/, runs/, *.pt, .DS_Store
5. configs/cameras.yaml şablonu: her kamera için camera_id, video_path, fps,
   clock_offset_seconds (varsayılan 0.0), notes alanları; camA ve camB örnek girdileriyle.
6. git init + anlamlı ilk commit ("chore: proje iskeleti").

Kısıtlar:
- Windows ortamı; yol işlemleri için pathlib kullan.
- Henüz hiçbir işlev kodu yazma, sadece iskelet.

Bitti sayılır: klasör ağacı yukarıdakiyle birebir aynı, git log'da 1 commit var.
```

#### P0.2 — Aday video/veri seti teknik analiz aracı

```text
scripts/00_probe_video.py adlı bir CLI aracı yaz. Amaç: Faz 0'da aday otopark videolarını
teknik olarak değerlendirip veri değerlendirme notuna girecek ham bilgiyi üretmek.

Girdi: --input <dosya veya klasör> (mp4/avi/mkv), opsiyonel --grid-preview
Her video için çıkar:
- çözünürlük, fps, süre, toplam kare, codec
- 5 eşit aralıklı karede: yolov8s ile tespit edilen araç sayısı (class: car, truck, bus;
  conf>=0.35) — böylece "araç görünüyor mu, model bu açıda çalışıyor mu" hızlıca anlaşılır
- her videodan 3 örnek kareyi kutularıyla outputs/probe/<video_adi>/ altına jpg kaydet
Çıktı: outputs/probe/probe_report.csv (video, çözünürlük, fps, süre, ort_araç_sayısı,
min_conf, not alanı boş) + konsola özet tablo.

Kısıtlar: pathlib kullan; model dosyası yoksa ultralytics otomatik indirir, buna izin ver;
GPU yoksa CPU'da çalışmalı (device parametresini otomatik seç).

Bitti sayılır: tek bir mp4 ile çalıştırıldığında CSV ve örnek jpg'ler oluşuyor.
```

> **Not:** Veri değerlendirme notunun kendisi (hangi aday neden uygun/değil) sizin yazacağınız
> bir karar metnidir — AI'a yazdırmayın, brif bunu stajyer teslimi olarak istiyor. AI'dan
> yalnızca yukarıdaki gibi ham teknik veri toplamasını isteyin.

### FAZ 1 — Tek kamera tespit + takip

#### P1.1 — YOLO + ByteTrack hattı

```text
Faz 1: tek kamera araç tespit + takip hattını yaz.

Dosyalar:
- src/mcmot/detect_track.py: çekirdek mantık (fonksiyon/sınıf olarak, script'ten bağımsız test edilebilir)
- src/mcmot/io_utils.py: JSONL yazıcı + şema sabitleri
- scripts/01_run_detect_track.py: CLI

CLI: python scripts/01_run_detect_track.py --camera camA [--model yolov8s.pt]
[--max-frames N] [--no-video]
- configs/cameras.yaml'dan video_path ve fps okunur.
- Ultralytics model.track(..., tracker="bytetrack.yaml", persist=True, classes=[2,5,7])
  kullan (COCO: car=2, bus=5, truck=7). conf>=0.35.

Her karede her araç için outputs/tracks/tracks_<camera>.jsonl dosyasına TAM OLARAK şu
şemayla bir satır yaz (alan adlarını değiştirme):
{"timestamp": <video başından saniye, float>, "frame": <int>, "camera_id": "<str>",
 "track_id": <int>, "class": "<car|bus|truck>", "conf": <float>,
 "bbox_xyxy": [x1,y1,x2,y2], "foot_point": [(x1+x2)/2, y2],
 "hints": {"dominant_color": [B,G,R], "size_class": "<small|medium|large>",
           "aspect_ratio": <w/h, float>}}
- dominant_color: bbox alt yarısının medyan BGR değeri (ucuz olsun, k-means kullanma)
- size_class: bbox alanının kare alanına oranına göre eşikle (küçük <%1, orta %1-4, büyük >%4)

Ek çıktı (--no-video verilmediyse): outputs/videos/annotated_<camera>.mp4 — kutu,
track_id ve conf yazılı.

YASAK: Re-ID, embedding, appearance tabanlı tracker kolu (BoT-SORT with_reid dahil) kullanma.

Bitti sayılır: data/samples/ içindeki kesitle koşunca JSONL ve annotated video oluşuyor;
JSONL'in ilk satırı json.loads ile parse edilip tüm alanları içeriyor.
```

#### P1.2 — Takip kalite raporu (ID switch günlüğü)

```text
scripts/01_track_quality.py yaz. Girdi: --tracks outputs/tracks/tracks_camA.jsonl

Üreteceği rapor (konsol + outputs/tracks/quality_<camera>.md):
- toplam benzersiz track_id sayısı, track başına ortalama/medyan yaşam süresi (sn)
- "kısa ömürlü track" listesi (<1.5 sn yaşayanlar) — muhtemel bölünme/hayalet
- "şüpheli ID değişimi" adayları: bir track kaybolduktan sonra <=1.0 sn içinde, son
  foot_point'ine <75 piksel mesafede yeni bir track doğuyorsa (eski_id -> yeni_id, t, mesafe)
  listesine yaz — brif ID switch'lerin loglanıp örneklenmesini istiyor
- kare başına ortalama araç sayısı grafiği (matplotlib png, outputs/tracks/ altına)

Bitti sayılır: rapor md dosyası oluşuyor ve şüpheli ID değişimi tablosu (boş da olsa) içeriyor.
```

### FAZ 2 — Konum ve bölge

#### P2.1 — Park yeri poligon çizim aracı

```text
scripts/02_define_zones.py: OpenCV ile interaktif park yeri/ROI tanımlama aracı yaz.

Akış:
- --camera camA --frame 100 → cameras.yaml'daki videodan o kareyi aç, pencerede göster
- Sol tık: poligon köşesi ekle; Enter: poligonu kapat ve park_id sor (konsoldan,
  varsayılan otomatik artan A-01, A-02...); 'r': ROI modu (zone_id sorar, A-EXIT-EAST gibi);
  'u': son noktayı geri al; 'd': son poligonu sil; 's': kaydet ve çık
- Kaydedilen poligonlar yarı saydam renkle ve id etiketiyle kare üstünde görünsün
- Çıktı: configs/zones_<camera>.yaml — şema:
  camera_id, image_size: [w, h], parks: [{park_id, polygon: [[x,y],...]}],
  rois: [{zone_id, polygon: [[x,y],...]}]
- Dosya varsa üstüne yazmadan önce yükleyip düzenlemeye devam etsin.

Bitti sayılır: 2 park yeri + 1 ROI çizip 's' ile kaydedince geçerli YAML oluşuyor ve
tekrar açınca poligonlar geri geliyor.
```

#### P2.2 — Araç ↔ park yeri ilişkilendirme

```text
Faz 2 bölge atama hattını yaz.

Dosyalar: src/mcmot/zones.py (çekirdek) + scripts/02_assign_zones.py (CLI)
Girdi: outputs/tracks/tracks_<cam>.jsonl + configs/zones_<cam>.yaml
CLI: python scripts/02_assign_zones.py --camera camA

Mantık:
- Her kayıt için foot_point'in hangi park poligonunda (shapely, point-in-polygon)
  olduğunu bul; hiçbirinde değilse park_id boş, ROI'lerde ise zone_id yaz.
- Titremeye karşı histerezis: bir araç bir parka "girdi" sayılması için foot_point'in
  o poligonda kesintisiz >=2.0 sn kalması; "çıktı" sayılması için >=2.0 sn dışarıda
  kalması gerekir (fps'i cameras.yaml'dan al).
- Çıktı 1: outputs/zones/zone_events_<cam>.csv — kolonlar TAM OLARAK:
  timestamp,camera_id,track_id,park_id,zone_id,foot_x,foot_y,event  (event: enter|exit)
- Çıktı 2: outputs/zones/occupancy_snapshot_<cam>.csv — her park_id için son durum:
  park_id,occupied(0|1),track_id,since_timestamp
- Çıktı 3 (opsiyonel --render): park poligonları dolu=kırmızı/boş=yeşil boyalı
  annotated video.

Bitti sayılır: sample veride enter/exit çiftleri tutarlı (her exit'ten önce enter var),
snapshot dosyası park sayısı kadar satır içeriyor.
```

### FAZ 3 — Kameralar arası birleştirme (ANA İŞ PAKETİ, %35)

#### P3.1 — Kamera saat senkron kontrolü (Gün 8 riskine erken önlem)

```text
scripts/03_check_sync.py yaz. Amaç: iki kameranın saat/zaman ofsetini ölçüp
configs/cameras.yaml'daki clock_offset_seconds alanını doğrulamak.

Yaklaşım:
- --camA-tracks ve --camB-tracks JSONL dosyalarını al.
- Örtüşen görüş alanı varsa: iki kamerada da kısa aralıklarla toplam araç sayısı zaman
  serisini çıkar (1 sn'lik kovalar), numpy ile çapraz korelasyon uygula, en iyi hizalama
  ofsetini (saniye) ve korelasyon gücünü raporla.
- Manuel doğrulama için: --peek t=<saniye> verilince iki videodan da o ana denk gelen
  kareleri yan yana tek jpg olarak outputs/matching/sync_peek_<t>.jpg'ye yaz
  (ofset uygulanmış halde).
- Sonucu konsola ve outputs/matching/sync_report.md'ye yaz: önerilen ofset, güven notu,
  "cameras.yaml'ı güncelle" uyarısı.

Bitti sayılır: sample verilerle koşunca önerilen ofset ve en az 2 peek görseli üretiliyor.
```

#### P3.2 — Zemin geometrisi: homografi / bölge eşlemesi (Gün 6)

```text
Faz 3 geometri katmanını yaz: örtüşen iki kameranın ortak zemin düzlemi eşlemesi.

Dosyalar: src/mcmot/geometry.py + scripts/03_build_mapping.py

1. Eşleme noktası toplama modu (--annotate):
   - İki kameradan aynı ana denk gelen birer kareyi yan yana göster
     (clock_offset uygulanmış).
   - Kullanıcı sırayla aynı fiziksel zemin noktasına önce sol sonra sağ görüntüde tıklar
     (kolon dibi, çizgi köşesi, park çizgisi kesişimi gibi). En az 4, ideal 8-12 çift.
   - Nokta çiftlerini configs/zone_mapping.yaml'a yaz.
2. Homografi modu (--fit):
   - Nokta çiftlerinden cv2.findHomography (RANSAC) ile camA→camB zemin homografisi H
     hesapla; H'yi, reprojeksiyon hatasını (ortalama/maks piksel) ve inlier sayısını
     zone_mapping.yaml'a ekle.
3. Doğrulama modu (--verify):
   - camA'daki park poligonlarının köşelerini H ile camB'ye projekte edip camB karesi
     üstüne çiz, outputs/matching/mapping_verify.jpg olarak kaydet — gözle kontrol için.

zone_mapping.yaml şeması:
  pair: [camA, camB]
  point_pairs: [{a: [x,y], b: [x,y], label: "kolon-3"}]
  homography: [[...3x3...]]
  reprojection_error_px: {mean: ..., max: ...}
  mode: overlap   # overlap | handoff

YASAK: görünüm/embedding tabanlı otomatik nokta eşleme (SIFT/SuperGlue vb. de kullanma —
noktalar elle tıklanacak, denetlenebilirlik önceliğimiz).

Bitti sayılır: --verify çıktısında camA park poligonları camB görüntüsünde makul yerlere
düşüyor (görsel kontrol) ve ortalama reprojeksiyon hatası raporlanmış.
```

#### P3.3 — Eşleştirme motoru: aday + skor + `match_id` (Gün 7)

```text
Faz 3'ün kalbi: kameralar arası eşleştirme motorunu yaz.

Dosyalar: src/mcmot/matching.py + scripts/03_run_matching.py + configs/matching.yaml

Girdiler: tracks_camA.jsonl, tracks_camB.jsonl, zone_events_*.csv, zone_mapping.yaml,
cameras.yaml (clock_offset).

configs/matching.yaml (tüm eşikler burada, kodda sabit sayı gömme):
  time_bucket_s: 0.5          # uzamsal eşleştirmede eş-an toleransı
  max_ground_dist_px: 80      # homografi sonrası aynı araç sayılacak maks zemin mesafesi
  handoff_window_s: [2, 45]   # ardışık modda min-maks geçiş süresi penceresi
  w_spatial: 0.6  w_temporal: 0.25  w_hint: 0.15   # skor ağırlıkları
  score_threshold: 0.55
  suspect_band: [0.55, 0.70]  # bu aralıktakiler suspect=1 işaretlenir

Akış:
1. DURUM 1 (overlap_spatial, birincil): zaman kovalarında (time_bucket_s) camA
   foot_point'lerini homografiyle camB düzlemine taşı; camB araçlarıyla mesafe matrisi kur;
   max_ground_dist_px altındaki çiftler aday. Track çifti bazında zaman içinde tutarlılık
   skoru üret (kaç kovada eşleştiler / birlikte görünür oldukları kova sayısı).
2. DURUM 2 (handoff_temporal, ikincil): camA exit event'i + camB enter event'i,
   handoff_window_s penceresinde ve zone_mapping'de tanımlı çıkış→giriş bölge çiftinde ise aday.
3. Opsiyonel ipucu skoru: hints.dominant_color benzerliği (BGR öklid, normalize) +
   size_class eşitliği. SADECE skora katkı, tek başına eşleştirme sebebi olamaz.
4. Atama: skor matrisi üzerinde greedy değil scipy.optimize.linear_sum_assignment
   (Macar algoritması) kullan; score_threshold altını eşleştirme.
5. match_id üretimi: M0001'den artan; eşleşmeyen track'ler de tekil match_id alır
   (kameralar arası köprüsü yok diye kaybolmasınlar).

Çıktılar (kolon adları TAM OLARAK böyle):
- outputs/matching/match_candidates.csv:
  camA_track_id,camB_track_id,method,s_spatial,s_temporal,s_hint,score,accepted
- outputs/matching/matches.csv:
  match_id,camera_id,track_id,t_start,t_end,method,score,suspect
- outputs/matching/global_tracks.csv:
  match_id,reid_id,camera_id,track_id,park_id,t_start,t_end   (reid_id BOŞ bırakılır)

YASAK: Re-ID, öğrenilmiş embedding, plaka tanıma. Ana sinyal geometri+zaman;
renk/boyut sadece yardımcı.

Bitti sayılır: sample verilerle koşunca 3 CSV de üretiliyor; matches.csv'de aynı match_id
en fazla kamera sayısı kadar satırda görünüyor; suspect bandındaki eşleşmeler işaretli.
```

#### P3.4 — Doğrulama aracı (Gün 8)

```text
Eşleştirme doğrulama aracı yaz: scripts/03_validate_matches.py

Amaç: brifin Faz 3 bitiş şartı B'si — yanlış eşleşme oranını ÖLÇÜP raporlamak.

Özellikler:
1. --review N: matches.csv'den N eşleşmeyi örnekle (accepted olanlardan; suspect'leri
   önceliklendir). Her biri için iki kameradan eş-an kareleri yan yana, ilgili araçlar
   renkli kutuyla vurgulu tek jpg üret: outputs/matching/review/<match_id>.jpg
   Konsolda sor: [y] doğru / [n] yanlış / [u] emin değilim → cevapları
   outputs/matching/review_labels.csv'ye yaz (match_id,label,reviewer,ts).
2. --report: review_labels.csv'den yanlış eşleşme oranını (n / (y+n)), emin olunamayan
   oranı ve yöntem bazında (overlap/handoff) kırılımı hesapla;
   docs/validation_note.md dosyasını şu bölümlerle oluştur/güncelle:
   Örneklem büyüklüğü, Yanlış eşleşme oranı, Yöntem kırılımı, Tipik hata örnekleri
   (jpg yolları), Bilinen sınırlılıklar (benzer renk araç yoğunluğu vb. — boş şablon bırak,
   yorum satırlarını biz dolduracağız).
3. Tutarlılık denetimleri (otomatik, rapora ekle): aynı match_id içinde zaman aralıkları
   mantıklı mı (Durum 2'de camB t_start >= camA t_end olmalı), bir track_id birden fazla
   match_id'de mi (hata), aynı anda aynı kamerada aynı match_id'li iki track var mı (hata).

Bitti sayılır: 10 örneklik bir review turu yapılıp validation_note.md sayısal oran içeriyor.
```

### FAZ 4 — Analitik

#### P4.1 — Otopark analitiği

```text
Faz 4 otopark analitiğini yaz: src/mcmot/analytics.py + scripts/04_run_analytics.py

Girdi: global_tracks.csv, zone_events_*.csv, occupancy_snapshot_*.csv, zones_*.yaml
Kural: TÜM metrikler match_id bazlı hesaplanır (track_id değil) — aynı araç iki kamerada
görünüyorsa BİR araç sayılır, süre kesintisiz birleştirilir. Bu, projenin var oluş sebebi.

Çıktılar (outputs/analytics/):
1. occupancy_timeseries.csv: timestamp,total_spots,occupied,free,occupancy_ratio
   (30 sn çözünürlük)
2. vehicle_durations.csv: match_id,first_seen,last_seen,total_minutes,parks_visited,
   cameras_seen — "kaç dakikadır orada" tablosu
3. alerts.csv: alert_type,match_id,park_ids,t_start,t_end,detail
   - multi_spot: aracın bbox/foot izdüşümü aynı anda >=2 park poligonuyla kesişiyor
     (shapely intersection alanı her iki poligonda da poligon alanının >%15'i ise)
   - long_stay: total_minutes > --long-stay-dk parametresi (varsayılan 120)
   - double_count_risk: suspect=1 eşleşmeden türeyen her metrik satırı işaretle
4. Konsola özet: anlık boş/dolu, bugünkü tekil araç sayısı (tekil match_id), ortalama
   kalış süresi.

Raporlarda "yaklaşık" ibaresi: suspect eşleşmelerden etkilenen satırlar ayrı kolonla
işaretlenmeli (is_approx=1) — brif gereği kesinlik iddiası yok.

Bitti sayılır: 3 CSV üretiliyor; elle kontrol edilen 2 araç için süre değerleri annotated
videoyla tutarlı.
```

#### P4.2 — Basit görsel panel

```text
Streamlit ile tek dosyalık panel yaz: scripts/04_panel.py
(çalıştırma: streamlit run scripts/04_panel.py)

Sekmeler:
1. "Canlı Durum": occupancy_snapshot birleşimi — park haritası (zones yaml'dan poligonları
   matplotlib ile çiz, dolu=kırmızı boş=yeşil, park_id etiketli), boş yer sayısı büyük rakam.
2. "Araçlar": vehicle_durations.csv tablosu, süreye göre sıralı; suspect/is_approx satırları
   sarı vurgulu; match_id seçilince o aracın kamera-zaman çizelgesi.
3. "Uyarılar": alerts.csv; multi_spot uyarılarında ilgili review jpg'si varsa göster.
4. "Doluluk Grafiği": occupancy_timeseries.csv çizgi grafik.

Veri yoksa boş durum mesajı göstersin, exception atmasın. Otomatik yenileme: 10 sn.

Bitti sayılır: mevcut çıktı dosyalarıyla panel açılıyor, 4 sekme de dolu/boş durumu düzgün.
```

### FAZ 5 — Re-ID adaptörü ve teslim

#### P5.1 — Re-ID adaptör noktası + mock test

```text
Faz 5: ileride bağlanacak Re-ID modülü için adaptör katmanı yaz. Re-ID modeli YOK ve
yazılmayacak; sadece temiz bir bağlantı noktası ve fallback davranışı istiyoruz.

Dosyalar: src/mcmot/reid_adapter.py + tests/test_reid_adapter.py

Tasarım:
- ReidProvider protokolü (abstract base): assign_reid_ids(global_tracks: DataFrame) ->
  DataFrame (reid_id kolonu doldurulmuş döner).
- NullReidProvider: reid_id'yi boş bırakır (varsayılan, bugünkü davranış).
- MockReidProvider (sadece test için): match_id'den deterministik sahte reid_id üretir
  ("R" + match_id numarası gibi).
- resolve_provider(config) → configs/matching.yaml'a reid: {enabled: false, provider: null}
  bloğu ekle; enabled=false iken NullReidProvider döner.
- Pipeline entegrasyonu: 04_run_analytics.py, veriyi okuduktan sonra provider'dan geçirir;
  reid_id doluysa loglara "reid aktif" yazar ama TÜM metrikler match_id ile çalışmaya
  devam eder (reid_id şimdilik sadece taşınan bir alan).

Testler:
- Null provider: reid_id kolonu var ve tümü boş; analytics kırılmıyor.
- Mock provider: tüm satırlarda reid_id dolu; aynı match_id → aynı reid_id.
- reid bloğu config'de hiç yokken de sistem NullReidProvider ile çalışıyor (geriye uyum).

Bitti sayılır: pytest tests/test_reid_adapter.py yeşil; enabled=false ile uçtan uca akış
davranış değiştirmiyor.
```

#### P5.2 — Uçtan uca pipeline + teslim paketi

```text
Teslim hazırlığı: uçtan uca tek komut + README.

1. scripts/05_run_pipeline.py: sırayla 01 (her kamera) → 02 → 03 (sync kontrol uyarısı,
   matching) → 04'ü çalıştıran orkestrasyon script'i. Her adımın süresini ve ürettiği
   dosyaları konsola tablo halinde özetler. --skip-video ile video render'ları atlanır.
   Bir adım hata verirse hangi adım, hangi girdi eksik net söyler ve durur.
2. tests/test_contracts.py: outputs/ altındaki her kontrat dosyası için kolon adları ve
   tip kontrolleri (Bölüm 6 şemaları — tracks.jsonl, zone_events, matches, global_tracks,
   analytics çıktıları). Dosya yoksa skip, varsa şema birebir doğrulanır.
3. README.md: Kurulum (venv + torch notu + requirements), veri yerleştirme (data/raw
   düzeni), configs açıklamaları, faz faz çalıştırma komutları, çıktı dosyaları sözlüğü
   (her CSV'nin ne olduğu tek satır), bilinen sınırlılıklar bölümü (validation_note.md'ye
   referans), "Re-ID durumu" bölümü (adaptör var, model yok, fallback match_id).

Bitti sayılır: temiz klonda README'deki komutlar sırayla çalıştırılınca demo çıktılar
üretiliyor; pytest yeşil.
```

### Genel amaçlı yardımcı promptlar

#### PG.1 — Hata ayıklama şablonu

```text
Şu hatayı ayıkla. Bağlam: MCMOT_Pro projesi, <script adı> çalıştırıyorum.

Komut: <tam komut>
Beklenen: <ne olmalıydı>
Olan: <ne oldu>
Hata çıktısı:
<traceback'i TAM yapıştır>

Kurallar: Önce hatanın kök nedenini teşhis et ve bana tek cümleyle söyle; sonra minimal
düzeltmeyi yap. Veri kontratı alanlarını (docs/PROJE_REHBERI.md Bölüm 6) değiştirme.
Düzeltme sonrası aynı komutun çalıştığını doğrula.
```

#### PG.2 — Kod inceleme (birbirinizin fazını AI'a denetletme)

```text
<dosya yolu> dosyasını incele. Bu, MCMOT_Pro projesinde Faz <N>'in parçası.

Şunlara göre denetle ve bulgularını önem sırasına göre listele:
1. Veri kontratına uyum: çıktı alan adları/tipleri docs/PROJE_REHBERI.md Bölüm 6 ile
   birebir aynı mı?
2. Yasak ihlali: Re-ID / embedding / plaka tanıma / öğrenilmiş görünüm modeli var mı,
   ima eden bağımlılık eklenmiş mi?
3. Windows uyumluluğu: pathlib kullanımı, encoding, yol ayracı sorunları.
4. Eşiklerin config'e taşınmışlığı: kodda gömülü sihirli sayı var mı?
5. Kenar durumlar: boş video, hiç araç yok, tek kamera verisi eksik.

Sadece raporla; benden onay almadan kod değiştirme.
```

---

## 10. Git ve Rapor

- **Branch düzeni basit:** `main` + kişi başı çalışma dalı (`efe/faz1-camA`, `atalay/faz1-camB`). Faz sonlarında PR/merge. Faz 3'te tek dalda birlikte (pair) çalışmak daha sağlıklı.
- **Commit mesajı:** `faz1: camB tracker parametreleri; ID switch %X azaldı` gibi — faz etiketi + ölçülebilir sonuç.
- **Büyük dosyalar git'e girmez:** `data/raw/`, `outputs/videos/`, `*.pt` (`.gitignore`'da hazır).

**Günlük rapor şablonu** (brifin "kanıtsız yüzde yazılmaz" kuralına uygun — `docs/gunluk/2026-07-23.md` gibi):

```markdown
# Günlük Rapor — <tarih> — <isim>
- Faz ilerlemesi: Faz <N> %<x> → %<y>   (toplam katkı: ağırlık × %y)
- Kanıt: <commit hash / ekran görüntüsü / çıktı dosyası yolu>
- Bugün çıkan hata/sınırlılık: <...>
- Yarının TEK net teslim hedefi: <...>
```

Toplam ilerleme = Σ(faz ağırlığı × faz tamamlanma yüzdesi). Örn. F0 %100 + F1 %50 + F3 %20 = 10 + 7.5 + 7 = **%24.5**.

---

## 11. Riskler

| Risk | Etki | Önlem (nerede çözülüyor) |
|---|---|---|
| Kamera saatleri senkron değil | Hem uzamsal hem zamansal eşleştirme çöker | P3.1 senkron aracı — Faz 3'ün İLK işi olarak koştur, Gün 8'i bekleme |
| Kameralar arası ışık/renk farkı | Renk ipucu yanıltır | İpucu ağırlığı düşük (w_hint=0.15) ve opsiyonel; ana sinyal geometri/zaman |
| Çok sayıda benzer araç (beyaz sedan) | Yanlış eşleşme artar | suspect bandı + P3.4 örneklem doğrulaması; oran validation_note.md'ye yazılır |
| Görüş alanları hiç örtüşmüyor | Sadece Durum 2 kalır | handoff_window geniş tutulur; belirsizlik raporda açıkça belirtilir |
| AI, Re-ID/DeepSORT önerip sınırı aşar | Brif ihlali | CLAUDE.md'deki yasak + her Faz 3 promptundaki YASAK satırı |
| Tam videoyla yavaş iterasyon | Zaman kaybı | data/samples/ kesitleri + --max-frames; tam koşu günde 1 kez |
| İki kişinin çıktıları uyuşmuyor | Faz 3 tıkanır | Tek script + --camera parametresi; tests/test_contracts.py her PR'da |

---

## 12. İlk Gün

- [ ] Bölüm 4'teki kurulum (venv + torch + bağımlılıklar + smoke test) — iki bilgisayarda da
- [ ] **P0.1** ile repo iskeletini kur, ilk commit'i at
- [ ] `CLAUDE.md`'yi (Bölüm 8.2) repo köküne koy, `AGENTS.md` olarak kopyala
- [ ] Bu rehberi ve brif HTML'ini `docs/` altına taşı
- [ ] Faz 0 veri aramasına başla: AI City Track 3 / CityFlow (MTMC), PKLot, CNRPark+EXT, Roboflow Universe — lisans koşullarını nota işle
- [ ] Aday videoları **P0.2** aracıyla teknik değerlendir
- [ ] `veri_degerlendirme_notu.md` taslağını aç (aday + uygun/değil gerekçesi tablosu)
- [ ] Karar netleşince **Faz 0 bitmeden Sezer'e bildir** — sonra Faz 1

---

*Bu rehber brifin yöntem serbestisini kullanarak somut bir yol öneriyor; brifle çelişen tek bir madde bile bulursanız brif kazanır. Değişiklik önerilerini bu dosyanın üstüne "Değişiklik Günlüğü" bölümü açarak işleyelim.*

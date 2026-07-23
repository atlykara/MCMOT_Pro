# MCMOT_Pro — FAZ 1 Rehberi · Tek Kamera Tespit + Takip (YOLO11m)

> **Faz 1 · Gün 3-4 · Ağırlık %15 · Hedef Senaryo A (Akıllı Otopark)**
> **Model kararı:** `yolo11m.pt` (Ultralytics YOLO11-medium) + ByteTrack
> **Amaç:** Her kamerada bağımsız olarak araçları bulmak ve aynı araca o kamera içinde geçici bir `track_id` vermek.
> **Bitti sayılır (brif):** En az iki kamerada araçların çoğu görünür şekilde tespit/takip ediliyor; belirgin ID değişimleri loglanır ve örneklenir.

Bu dosya sadece **Faz 1** içindir. Faz 1'de **kameralar arası birleştirme YOK** — o Faz 3'ün işi. Burada her kamera **kendi başına** bir tespit + takip hattı olur.

---

## İçindekiler

1. [Faz 1'in Zihinsel Modeli](#1-faz-1in-zihinsel-modeli)
2. [Model Kararı: Neden YOLO11m](#2-model-kararı-neden-yolo11m)
3. [Veri Kontratı — tracks_<cam>.jsonl](#3-veri-kontratı)
4. [Başlamadan Önce Kontrol](#4-başlamadan-önce-kontrol)
5. [Claude Code Promptları (P1.0 → P1.4)](#5-claude-code-promptları)
6. [Faz 1 Bitirme Kontrol Listesi](#6-faz-1-bitirme-kontrol-listesi)
7. [İpuçları ve Sık Hatalar](#7-ipuçları-ve-sık-hatalar)

---

## 1. Faz 1'in Zihinsel Modeli

Faz 1 tek bir soruyu güvenilir cevaplıyor: **"Bu kamerada, bu karede hangi araçlar var ve her biri bir önceki karedeki hangi araçla aynı?"**

İki ayrı bileşen var — karıştırma:

| Bileşen | Ne yapar | Hafızası |
|---|---|---|
| **Tespit** (YOLO11m) | Her karede araçları kutular | Yok — her kare sıfırdan |
| **Takip** (ByteTrack) | Ardışık karelerdeki kutuları bağlar, `track_id` verir | Var — kareler arası eşleştirme |

`track_id` **tek kamera içinde geçerli ve geçicidir.** Araç kameradan çıkıp geri girerse numarası değişebilir — bu normaldir. Faz 1'de bunu **düzeltmeye çalışmıyoruz**; brifin istediği gibi sadece **loglayıp örnekliyoruz**. Kalıcı kimlik (`match_id`) Faz 3'ün işi.

**Değişmez kurallar (Faz 1'de de geçerli):**

- **Re-ID YASAK:** Görünüm embedding'i, derin Re-ID ağı, plaka OCR yok. ByteTrack sadece hareket/konum kullanır — kurala uyar.
- **Konum referansı `foot_point`:** Aracın konumu = kutunun **alt-orta noktası** `[(x1+x2)/2, y2]`. Faz 2 ve 3'ün tüm mantığı buna dayanır.
- **Kontrat sabit:** Aşağıdaki JSONL şemasının alan adları asla değiştirilmez — Efe (camA) ve Atalay (camB) çıktılarının Faz 3'te birleşmesinin garantisi budur.

---

## 2. Model Kararı: Neden YOLO11m

Bu proje `yolo11m.pt` ile yürüyecek. Gerekçe (Faz 1 teslim notuna da yazılacak):

- **Doğruluk:** Medium gövde, nano/small'a göre uzak ve küçük araçlarda belirgin daha isabetli — otopark kamerasında araçlar çoğunlukla uzakta ve sabit, kaçırmamak önemli.
- **Hız:** GPU'muz var (CUDA); 11m gerçek zamanlı gereksinimimiz olmayan video işleme için rahat yeterli.
- **Sıfır eğitim:** `car(2)`, `bus(5)`, `truck(7)` COCO sınıfları hazır — fine-tune gerekmiyor.
- **Ekosistem:** Ultralytics API'si tek satırda ByteTrack entegrasyonu veriyor; iki hat (camA/camB) aynı kodla çalışır.

Pratik notlar:

- Model dosyası ilk çalıştırmada otomatik iner (~40 MB); `*.pt` dosyaları `.gitignore`'da, git'e girmez.
- CPU'ya düşülürse 11m yavaş kalabilir → geliştirme sırasında `--max-frames` ve sample kesitleriyle çalış; gerekirse hız kıyası için P1.4'te `yolo11s` de ölçülüyor.
- Tüm scriptlerde model adı **parametre** (`--model yolo11m.pt` varsayılan) — koda gömülmez; ileride değiştirmek tek bayrak.

---

## 3. Veri Kontratı

**Faz 1'in ana çıktısı:** `outputs/tracks/tracks_<cam>.jsonl` — her karede tespit edilen **her araç için bir satır** (JSON Lines).

```json
{"timestamp": 12.4833, "frame": 299, "camera_id": "camA", "track_id": 17,
 "class": "car", "conf": 0.87,
 "bbox_xyxy": [412.1, 220.5, 588.9, 340.2],
 "foot_point": [500.5, 340.2],
 "hints": {"dominant_color": [128, 130, 135], "size_class": "medium", "aspect_ratio": 1.62}}
```

| Alan | Tip | Açıklama |
|---|---|---|
| `timestamp` | float | Videonun başından saniye (`frame / fps`). Gerçek saat ofseti `cameras.yaml`'da tutulur, buraya işlenmez |
| `frame` | int | Kare numarası (0'dan başlar) |
| `camera_id` | str | `cameras.yaml`'daki kimlik (`camA`, `camB`) |
| `track_id` | int | ByteTrack'in verdiği tek-kamera içi kimlik |
| `class` | str | `car` \| `bus` \| `truck` |
| `conf` | float | Tespit güven skoru (0-1) |
| `bbox_xyxy` | [float×4] | `[x1, y1, x2, y2]` — sol-üst, sağ-alt köşe |
| `foot_point` | [float×2] | `[(x1+x2)/2, y2]` — zemin teması. **Faz 2/3'ün tek konum referansı** |
| `hints.dominant_color` | [int×3] | Kutu alt yarısının medyan **BGR** değeri (OpenCV BGR kullanır!) |
| `hints.size_class` | str | `small` \| `medium` \| `large` (kutu alanı / kare alanı oranı) |
| `hints.aspect_ratio` | float | `w/h` |

> `hints` yalnızca Faz 3'ün **opsiyonel** yardımcı skoru için ucuz, elle tanımlı özelliklerdir. Buraya embedding/öğrenilmiş vektör konulmaz — o Re-ID sayılır ve yasaktır.

---

## 4. Başlamadan Önce Kontrol

```bat
cd /d E:\MCMOT_Pro
.venv\Scripts\activate
python -c "import torch, ultralytics, cv2, pandas, yaml; print('ortam OK | cuda:', torch.cuda.is_available())"
```

- [ ] `cuda: True` görünüyor (GPU'da çalışacağız; 11m için önemli)
- [ ] Videolar yerinde: `data/raw/camA/...mp4` ve `data/raw/camB/...mp4`
- [ ] `configs/cameras.yaml` gerçek `video_path` ve **doğru fps** ile dolu (fps'i probe çıktısından teyit et — `timestamp = frame/fps`, yanlış fps tüm zamanları kaydırır)
- [ ] Repo kökünde `CLAUDE.md` var (Claude Code her oturumda otomatik okur)

**Çalışma disiplini:** Bir prompt = bir teslim; sırayla git. Önce `data/samples/` kesitiyle koştur, tam videoyu gün sonunda bir kez. Her çıktıyı gözle doğrula — kod çalışsa bile alan adı yanlışsa Faz 3'te patlar.

---

## 5. Claude Code Promptları

> Kullanım: promptu code block'tan kopyala, `<...>` yerlerini kendi değerinle doldur, Claude Code'a yapıştır. Her promptun **"Bitti sayılır"** satırı kabul kriterindir — o sağlanmadan sonraki prompta geçme.

### P1.0 — Sample kesiti üretici (hızlı iterasyon için)

```text
scripts/00_make_sample.py adlı küçük bir yardımcı araç yaz. Amaç: büyük ham videodan
Faz 1 geliştirmesi için kısa bir test kesiti çıkarmak.

CLI: python scripts/00_make_sample.py --camera camA --start 60 --duration 45
- configs/cameras.yaml'dan camera_id'ye ait video_path'i oku.
- --start saniyesinden başlayıp --duration saniye uzunluğunda kesit al.
- OpenCV ile kare kare yeniden kodlayarak data/samples/<camera>_sample.mp4 olarak kaydet;
  orijinal fps'i ve çözünürlüğü koru.
- Konsola: yazılan kare sayısı, süre ve çıktı yolunu yaz.

Kısıtlar: Windows/pathlib; büyük video olabilir, tüm videoyu belleğe alma, kare kare işle.

Bitti sayılır: komut çalışınca data/samples/camA_sample.mp4 oluşuyor ve oynatılabiliyor.
```

---

### P1.1 — Tespit + takip hattı (Faz 1'in ANA teslimi)

```text
Faz 1: tek kamera araç tespit + takip hattını yaz. Model: yolo11m.pt (Ultralytics YOLO11).
Çekirdek mantık modülde, çalıştırma script'te olacak (test edilebilirlik için ayrık).

Dosyalar:
- src/mcmot/io_utils.py: JSONL yazıcı (satır satır, akış halinde) + kontrat alan adlarını
  tek yerde tutan sabit (TRACK_FIELDS gibi).
- src/mcmot/detect_track.py: çekirdek — video yolu + config alır, kare kare tespit+takip
  yapar, her araç için kontrat kaydı üretir (generator/yield tercih et).
- scripts/01_run_detect_track.py: argparse CLI.

CLI:
  python scripts/01_run_detect_track.py --camera camA [--model yolo11m.pt]
         [--source data/samples/camA_sample.mp4] [--max-frames N] [--conf 0.35] [--no-video]
- --source verilmezse configs/cameras.yaml'daki video_path kullanılır; fps de oradan okunur.
- Model adı parametredir, koda sabit gömme; varsayılan yolo11m.pt.
- Takip: model.track(source, tracker="bytetrack.yaml", persist=True, classes=[2,5,7],
  conf=<conf>, stream=True) kullan. (COCO: car=2, bus=5, truck=7)
- device otomatik: CUDA varsa GPU, yoksa CPU (konsola hangisi seçildiğini yaz).

Her karede tespit edilen HER araç için outputs/tracks/tracks_<camera>.jsonl dosyasına
TAM OLARAK şu şemayla bir JSON satırı yaz (alan adlarını DEĞİŞTİRME):
{"timestamp": <frame/fps, float>, "frame": <int>, "camera_id": "<str>",
 "track_id": <int>, "class": "<car|bus|truck>", "conf": <float, 4 hane>,
 "bbox_xyxy": [x1,y1,x2,y2], "foot_point": [(x1+x2)/2, y2],
 "hints": {"dominant_color": [B,G,R], "size_class": "<small|medium|large>",
           "aspect_ratio": <w/h, float>}}

hints hesaplama (UCUZ olsun, model kullanma):
- dominant_color: bbox'un ALT YARISINDAKİ piksellerin medyan BGR değeri (numpy median, int).
- size_class: (bbox alanı / kare alanı); <0.01 small, 0.01-0.04 medium, >0.04 large.
- aspect_ratio: (x2-x1)/(y2-y1), 2 hane.

track_id atanmamış tespitleri ATLA — sadece track_id'si olanları yaz.

Ek çıktı (--no-video verilmediyse): outputs/videos/annotated_<camera>.mp4
- her araca kutu, üstüne "id:<track_id> <class> <conf>" etiketi, foot_point'e küçük dolu daire.

YASAK: Re-ID, görünüm embedding'i, appearance tabanlı tracker kolu (BoT-SORT with_reid),
plaka/OCR. Bunları kullanma ve önerme.

Kısıtlar: Windows/pathlib; JSONL'i akış halinde yaz (kareleri bellekte biriktirme);
script tekrar çalıştırılınca eski çıktının üstüne baştan yazsın (append değil).

Bitti sayılır: data/samples/camA_sample.mp4 ile koşunca hem JSONL hem annotated video
oluşuyor; JSONL'in ilk satırı json.loads ile parse edilip yukarıdaki TÜM alanları içeriyor;
annotated videoda kutular araçlara oturuyor ve konsolda device=cuda görünüyor.
```

---

### P1.2 — Kontrat testi (iki hattın uyum güvencesi)

```text
tests/test_contracts.py içine tracks JSONL çıktısı için pytest şema testleri yaz. Amaç:
camA ve camB hatlarının ürettiği dosyaların Faz 3'te birleşebilmesi için formatın birebir
aynı olduğunu garanti etmek.

Testler:
- outputs/tracks/ altındaki tracks_*.jsonl dosyalarını bul; hiç yoksa skip.
- Her dosyanın ilk 500 satırında doğrula:
  * geçerli JSON
  * anahtarlar TAM olarak: timestamp, frame, camera_id, track_id, class, conf,
    bbox_xyxy, foot_point, hints
  * tipler: timestamp float, frame int, track_id int, conf float (0-1),
    class ∈ {car, bus, truck}
  * bbox_xyxy 4 sayı; x2>x1 ve y2>y1
  * foot_point == [(x1+x2)/2, y2] (küçük float toleransıyla)
  * hints: dominant_color 3 int (0-255), size_class ∈ {small, medium, large},
    aspect_ratio float > 0
- Ayrı test: frame numaraları dosya içinde azalmıyor (artan/eşit sırada).

Bitti sayılır: pytest tests/test_contracts.py kontrata uygun dosyada yeşil; kasten
bozulmuş bir satırda ilgili testi kırmızı veriyor.
```

---

### P1.3 — Takip kalite raporu + ID switch günlüğü (brif şartı)

```text
scripts/01_track_quality.py yaz. Amaç: Faz 1 bitiş şartı — belirgin ID değişimlerini
LOGLAMAK ve ÖRNEKLEMEK.

CLI: python scripts/01_track_quality.py --camera camA [--source <video>]
Girdi: outputs/tracks/tracks_<camera>.jsonl (+ örnek kareler için video)

Üret (konsol + outputs/tracks/quality_<camera>.md):
1. Genel istatistik: toplam benzersiz track_id, kare başına ortalama araç sayısı,
   track başına ortalama/medyan yaşam süresi (sn).
2. Kısa ömürlü track listesi: <1.5 sn yaşayanlar (muhtemel hayalet/bölünme) — id, süre.
3. ŞÜPHELİ ID DEĞİŞİMİ adayları (asıl istenen): bir track_id kaybolduktan sonra <=1.0 sn
   içinde, son foot_point'ine <75 piksel mesafede yeni bir track_id doğuyorsa
   (eski_id, yeni_id, timestamp, mesafe_px) tablosuna yaz.
4. ÖRNEKLEME: şüpheli listeden en fazla 5 tanesi için o timestamp'in karesini videodan al;
   eski ve yeni track'i farklı renkle kutula, foot_point'leri işaretle;
   outputs/tracks/switch_samples/<camera>_<eski>_<yeni>.jpg olarak kaydet.

Rapor sonuna "## Yorum" başlığı altında boş bölüm bırak (switch sebeplerini ben yazacağım).

Kısıtlar: Windows/pathlib; JSONL'i akış halinde oku.

Bitti sayılır: quality_<camera>.md oluşuyor, şüpheli switch tablosu (boş da olsa) var,
ve en az bir switch varsa örnek jpg üretiliyor.
```

---

### P1.4 — Parametre karşılaştırması (ölçerek karar ver)

```text
Tracker/model parametrelerini ölçerek karşılaştırmam için scripts/01_sweep_tracker.py yaz.

Aynı --source üzerinde şu ayarları sırayla koştur:
  (a) conf eşiği: 0.25 / 0.35 / 0.50 (model: yolo11m.pt, tracker: bytetrack)
  (b) tracker kıyası: bytetrack.yaml vs botsort.yaml (botsort'ta with_reid KAPALI olacak —
      ReID kolunu açma)
  (c) hız referansı için tek koşu: yolo11s.pt + bytetrack + conf 0.35

Her koşu için ölç:
- toplam benzersiz track_id sayısı
- kare başına ortalama araç sayısı
- tahmini switch sayısı (01_track_quality'deki şüpheli-switch mantığını ortak fonksiyona
  çıkar ve paylaş)
- işleme hızı (fps)

Çıktı: outputs/tracks/sweep_<camera>.csv
(kolonlar: ayar, model, tracker, conf, benzersiz_id, ort_arac, tahmini_switch, islem_fps)
+ konsola okunabilir tablo. Video render ETME, sadece metrik.

Not: "en az benzersiz id" tek başına iyi değildir — araç kaçırarak id azaltmak kötüdür.
Ham sayıları ver, kararı ben vereceğim.

YASAK: botsort with_reid=True; herhangi bir görünüm/embedding modeli.

Bitti sayılır: sweep CSV'sinde en az 5 ayar satırı var ve tüm metrik kolonları dolu.
```

---

### PG.1 — Hata ayıklama şablonu (gerektiğinde)

```text
Şu hatayı ayıkla. Bağlam: MCMOT_Pro Faz 1, <script adı> çalıştırıyorum.
Komut: <tam komut>
Beklenen: <ne olmalıydı>
Olan: <ne oldu>
Hata çıktısı:
<traceback'i TAM yapıştır>

Önce kök nedeni tek cümleyle söyle, sonra minimal düzeltmeyi yap. tracks JSONL kontrat
alanlarını (FAZ1_REHBERI.md Bölüm 3) değiştirme. Düzeltince aynı komutu sample ile
çalıştırıp doğrula.
```

### PG.2 — Kod inceleme şablonu (ekip arkadaşının hattını denetlet)

```text
<dosya yolu> dosyasını incele. MCMOT_Pro Faz 1'in parçası (tek kamera tespit+takip,
yolo11m + bytetrack). Önem sırasına göre raporla; benden onay almadan kod DEĞİŞTİRME.

Denetim maddeleri:
1. Kontrat uyumu: tracks JSONL alan adları/tipleri FAZ1_REHBERI.md Bölüm 3 ile birebir mi?
2. Yasak ihlali: Re-ID / embedding / appearance-ReID / OCR var mı, ima eden import var mı?
3. foot_point doğru mu ([(x1+x2)/2, y2])? hints ucuz mu (model kullanmıyor)?
4. Model adı parametre mi, koda gömülü mü?
5. Windows/pathlib, encoding, akış halinde okuma-yazma (bellek).
6. Kenar durumlar: hiç araç yok, track_id None, video açılamadı, fps 0.
```

---

## 6. Faz 1 Bitirme Kontrol Listesi

Faz 1'i "%100 bitti" saymadan önce hepsini işaretle:

- [ ] **P1.1** ile `tracks_camA.jsonl` **ve** `tracks_camB.jsonl` üretildi (iki kamera da, yolo11m ile)
- [ ] Her kamera için `annotated_<cam>.mp4` var; **gözle izlendi** — kutular araçlara oturuyor, `track_id`'ler makul sabit
- [ ] **P1.2** kontrat testi iki dosyada da yeşil
- [ ] **P1.3** ile `quality_<cam>.md` üretildi; ID değişimleri **loglandı ve örneklendi** (switch jpg'leri var)
- [ ] **P1.4** sweep'i koşuldu; seçilen conf/tracker ayarı ve gerekçesi teslim notuna yazıldı
- [ ] "Araçların çoğu" tespit ediliyor — annotated video + kare başına ortalama araç sayısıyla teyit
- [ ] Günlük rapor yazıldı (kanıt: commit + JSONL örneği + annotated video görüntüsü + quality raporu)

Hepsi tamamsa → **Faz 2** (kamera içi konum/bölge) rehberine geç.

---

## 7. İpuçları ve Sık Hatalar

- **`fps`'i doğru gir.** `timestamp = frame / fps`; `cameras.yaml`'daki fps yanlışsa tüm zaman damgaları kayar ve Faz 3 zaman eşleştirmesi çöker. Gerçek fps'i probe çıktısından teyit et.
- **`persist=True` şart.** `model.track()` çağrısında `persist=True` olmazsa her kare bağımsız sayılır, takip kopar.
- **ID switch'i Faz 1'de düzeltmeye çalışma.** Brif sadece log + örnek istiyor; kalıcı kimlik Faz 3'ün işi. Burada mükemmeliyetçilik zaman kaybı.
- **BGR/RGB tuzağı.** OpenCV BGR okur; `dominant_color` kontratta BGR — Faz 3'te renk kıyası yapan da BGR beklemeli.
- **Küçük/uzak araçlar kaçıyorsa** önce `--conf 0.25` dene; hâlâ kaçırıyorsa P1.4'e `yolo11l` satırı ekletip ölç — tahminle değil, sweep sonucuyla karar ver.
- **GPU belleği yetmezse** (11m + yüksek çözünürlük): Ultralytics `imgsz` parametresini 960'a düşürmek çoğu zaman yeterli; bunu da sweep'e ekletebilirsin.
- **İki hattı senkron tut.** Efe ve Atalay aynı script sürümünü kullanmalı; `01_run_detect_track.py`'de değişiklik yapan diğerine haber verir; `test_contracts.py`'yi ikiniz de koşun.
- **Tam videoyu en son koştur.** Geliştirme sample ile; tam koşu gün sonunda bir kez.

---

*Bu rehber yalnızca Faz 1'i kapsar ve model kararı olarak yolo11m.pt'yi temel alır. Faz 1 çıktıları güvenilir olduğunda Faz 2 rehberini açarız. Brifle çelişen bir şey görürsen brif kazanır.*

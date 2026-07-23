# MCMOT_Pro — FAZ 0 Rehberi (Veri & Karar) + Claude Code Başlangıç Promptları

> **Faz 0 · Gün 1-2 · Ağırlık %10 · Hedef Senaryo A (Akıllı Otopark)**
> **Bitti sayılır:** En az iki kameradan aynı alanı gösteren, teknik olarak kullanılabilir bir kaynak **seçilmiş ve gerekçelendirilmiş**; karar **Sezer'e bildirilmiş**.

Bu dosya sadece **Faz 0** içindir. Amaç: ortamı hazırlamak, uygun çok-kameralı otopark verisini bulup teknik olarak değerlendirmek ve senaryo kararını gerekçeleriyle vermek. Kod yazma işi (tespit/takip) Faz 1'de başlar — Faz 0'da **tek satır bile YOLO modeli eğitmiyoruz.**

---

## 0. Faz 0 neyi teslim ediyor?

| Teslim | Ne | Kim üretir |
|---|---|---|
| `veri_degerlendirme_notu.md` | Aday veri setleri + her biri neden uygun/uygun değil + seçilen kaynak | **Sen yazarsın** (karar metni, AI'a yazdırma) |
| Seçilen video(lar)/veri seti | `data/raw/` altına indirilmiş, oynatılabilir | Sen indirir, AI teknik doğrular |
| Yöntem önerisi taslağı | Hangi tespit/takip yaklaşımı denenecek, gerekçesiyle | Sen + AI birlikte |
| Repo iskeleti | Boş klasör yapısı + git | **Claude Code** (P0.2) |

> **Kritik ayrım:** Karar ve gerekçe metinlerini **sen** yazarsın — brif bunu stajyer teslimi olarak istiyor, AI'a yazdırılmaz. AI'ya sadece **ham teknik analiz** (çözünürlük, fps, araç görünüyor mu) ve **iskelet kurulumu** yaptırılır.

---

## 1. Ortam Hazırlığı (kod yazmadan önce)

### 1.1 GPU kontrolü

```bat
nvidia-smi
```

Tablo geliyorsa GPU var (CUDA sürümünü not et). "not recognized" ise CPU yolu — Faz 0 için ikisi de fazlasıyla yeterli, çünkü sadece birkaç kareyi analiz edeceğiz.

### 1.2 Sanal ortam + temel paketler

```bat
cd /d E:\MCMOT_Pro
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

:: PyTorch — GPU (CUDA 12.x); komutu pytorch.org'dan doğrula:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
:: CPU-only ise:  pip install torch torchvision

:: Faz 0 için gereken minimum:
pip install ultralytics opencv-python pandas pyyaml
```

### 1.3 Doğrulama (smoke test)

```bat
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
yolo predict model=yolov8s.pt source="https://ultralytics.com/images/bus.jpg"
```

İkinci komut `runs/detect/predict/` altına kutulu bir görsel yazıyorsa ortam hazır.

---

## 2. Veri Arama Stratejisi

Senaryo A için aradığımız ideal kaynak sırayla:

1. **En iyi:** Görüş alanları **örtüşen** iki kamera — araç aynı anda ikisinde de görünüyor (uzamsal eşleştirme = en sağlam).
2. **İyi:** Katlı/komşu otopark — araç birinden çıkıp diğerine giriyor (zamansal handoff).
3. **Kabul:** Park yerleri/çizgileri görünür, yeterli çözünürlük ve aydınlatma.

### Aday kaynaklar (lisansı nota işlenecek)

| Kaynak | Tip | Senaryo A uygunluğu |
|---|---|---|
| **NVIDIA AI City Challenge — Track 3 (MTMC) / CityFlow** | Çok kameralı, örtüşen/ardışık | En yakın aday; şehir trafiği ağırlıklı ama çok-kamera eşleştirme mantığı birebir |
| **PKLot / CNRPark+EXT** | Otopark doluluk | Tek kamera/tek kare ağırlıklı — çok-kamera eşleştirme için zayıf, doluluk için iyi |
| **Roboflow Universe** ("parking", "multi camera parking") | Karışık | Bazı setlerde örtüşen kamera bulunabilir; tek tek kontrol |
| **Kendi kanıtlama çekimi** | 2 açı (ör. giriş + iç avlu) | Final sayılmaz ama MVP'yi başlatmak için köprü |

> **Mevcut DATASET klasörün Senaryo A'ya uygun değil:** İçindeki AICity Track2 (araç kırpıntısı, ReID) ve Track4 (tek kamera otoyol videosu) çok-kameralı örtüşen otopark verisi değil. Faz 0'ın gerçek işi bu — uygun veriyi bulmak.

---

## 3. Claude Code ile Çalışmaya Başlama

### 3.1 Önce projeyi anlat: `CLAUDE.md`

Repo köküne bir `CLAUDE.md` koy — Claude Code bunu **her oturumda otomatik okur**, böylece her prompta proje kurallarını tekrar yazmazsın. Bu dosyayı **P0.1** oluşturacak. İçeriği (Faz 0 için sadeleştirilmiş):

```markdown
# MCMOT_Pro — Çoklu Kamera Otopark Araç Takibi (SOOS PV-02)

## Proje
2-3 otopark kamerasında YOLO ile araç tespiti + takibi; farklı kameralardaki aynı aracı
Re-ID KULLANMADAN (geometri + zaman + ucuz görünüm ipucu) tek match_id altında birleştirme.
Şu an FAZ 0'dayız: uygun çok-kameralı otopark verisi bulma ve senaryo kararı. Kod yok, hazırlık var.

## Kesin kurallar
- Re-ID YASAK: öğrenilmiş görünüm embedding'i, derin Re-ID ağı, plaka OCR kullanma/önerme.
- Faz 0'da model EĞİTİLMEZ. Sadece hazır yolov8s ile birkaç kare üstünde "araç görünüyor mu"
  kontrolü yapılır.
- Karar/gerekçe metinlerini BEN yazarım; sen sadece ham teknik veri üret ve iskelet kur.
- Windows ortamı: yol işlemleri için pathlib kullan, os.path string birleştirme yapma.

## Teknik
- Python 3.11, venv: .venv | Ultralytics YOLO (yolov8s) | OpenCV, pandas, PyYAML.
- Her script scripts/ altında CLI (argparse). GPU yoksa CPU'da çalışsın (device otomatik).
```

### 3.2 Claude Code'u başlatma

```bat
cd /d E:\MCMOT_Pro
.venv\Scripts\activate
claude
```

Sonra aşağıdaki promptları **sırayla** ver. Her prompt tek bir teslim üretir; birini bitirmeden diğerine geçme.

---

## 4. FAZ 0 BAŞLANGIÇ PROMPTLARI

> Kullanım: `<...>` içindeki yerleri kendi değerlerinle doldur. Promptlar Türkçe — Claude Code sorunsuz anlar. Her promptun sonundaki **"Bitti sayılır"** satırı, çıktının doğru olup olmadığını anlamanın ölçüsüdür.

### P0.1 — Proje iskeleti + CLAUDE.md kurulumu

```text
E:\MCMOT_Pro içinde çok kameralı otopark araç takip projesi (MCMOT) için repo iskeleti kur.
Şu an Faz 0'dayız; sadece iskelet ve hazırlık istiyorum, işlev kodu YAZMA.

Yapılacaklar:
1. Şu klasör ağacını oluştur (boş klasörlere .gitkeep koy):
   docs/, data/raw/camA/, data/raw/camB/, data/samples/, configs/, src/mcmot/, scripts/,
   outputs/probe/, tests/
2. src/mcmot/__init__.py oluştur (bir satır docstring).
3. requirements.txt: ultralytics>=8.2, opencv-python>=4.9, pandas>=2.0, pyyaml>=6.0
   (torch/torchvision'ın pytorch.org'dan platforma göre ayrı kurulduğunu yorum satırıyla belirt).
4. .gitignore: .venv/, __pycache__/, data/raw/, outputs/, runs/, *.pt, .DS_Store
5. configs/cameras.yaml şablonu oluştur — her kamera için şu alanlar:
   camera_id, video_path, fps, clock_offset_seconds (varsayılan 0.0), notes.
   camA ve camB için boş örnek girdiler koy.
6. Repo köküne CLAUDE.md dosyası oluştur. İçeriği AYNEN şu olsun:
---
<CLAUDE.md içeriğini buraya yapıştır — yukarıdaki Bölüm 3.1'den>
---
7. git init yap ve anlamlı ilk commit at: "chore: Faz 0 proje iskeleti ve CLAUDE.md".

Kısıtlar:
- Windows ortamı; tüm yol işlemleri pathlib ile.
- Hiçbir tespit/takip/analiz kodu yazma; bu Faz 1'in işi.

Bitti sayılır: klasör ağacı yukarıdakiyle birebir aynı; CLAUDE.md kökte; git log'da 1 commit var.
```

> **Not:** 6. maddeye Bölüm 3.1'deki CLAUDE.md metnini olduğu gibi yapıştır. Claude Code kendi kurallarını böyle öğrenir.

---

### P0.2 — Aday video/veri seti teknik analiz aracı

```text
scripts/00_probe_video.py adlı bir CLI aracı yaz. Amaç: Faz 0'da aday otopark videolarını
teknik olarak değerlendirmek — böylece hangisinin projeye uygun olduğuna sağlıklı karar verebilirim.

Girdi: --input <dosya veya klasör> (mp4/avi/mkv/mov destekle, klasörse içindeki tüm videoları tara)

Her video için çıkar:
- çözünürlük (genişlik x yükseklik), fps, süre (sn), toplam kare, codec (mümkünse)
- 5 eşit aralıklı karede yolov8s ile araç sayısı (sadece class: car=2, bus=5, truck=7; conf>=0.35)
  → böylece "bu açıdan araçlar görünüyor mu, model çalışıyor mu" anlaşılır
- her videodan 3 örnek kareyi tespit kutularıyla birlikte outputs/probe/<video_adı>/ altına jpg kaydet

Çıktı:
- outputs/probe/probe_report.csv — kolonlar: video, cozunurluk, fps, sure_sn, toplam_kare,
  ort_arac_sayisi, min_conf, uygunluk_notu(boş, elle dolduracağım)
- konsola okunabilir özet tablo (hangi video kaç araç, ne çözünürlük)

Kısıtlar:
- Windows; pathlib kullan.
- yolov8s.pt yoksa ultralytics otomatik indirsin, buna izin ver.
- GPU yoksa CPU'da çalışsın (device'ı otomatik seç).
- Model EĞİTME, sadece hazır ağırlıkla çıkarım yap. Re-ID/embedding kullanma.

Bitti sayılır: elimdeki bir mp4 ile çalıştırdığımda probe_report.csv ve örnek jpg'ler oluşuyor;
konsolda video başına araç sayısı görünüyor.
```

---

### P0.3 — Örtüşme/handoff hızlı görsel kontrolü (iki kamera geldiğinde)

```text
İki aday kameranın görüş alanlarının örtüşüp örtüşmediğini gözle kontrol etmem için basit bir
araç yaz: scripts/00_check_overlap.py

Girdi: --camA <video yolu> --camB <video yolu> [--t <saniye, varsayılan 0>]
Yap:
- Her iki videodan --t saniyesine denk gelen kareyi al.
- İkisini yan yana tek bir görüntüde birleştir (üstlerine "CAM A" / "CAM B" ve zaman etiketi yaz).
- yolov8s ile araçları tespit edip her iki karede de kutula (Faz 0 kontrolü için, conf>=0.35).
- outputs/probe/overlap_<t>.jpg olarak kaydet ve konsola "kaydedildi" yolunu yaz.
- --scan verilirse: 0'dan videonun sonuna kadar 15 sn aralıklarla bu yan-yana görselleri üret
  (örtüşme olup olmadığını hızlıca taramam için).

Amaç: iki kamerada AYNI araçların/AYNI alanın görünüp görünmediğini bana göstermek — bu,
projeyi Durum 1 (uzamsal/örtüşen) mi yoksa Durum 2 (zamansal/ardışık) mı yürüteceğimizi belirler.

Kısıtlar: Windows/pathlib; GPU yoksa CPU; Re-ID/embedding kullanma; model eğitme.

Bitti sayılır: iki video verince yan yana kutulu görsel oluşuyor; --scan ile birden çok tarama
görseli üretiliyor.
```

---

### P0.4 — Veri değerlendirme notu iskeleti (metni SEN dolduracaksın)

```text
docs/veri_degerlendirme_notu.md için bir ŞABLON oluştur. İçeriği ben dolduracağım —
sen sadece boş yapıyı kur, kendi kararını/verini uydurma.

Şablon şu bölümleri içersin (her birine kısa italik açıklama + doldurulacak boş alan koy):

# Faz 0 — Veri Değerlendirme Notu
- Tarih / Hazırlayan:
- Hedef senaryo: Senaryo A (Akıllı Otopark)

## 1. Aranan koşullar (hatırlatma)
   (Bölüm: en az 2 kamera, örtüşen tercih, park yerleri görünür, yeterli çözünürlük)

## 2. İncelenen aday kaynaklar
   Şu kolonlu bir tablo koy (boş satırlarla):
   | Aday kaynak | Erişim/Lisans | Kamera sayısı | Örtüşme var mı | Çözünürlük/FPS | probe sonucu | Uygun mu | Gerekçe |

## 3. Teknik analiz özeti
   (outputs/probe/probe_report.csv ve overlap görsellerine referans için boş alan)

## 4. Seçilen kaynak ve gerekçe
   (Hangisi seçildi, neden; örtüşen mi ardışık mı; hangi senaryo — A/B)

## 5. Yöntem önerisi taslağı
   (Hangi tespit modeli/tracker denenecek ve neden — kısa gerekçe)

## 6. Sezer'e bildirim
   (Ne zaman, ne bildirildi — karar noktası kuralı gereği)

Sadece şablonu oluştur, içini doldurma. Markdown temiz ve okunabilir olsun.

Bitti sayılır: docs/veri_degerlendirme_notu.md yukarıdaki 6 bölümle, boş doldurulabilir halde var.
```

---

## 5. Faz 0 Bitirme Kontrol Listesi

Faz 0'ı "%100 bitti" saymadan önce hepsini işaretle:

- [ ] Ortam kuruldu, smoke test geçti (torch + `yolo predict` çalışıyor)
- [ ] **P0.1** ile repo iskeleti + `CLAUDE.md` hazır, ilk commit atıldı
- [ ] En az 2 kameralı aday veri seti bulundu ve `data/raw/` altına indirildi
- [ ] **P0.2** ile adaylar teknik analiz edildi (`probe_report.csv` var)
- [ ] **P0.3** ile örtüşme/handoff durumu görsel kontrol edildi — Durum 1 mi Durum 2 mi belli
- [ ] `veri_degerlendirme_notu.md` **senin elinle** dolduruldu (aday tablosu + seçim + gerekçe)
- [ ] Yöntem önerisi taslağı yazıldı (hangi tracker, neden)
- [ ] Senaryo kararı verildi → **Faz 0 bitmeden Sezer'e bildirildi**
- [ ] Günlük rapor yazıldı (kanıt: commit + probe çıktısı + not dosyası)

Ancak bu liste tamamlanınca Faz 1'e geçilir.

---

## 6. Faz 0'a Özel İpuçları

- **Kod yazmaya değil veri bulmaya odaklan.** Faz 0'ın gerçek zorluğu doğru veriyi bulmak; scriptler sadece kararını desteklemek için.
- **Örtüşen kamera bulabilirsen Faz 3 çok kolaylaşır.** Bir gün fazla arayıp örtüşen veri bulmak, sonraki bir haftayı kurtarır. P0.3 bunu erken görmen için var.
- **Lisansı ilk günden not et.** "Sonra bakarız" deme — veri değerlendirme notunda lisans kolonu boş kalırsa teslim eksik sayılır.
- **AI'a karar verdirme.** "Hangi veri seti daha iyi?" diye sorabilirsin ama nihai seçim ve gerekçe metni senin — brif bunu açıkça stajyer çıktısı olarak istiyor.
- **Küçük başla:** Büyük veri setinden 30-60 sn'lik kesitler alıp `data/samples/`'a koy; Faz 1'de bunlarla hızlı iterasyon yaparsın.

---

*Bu rehber yalnızca Faz 0'ı kapsar. Faz 1 (tespit/takip) başladığında bir sonraki rehberi birlikte açarız. Brifle çelişen bir şey görürsen brif kazanır.*

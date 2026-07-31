# Faz 3 - Girdi Envanteri

Bu rapor `scripts/03_check_inputs.py` tarafindan 2026-07-31 20:06:25 tarihinde uretildi. Script salt okur; girdi dosyalari degistirilmemistir.

- Proje koku: `E:\MCMOT_Pro\MCMOT_Pro`
- Kamera config: `configs/cameras.yaml`
- Kameralar: camA, camB
- hints ornegi: her kameranin ilk 200 kaydi

## ENGEL (0)

Bu script'in kontrol ettigi kalemlerde engel olculmedi: beklenen girdi dosyalarinin tamami diskte ve `zone_tracks_mapped_<cam>.csv` semasi beklenen kolonlarla birebir ayni.


## 1. Dosya varligi

| dosya | rol | var mi | boyut | satir | son degisiklik |
|---|---|---|---|---|---|
| `configs/cameras.yaml` | kamera config | var | 693 B (693 bayt) | 20 | 2026-07-31 18:53:25 |
| `configs/zone_params.yaml` | bolge tanimi | var | 1.2 KB (1204 bayt) | 31 | 2026-07-30 21:27:03 |
| `configs/direction_mapping.yaml` | yon kural tablosu | var | 2.9 KB (2936 bayt) | 112 | 2026-07-31 19:27:05 |
| `configs/zones_camA.yaml` | camA bolge poligonlari | var | 618 B (618 bayt) | 30 | 2026-07-31 19:05:23 |
| `configs/zones_camB.yaml` | camB bolge poligonlari | var | 551 B (551 bayt) | 27 | 2026-07-31 19:22:23 |
| `outputs/tracks/tracks_camA.jsonl` | camA takip kayitlari | var | 11.5 MB (12012074 bayt) | 43013 | 2026-07-31 18:56:25 |
| `outputs/tracks/tracks_camB.jsonl` | camB takip kayitlari | var | 8.8 MB (9205863 bayt) | 32956 | 2026-07-31 18:59:30 |
| `outputs/zones/zone_events_camA.csv` | camA bolge olaylari | var | 20.7 KB (21237 bayt) | 267 | 2026-07-31 19:23:20 |
| `outputs/zones/zone_tracks_camA.csv` | camA ziyaret ozeti | var | 14.5 KB (14832 bayt) | 106 | 2026-07-31 19:23:33 |
| `outputs/zones/zone_tracks_mapped_camA.csv` | camA etiketli ziyaret | var | 18.7 KB (19171 bayt) | 106 | 2026-07-31 19:27:32 |
| `outputs/zones/zone_events_camB.csv` | camB bolge olaylari | var | 22.3 KB (22820 bayt) | 285 | 2026-07-31 19:23:22 |
| `outputs/zones/zone_tracks_camB.csv` | camB ziyaret ozeti | var | 13.8 KB (14103 bayt) | 101 | 2026-07-31 19:23:35 |
| `outputs/zones/zone_tracks_mapped_camB.csv` | camB etiketli ziyaret | var | 17.8 KB (18222 bayt) | 101 | 2026-07-31 19:27:32 |

Bu tabloya gore beklenen 13 dosyadan 13 tanesi diskte, 0 tanesi eksiktir.

## 2. Sema - zone_tracks_mapped_<cam>.csv

Beklenen kolon sayisi: 25

### camA

- Gercek kolon sayisi: 25
- Gercek kolon listesi: `['camera_id', 'zone_id', 'track_id', 'visit_index', 't_enter', 't_exit', 'frame_enter', 'frame_exit', 'dwell_s', 'n_frames_inside', 'start_foot_x', 'start_foot_y', 'end_foot_x', 'end_foot_y', 'dx', 'dy', 'distance_px', 'speed_px_s', 'direction_label', 'lane', 'class_mode', 'conf_mean', 'exit_reason', 'movement_label', 'mapping_rule']`
- Eksik kolon (0): yok
- Fazla kolon (0): yok
- Sira farki (0): yok

### camB

- Gercek kolon sayisi: 25
- Gercek kolon listesi: `['camera_id', 'zone_id', 'track_id', 'visit_index', 't_enter', 't_exit', 'frame_enter', 'frame_exit', 'dwell_s', 'n_frames_inside', 'start_foot_x', 'start_foot_y', 'end_foot_x', 'end_foot_y', 'dx', 'dy', 'distance_px', 'speed_px_s', 'direction_label', 'lane', 'class_mode', 'conf_mean', 'exit_reason', 'movement_label', 'mapping_rule']`
- Eksik kolon (0): yok
- Fazla kolon (0): yok
- Sira farki (0): yok

Bu tabloya gore 2 / 2 kamerada kolon adi, sayisi ve sirasi beklenen 25 kolonla birebir aynidir.

## 3. Aday havuzlari

### camA - zone_id x movement_label

| zone_id | camA_to_camB | camB_to_camA | toplam |
|---|---|---|---|
| camA_entry | 0 | 56 | 56 |
| camA_exit | 49 | 0 | 49 |
| **toplam** | 49 | 56 | 105 |

### camB - zone_id x movement_label

| zone_id | camA_to_camB | camB_to_camA | toplam |
|---|---|---|---|
| camB_entry | 50 | 0 | 50 |
| camB_exit | 0 | 50 | 50 |
| **toplam** | 50 | 50 | 100 |

### Yon bazli aday havuzlari

- Yon camA_to_camB: camA/camA_exit 49 ziyaret -> camB/camB_entry 50 ziyaret, teorik tavan min(49,50) = 49
- Yon camB_to_camA: camB/camB_exit 50 ziyaret -> camA/camA_entry 56 ziyaret, teorik tavan min(50,56) = 50

- Iki yonun teorik tavan toplami: 99 eslesme

Bu tabloya gore uretilebilecek eslesme sayisi en fazla 99'dir; bu sayi ust sinirdir, gercek eslesme sayisi bunun altinda kalir.

## 4. Zaman

| kamera | t_enter min | t_enter maks | t_exit maks | video suresi tahmini (sn) | fps (cameras.yaml) | clock_offset_seconds |
|---|---|---|---|---|---|---|
| camA | 0.167 | 58.583 | 59.983 | 59.983 | 60 | 0.0 |
| camB | 1.300 | 59.300 | 59.983 | 59.983 | 60 | 0.0 |

> UYARI: timestamp degerleri kendi videosuna gore saniyedir; ortak saat henuz kurulmamistir.

- `clock_offset_seconds` diskteki degerler: camA=0.0, camB=0.0
- Ziyaret verisinin kapsadigi sure (t_exit maks): camA=59.983 sn, camB=59.983 sn

Bu tabloya gore iki kameranin zaman ekseni ayri ayri olculmustur; aralarindaki gercek saat farki bu veriden olculememektedir.

## 5. Veri kalitesi bayraklari

Yuzdeler ilgili kameradaki ziyaret satiri sayisina goredir.

| bayrak | camA (n=105) | camB (n=100) | toplam (n=205) |
|---|---|---|---|
| `exit_reason == "track_end"` (t_exit guvenilmez) | 9 (%8.6) | 5 (%5.0) | 14 (%6.8) |
| `movement_label == "other"` (yon atanamadi) | 0 (%0.0) | 0 (%0.0) | 0 (%0.0) |
| `direction_label` ['stationary', 'up', 'down'] icinde ama gercek movement_label almis | 8 (%7.6) | 6 (%6.0) | 14 (%6.8) |
| `dwell_s > 5.0` (durmus arac adayi) | 3 (%2.9) | 0 (%0.0) | 3 (%1.5) |
| `visit_index > 0` (ayni track bolgeye birden cok kez girmis) | 0 (%0.0) | 0 (%0.0) | 0 (%0.0) |
| `distance_px < 40` (yon kaniti zayif) | 3 (%2.9) | 1 (%1.0) | 4 (%2.0) |

Bu tabloya gore en sik gorulen bayrak `exit_reason == "track_end"` (t_exit guvenilmez), 205 ziyaretin 14 tanesinde (%6.8) gorulmektedir.

## 6. `direction_mapping.yaml` denetimi

- `default_movement`: `other`
- Kural sayisi: 18
- `zones_camA.yaml` gercek zone_id listesi: `['camA_exit', 'camA_entry']`
- `zones_camB.yaml` gercek zone_id listesi: `['camB_exit', 'camB_entry']`

### 6a. Olu kurallar (gercek bir bolgeye karsilik gelmeyen)

| kural # | camera_id | zone_id | neden |
|---|---|---|---|
| 14 | camA | camA-Exit | zone_id `zones_camA.yaml` icinde yok |
| 15 | camB | camB-entry | zone_id `zones_camB.yaml` icinde yok |
| 16 | camB | camB-entry | zone_id `zones_camB.yaml` icinde yok |
| 17 | camA | camA-Exit | zone_id `zones_camA.yaml` icinde yok |
| 18 | camB | camB-entry | zone_id `zones_camB.yaml` icinde yok |

### 6b. Hicbir kurala dusmeyen veri kombinasyonlari (default_movement'e duser)

Veride kural disinda kalan kombinasyon yok.

- Kurala dusen ziyaret: 205 / 205 (%100.0)
- Kurala dusmeyen ziyaret: 0 / 205 (%0.0)
- Capraz kontrol, CSV'deki `mapping_rule == "default"` satirlari: camA=0, camB=0

Bu tabloya gore kural tablosundaki 18 kuraldan 5 tanesi olu, 205 ziyaretin 0 tanesi hicbir kurala dusmemektedir.

## 7. `hints` alani (her kameranin ilk 200 kaydi)

Beklenen: `dominant_color` (3 elemanli liste), `size_class` (str), `aspect_ratio` (float)

### camA

- Incelenen kayit: 200 (dosyada cozulen toplam: 43013, bozuk satir: 0)
- `hints` sozlugu bulunan kayit: 200 / 200, bulunmayan: 0

| hints anahtari | bulundugu kayit | gorulen tipler | beklenen | uyum |
|---|---|---|---|---|
| `aspect_ratio` | 200 | float (200) | float | evet |
| `dominant_color` | 200 | liste[3] (200) | 3 elemanli liste | evet |
| `size_class` | 200 | str (200) | str | evet |

### camB

- Incelenen kayit: 200 (dosyada cozulen toplam: 32956, bozuk satir: 0)
- `hints` sozlugu bulunan kayit: 200 / 200, bulunmayan: 0

| hints anahtari | bulundugu kayit | gorulen tipler | beklenen | uyum |
|---|---|---|---|---|
| `aspect_ratio` | 200 | float (200) | float | evet |
| `dominant_color` | 200 | liste[3] (200) | 3 elemanli liste | evet |
| `size_class` | 200 | str (200) | str | evet |

Bu tabloya gore ornekleme icinde beklenen hints semasindan 0 sapma bulunmustur.

## FAZ 3'E GECIS DEGERLENDIRMESI

### 1. Brief'in Faz 3 gecis sarti

Sart: her arac kaydinda en az `timestamp`, `camera_id`, `track_id`, `zone_id` ve zemin/ayak noktasi (piksel) bulunmali.

| kamera | timestamp | camera_id | track_id | foot_point | cozulen kayit |
|---|---|---|---|---|---|
| camA | 43013 (%100.0) | 43013 (%100.0) | 43013 (%100.0) | 43013 (%100.0) | 43013 |
| camB | 32956 (%100.0) | 32956 (%100.0) | 32956 (%100.0) | 32956 (%100.0) | 32956 |

`zone_id` alani takip kaydinda degil, ziyaret ozetindedir; iki dosya `camera_id` + `track_id` uzerinden birlesir:

| kamera | zone_id kolonu | foot kolonlari | ziyaret satiri | ziyaretteki track_id'lerin takip dosyasindaki karsiligi |
|---|---|---|---|---|
| camA | var | 4/4 (start_foot_x, start_foot_y, end_foot_x, end_foot_y) | 105 | 105 / 105 benzersiz track_id |
| camB | var | 4/4 (start_foot_x, start_foot_y, end_foot_x, end_foot_y) | 100 | 100 / 100 benzersiz track_id |

- **Sonuc: Saglaniyor.** Kanit kolonlari: takip kaydinda `timestamp`, `camera_id`, `track_id`, `foot_point`; ziyaret ozetinde `zone_id`, `start_foot_x/y`, `end_foot_x/y`, `t_enter`, `t_exit`.

### 2. Gercekci eslesme sayisi mertebesi

- camA_to_camB: kaynak 49, hedef 50, teorik tavan 49
- camB_to_camA: kaynak 50, hedef 56, teorik tavan 50
- Iki yonun teorik tavan toplami: **99** eslesme
- Ayni veride yon atanamamis (`other`) ziyaret: 0 / 205 (%0.0)
- Yon kaniti zayif (`distance_px < 40`) ziyaret: 4 / 205 (%2.0)
- `exit_reason == "track_end"` ziyaret: 14 / 205 (%6.8)

- **Sonuc:** mertebe on-larla olculur, yuzlerle degil. Ust sinir 99 eslesmedir; her kaynak ziyaretin hedefte karsiligi olacaginin garantisi yoktur (kor bolgede baska yola sapan, duran veya takibi kopan araclar bu sayiyi dusurur). Beklenen gercek sayi bu ust sinirin altindadir ve bu script ile olculemez; ancak Faz 3 eslestirmesi calistirildiktan sonra dogrulanabilir.

### 3. Devam edilebilir mi?

- Bu script'in kontrol ettigi kalemlerde ENGEL olcumu yoktur (eksik dosya, sema farki, okunamayan girdi bulunmadi).

- Faz 3'e girmeden once acik kalan olculmus riskler:
  - Ortak saat kurulmamis: `clock_offset_seconds` her kamerada diskteki degeriyle raporun 4. bolumunde listelenmistir; kameralar arasi gercek zaman farki bu veriden olculememektedir. Transit suresi bir eslestirme sinyali olacaksa bu deger olculmelidir.
  - Yon atanamamis ziyaretler: 0 / 205 (%0.0) satir `other` etiketli olup aday havuzuna girmez.
  - `track_end` ile biten 14 ziyarette `t_exit` gercek bolge cikisini degil takibin kopus anini gosterir; zaman tabanli skorlamada bu satirlar ayri ele alinmalidir.

Bu maddeler olcum sonucudur; Faz 3'e gecis karari kullaniciya aittir.

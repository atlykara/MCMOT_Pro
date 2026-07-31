# Faz 2 - Girdi Envanteri

Bu rapor `scripts/02_check_tracks.py` tarafindan uretildi. Script salt okur; girdi dosyalari degistirilmez.

- Takip dizini: `E:\MCMOT_Pro\MCMOT_Pro\outputs\tracks`
- Kamera config: `E:\MCMOT_Pro\MCMOT_Pro\configs\cameras.yaml`
- Kapsam: her kamera bagimsiz. Kameralar arasi karsilastirma veya birlestirme yapilmadi (Faz 3 isi).

## Ozet

| kamera | tracks kaydi | benzersiz track | timestamp span (sn) | video fps (gercek/config) | uyari |
|---|---|---|---|---|---|
| camA | 43013 | 501 | 59.98 | 60.00 / 60.0 | 1 |
| camB | 32956 | 445 | 59.98 | 60.00 / 60.0 | 1 |

## Uyarilar

- camA: 143 / 501 track 5 kareden kisa (parcalanma veya hayalet tespit adayi).
- camB: 196 / 445 track 5 kareden kisa (parcalanma veya hayalet tespit adayi).

## Kamera - camA

### cameras.yaml girdisi

| alan | deger |
|---|---|
| camera_id | camA |
| video_path | data/samples/camA_sample.mp4 |
| fps | 60.0 |
| clock_offset_seconds | 0.0 |
| frame_size | EKSIK |
| notes | Kayseri otopark kamera 1, 1728x1080 |

### Takip dosyasi

| alan | deger |
|---|---|
| yol | `E:\MCMOT_Pro\MCMOT_Pro\outputs\tracks\tracks_camA.jsonl` |
| dosya var mi | evet |
| satir sayisi | 43013 |
| cozulen kayit | 43013 |
| bozuk satir | 0 |
| dosya boyutu | 11.46 MB (12012074 bayt) |

### Sema (ilk kayit)

- Ust duzey anahtarlar (9): `['timestamp', 'frame', 'camera_id', 'track_id', 'class', 'conf', 'bbox_xyxy', 'foot_point', 'hints']`
- `hints` anahtarlari: `['dominant_color', 'size_class', 'aspect_ratio']`
- Kontratta olup ilk kayitta olmayan alan: yok
- Kontratta olup hints icinde olmayan alan: yok
- Farkli anahtar setine sahip kayit: 0 (hints: 0)

### Zaman ve kare araligi

| olcum | deger |
|---|---|
| timestamp min | 0.000 |
| timestamp max | 59.983 |
| timestamp span (sn) | 59.983 |
| frame min | 0 |
| frame max | 3599 |
| benzersiz track_id | 501 |

### class dagilimi

| class | kayit |
|---|---|
| car | 39738 (92.4%) |
| truck | 2333 (5.4%) |
| bus | 817 (1.9%) |
| motorcycle | 120 (0.3%) |
| bicycle | 5 (0.0%) |

### conf, bbox ve foot_point istatistikleri

| olcum | min | p05 | p50 | p95 | max |
|---|---|---|---|---|---|
| conf | 0.350 | 0.385 | 0.723 | 0.876 | 0.938 |
| bbox genislik (px) | 13.8 | 30.0 | 84.6 | 238.6 | 390.1 |
| bbox yukseklik (px) | 14.3 | 23.5 | 49.0 | 188.9 | 281.9 |
| foot_point x | 6.9 | 82.2 | 717.5 | 1559.5 | 1719.6 |
| foot_point y | 327.8 | 373.7 | 471.9 | 860.9 | 1028.3 |

### foot_point kare sinirlari icinde mi

| alan | deger |
|---|---|
| sinir kaynagi | video (camA_sample.mp4) |
| kare boyutu | 1728x1080 |
| x sinir disi kayit | 0 |
| y sinir disi kayit | 0 |

> `cameras.yaml` icinde `frame_size` alani tanimli degil; kare sinirlari yalnizca videodan okunur, tahmin edilmez.

### Saglik kontrolleri

| kontrol | kapsam | sonuc |
|---|---|---|
| timestamp ~ frame/fps | ilk 1000 kayit, fps=60.0 | maks sapma 0.0000 sn, ortalama 0.0000 sn, tolerans disi 0 |
| foot_point = bbox alt-orta | 43013 kayit, tolerans 0.06 | bozuk x 0, bozuk y 0, toplam bozuk 0 |
| frame sirasi azalmiyor mu | 43013 kayit | azalan gecis 0 |
| ayni frame'de ayni track_id | 43013 kayit | tekrar eden kayit 0 |

### Track omru (kac karede gorunuyor)

| olcum | deger |
|---|---|
| toplam track | 501 |
| min kare | 1 |
| p05 kare | 1.0 |
| p50 kare | 8.0 |
| p95 kare | 355.0 |
| max kare | 1659 |
| medyan kare araligi (ilk-son) | 9.0 |
| en genis kare araligi | 1699 |
| 5 kareden kisa track | 143 |

### Video dosyasi

| alan | deger |
|---|---|
| yol | `E:\MCMOT_Pro\MCMOT_Pro\data\samples\camA_sample.mp4` |
| dosya var mi | evet |
| cv2 ile acildi mi | evet |
| gercek fps | 60.000 |
| config fps | 60.0 |
| fps karsilastirmasi | uyumlu |
| genislik x yukseklik | 1728x1080 |
| toplam kare | 3600 |
| sure (sn) | 60.0 |

## Kamera - camB

### cameras.yaml girdisi

| alan | deger |
|---|---|
| camera_id | camB |
| video_path | data/samples/camB_sample.mp4 |
| fps | 60.0 |
| clock_offset_seconds | 0.0 |
| frame_size | EKSIK |
| notes | Kayseri otopark kamera 2, 1920x1080 |

### Takip dosyasi

| alan | deger |
|---|---|
| yol | `E:\MCMOT_Pro\MCMOT_Pro\outputs\tracks\tracks_camB.jsonl` |
| dosya var mi | evet |
| satir sayisi | 32956 |
| cozulen kayit | 32956 |
| bozuk satir | 0 |
| dosya boyutu | 8.78 MB (9205863 bayt) |

### Sema (ilk kayit)

- Ust duzey anahtarlar (9): `['timestamp', 'frame', 'camera_id', 'track_id', 'class', 'conf', 'bbox_xyxy', 'foot_point', 'hints']`
- `hints` anahtarlari: `['dominant_color', 'size_class', 'aspect_ratio']`
- Kontratta olup ilk kayitta olmayan alan: yok
- Kontratta olup hints icinde olmayan alan: yok
- Farkli anahtar setine sahip kayit: 0 (hints: 0)

### Zaman ve kare araligi

| olcum | deger |
|---|---|
| timestamp min | 0.000 |
| timestamp max | 59.983 |
| timestamp span (sn) | 59.983 |
| frame min | 0 |
| frame max | 3599 |
| benzersiz track_id | 445 |

### class dagilimi

| class | kayit |
|---|---|
| car | 29845 (90.6%) |
| truck | 1882 (5.7%) |
| bus | 1096 (3.3%) |
| motorcycle | 133 (0.4%) |

### conf, bbox ve foot_point istatistikleri

| olcum | min | p05 | p50 | p95 | max |
|---|---|---|---|---|---|
| conf | 0.350 | 0.385 | 0.693 | 0.882 | 0.935 |
| bbox genislik (px) | 16.9 | 34.1 | 91.0 | 307.0 | 879.6 |
| bbox yukseklik (px) | 15.7 | 26.2 | 61.5 | 175.5 | 300.2 |
| foot_point x | 8.7 | 221.9 | 1371.1 | 1814.7 | 1910.8 |
| foot_point y | 179.2 | 247.3 | 396.5 | 936.9 | 1079.2 |

### foot_point kare sinirlari icinde mi

| alan | deger |
|---|---|
| sinir kaynagi | video (camB_sample.mp4) |
| kare boyutu | 1920x1080 |
| x sinir disi kayit | 0 |
| y sinir disi kayit | 0 |

> `cameras.yaml` icinde `frame_size` alani tanimli degil; kare sinirlari yalnizca videodan okunur, tahmin edilmez.

### Saglik kontrolleri

| kontrol | kapsam | sonuc |
|---|---|---|
| timestamp ~ frame/fps | ilk 1000 kayit, fps=60.0 | maks sapma 0.0000 sn, ortalama 0.0000 sn, tolerans disi 0 |
| foot_point = bbox alt-orta | 32956 kayit, tolerans 0.06 | bozuk x 0, bozuk y 0, toplam bozuk 0 |
| frame sirasi azalmiyor mu | 32956 kayit | azalan gecis 0 |
| ayni frame'de ayni track_id | 32956 kayit | tekrar eden kayit 0 |

### Track omru (kac karede gorunuyor)

| olcum | deger |
|---|---|
| toplam track | 445 |
| min kare | 1 |
| p05 kare | 1.0 |
| p50 kare | 7.0 |
| p95 kare | 337.8 |
| max kare | 456 |
| medyan kare araligi (ilk-son) | 8.0 |
| en genis kare araligi | 473 |
| 5 kareden kisa track | 196 |

### Video dosyasi

| alan | deger |
|---|---|
| yol | `E:\MCMOT_Pro\MCMOT_Pro\data\samples\camB_sample.mp4` |
| dosya var mi | evet |
| cv2 ile acildi mi | evet |
| gercek fps | 60.000 |
| config fps | 60.0 |
| fps karsilastirmasi | uyumlu |
| genislik x yukseklik | 1920x1080 |
| toplam kare | 3600 |
| sure (sn) | 60.0 |

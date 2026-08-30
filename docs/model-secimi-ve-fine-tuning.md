# YOLO Model Secimi ve Fine-Tuning

## Karar

Kayseri MOBESE arac tespiti icin temel model olarak `YOLO11s` secildi.

| Olcut | YOLO11s | YOLO26n |
|---|---:|---:|
| COCO mAP50-95 | 47.0 | 40.9 |
| Yerel isleme hizi (CPU, 960 px) | 4.42 FPS | 9.22 FPS |
| Yerel tespit/kare | 9.56 | 6.81 |
| Yerel ortalama guven | 0.644 | 0.622 |

Yerel test iki kameranin ilk 15 saniyesindeki toplam 600 kareyle yapildi.
Bu testte ground-truth etiket bulunmadigi icin yerel sayilar gercek mAP veya
recall degildir. Resmi COCO mAP avantaji ve daha fazla arac gozlemi birlikte
degerlendirilerek dogruluk hedefi icin YOLO11s secildi. YOLO26n, hiz veya dusuk
donanim maliyeti birinci oncelik oldugunda alternatif olarak kalir.

## Fine-Tuning Neyi Degistirir?

COCO agirliklari `car`, `bus`, `truck` gibi genel siniflari bilir; panelvan,
minibus, pickup ve ambulansi ayri hedef siniflar olarak ogrenmemistir. Fine-tuning,
YOLO11s'in daha once ogrendigi kenar, doku ve arac bicimi bilgisini koruyup son
katmanlari Kayseri siniflarina uyarlar.

## Is Akisi

1. `python scripts/01d_prepare_vehicle_dataset.py` ile 360 etiketleme karesi cikar.
2. Kareleri `data/vehicle_dataset/dataset.yaml` siniflarina gore etiketle.
3. Her goruntu icin ayni ada sahip `.txt` etiketi oldugunu kontrol et.
4. `python scripts/01e_train_vehicle_detector.py` ile iki asamali egitimi baslat.
5. Egitim sonunda test bolumu olculur ve en iyi model
   `models/yolo11s_kayseri.pt` olarak kopyalanir.

Birinci egitim asamasinda omurga dondurulur; model once yeni sinif basligini
ogrenir. Ikinci asamada tum katmanlar dusuk ogrenme hizi ile acilir; kamera acisi
ve MOBESE goruntu karakterine uyum saglanir.

Etiket olmadan fine-tuning baslatilmaz. Etiketsiz kareleri "arka plan" sanarak
egitmek modelin araclari gormemeyi ogrenmesine neden olur; egitim betigi eksik
etiket dosyasinda bu nedenle durur.

# Arac Egitim Verisi

Bu klasor YOLO11s modelini Kayseri MOBESE goruntulerine uyarlamak icindir.

```text
vehicle_dataset/
├── dataset.yaml
├── manifest.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Her goruntunun ayni ada sahip bir YOLO etiket dosyasi bulunmalidir. Ornek:

```text
images/train/camA_000010_013.50.jpg
labels/train/camA_000010_013.50.txt
```

Etiket satiri bicimi:

```text
class_id x_center y_center width height
```

Koordinatlar `0-1` araliginda normalize edilir. Goruntude arac yoksa ayni ada
sahip bos bir `.txt` dosyasi birakilir. Eksik `.txt` dosyasi etiketlenmemis
goruntu demektir; egitim betigi bunu hata kabul eder.

Siniflar:

| ID | Sinif | Kapsam |
|---:|---|---|
| 0 | car | Binek otomobil ve SUV |
| 1 | van_minibus | Panelvan, minivan ve minibus |
| 2 | pickup | Acik/kasali pickup |
| 3 | bus | Otobus |
| 4 | truck | Kamyon ve agir ticari |
| 5 | special_vehicle | Ambulans, itfaiye, polis vb. |
| 6 | motorcycle | Motosiklet |

`train`, `val` ve `test` ayni karenin komsu anlarini paylasmamalidir. Hazirlama
betigi bu nedenle videolari zamansal bolumlere ayirir; rastgele kare bolmez.

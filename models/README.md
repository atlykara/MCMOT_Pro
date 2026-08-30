# models

YOLO model agirliklari burada tutulur.

Temel ve egitilmis dosyalar:

```text
models/yolo11s.pt
models/yolo11s_kayseri.pt
```

`yolo11s.pt` COCO on-egitimli temel agirliktir. `yolo11s_kayseri.pt`,
`scripts/01e_train_vehicle_detector.py` tamamlandiginda uretilir ve projede
tercih edilen arac modeli olur. `yolo11n.pt` eski hizli baseline olarak kalir.

Model dosyalari buyuk oldugu icin Git'e eklenmemelidir.

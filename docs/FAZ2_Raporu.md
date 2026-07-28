# MCMOT_Pro — Faz 2 Raporu

> **Faz:** 2 — Konum / Bölge / ROI Ataması  
> **Tarih:** 28 Temmuz 2026  
> **Senaryo:** Senaryo B — Trafik MOBESE kameraları arası araç geçişi  
> **Veri:** Kayseri Belediyesi trafik kamera kayıtları (`camA`, `camB`)  
> **Durum:** Teknik çıktı tamamlandı, Faz 3’e geçişe hazır

---

## 1. Faz 2’nin Amacı

Faz 1’de iki kamera videosu üzerinde YOLO + ByteTrack ile araç tespiti ve tek kamera içi takip yapılmıştı. Faz 1 çıktısı olarak her araç için `track_id`, `bbox_xyxy`, `foot_point`, `timestamp` ve yardımcı ipuçları üretilmişti.

Faz 2’nin amacı bu ham takip kayıtlarını görüntü üzerindeki anlamlı bölgelere bağlamaktır.

Bu projede Senaryo B’ye geçtiğimiz için Faz 2’de park yeri doluluğu yerine **trafik kameraları arasında araç geçişini temsil eden ROI / zone bölgeleri** kullanılmıştır. Yani temel soru şudur:

> Bir araç, kameralar arası geçiş açısından önemli olan bölgeye ne zaman girdi, ne zaman çıktı ve hangi yönde hareket etti?

---

## 2. Kullanılan Kamera ve Bölge Mantığı

Kayseri veri setinde iki kamera bulunmaktadır:

- `camA`: Kayseri trafik kamera 1
- `camB`: Kayseri trafik kamera 2

Bu iki kamera aynı alanı birebir örtüşen şekilde görmemektedir; ancak yolun birbirini takip eden kısımlarını izlemektedir. Bu nedenle Faz 3’te kullanılacak ana mantık **zamansal handoff** olacaktır.

Her iki kamerada da toplam dört fiziksel şerit mantığı vardır:

- `camA` üst şerit: `camB` tarafından gelen ve `camA` yönüne giden araçlar
- `camA` alt şerit: `camA` tarafından çıkıp `camB` yönüne giden araçlar
- `camB` üst şerit: `camB` tarafından çıkıp `camA` yönüne giden araçlar
- `camB` alt şerit: `camA` tarafından gelip `camB` yönüne giren araçlar

Bu nedenle sadece görüntü yönü (`left`, `right`, `up`, `down`) yeterli değildir. Aynı zamanda aracın zone içindeki üst / alt şerit konumu da dikkate alınmıştır.

---

## 3. Üretilen Zone Tanımları

Faz 2’de her kamera için birer ROI/zone tanımlanmıştır:

- `A-EXIT-01`
- `B-ENTRY-01`

Bu bölgeler OpenCV tabanlı interaktif çizim aracı ile referans kareler üzerinde çizilmiştir. İlk çizimlerden sonra ROI sınırları gözle kontrol edilmiş, sınırdan geçen ve sayılmaması gereken bazı araçları azaltmak için bölgeler daraltılmıştır.

Kullanılan zone dosyaları:

- `configs/zones_camA.yaml`
- `configs/zones_camB.yaml`

Güncel zone önizlemeleri:

- `outputs/zones/previews/zones_preview_camA.jpg`
- `outputs/zones/previews/zones_preview_camB.jpg`

---

## 4. Kullanılan Konum Referansı: `foot_point`

Faz 2’de bölge kontrolü için araç kutusunun merkezi değil, `foot_point` kullanılmıştır.

`foot_point` şu şekilde hesaplanır:

```text
foot_point = [(x1 + x2) / 2, y2]
```

Yani araç kutusunun alt-orta noktasıdır.

Bu tercih özellikle eğik kamera görüntülerinde önemlidir. Çünkü ROI / zone poligonları görüntüdeki zemin bölgesini temsil eder. Araç kutusunun merkezi aracın gövdesine denk gelir; ancak `foot_point` aracın zemine temas ettiği noktaya daha yakın bir tahmindir.

Bu yüzden Faz 2’de temel kontrol şudur:

```text
Araç foot_point noktası çizilen ROI poligonunun içinde mi?
```

---

## 5. Yazılan Scriptler

Faz 2 kapsamında aşağıdaki scriptler ve yardımcı modüller kullanılmıştır.

| Dosya | Görev |
|---|---|
| `scripts/02_extract_reference_frames.py` | Kamera videolarından zone çizimi için referans kareler çıkarır |
| `scripts/02_define_zones.py` | Referans kare üzerinde tıklayarak ROI/zone çizdirir |
| `src/mcmot/zones.py` | Point-in-polygon, zone okuma ve zone eşleştirme mantığını içerir |
| `scripts/02_assign_zones.py` | `tracks_<cam>.jsonl` kayıtlarını okuyup `zone_events_<cam>.csv` üretir |
| `scripts/02_summarize_zone_tracks.py` | Zone içindeki her track için giriş/çıkış, süre, yön ve hareket vektörü üretir |
| `scripts/02_apply_direction_mapping.py` | Görüntü yönünü fiziksel hareket etiketine çevirir |
| `scripts/02_preview_zones.py` | Zone poligonlarını referans kare üzerinde görselleştirir |
| `scripts/02_render_zone_video.py` | Sadece ROI içindeki araçları gösteren zone odaklı video üretir |

---

## 6. Üretilen Çıktılar

Faz 2 sonunda aşağıdaki çıktılar üretilmiştir.

### 6.1 Zone event çıktıları

| Kamera | Dosya | Enter | Exit | Toplam event |
|---|---|---:|---:|---:|
| camA | `outputs/zones/zone_events_camA.csv` | 798 | 798 | 1596 |
| camB | `outputs/zones/zone_events_camB.csv` | 956 | 956 | 1912 |

Her iki kamerada da `enter` ve `exit` sayıları dengelidir. Bu, zone içine giren track’lerin daha sonra zone’dan çıktığını ve olay üretiminin temel olarak tutarlı çalıştığını göstermektedir.

### 6.2 Zone track özetleri

| Kamera | Dosya | Toplam zone-track |
|---|---|---:|
| camA | `outputs/zones/zone_tracks_camA.csv` | 796 |
| camB | `outputs/zones/zone_tracks_camB.csv` | 948 |

Bu dosyalarda her `track_id` için zone içindeki hareket özeti tutulmaktadır:

- zone’a giriş zamanı,
- zone’dan çıkış zamanı,
- zone içinde kalma süresi,
- başlangıç / bitiş foot point koordinatı,
- `dx`, `dy`,
- `distance_px`,
- `direction_label`,
- ortalama confidence.

### 6.3 Fiziksel hareket yönü çıktıları

Görüntü yönleri (`left`, `right`, `up`, `down`, `stationary`) tek başına yeterli olmadığı için `configs/direction_mapping.yaml` dosyası oluşturulmuştur.

Bu dosya ile `upper/lower` şerit bilgisi ve görüntü yönü birlikte kullanılarak fiziksel hareket etiketi üretilmiştir:

- `camA_to_camB`
- `camB_to_camA`
- `other`

| Kamera | Dosya | `camA_to_camB` | `camB_to_camA` | `other` |
|---|---|---:|---:|---:|
| camA | `outputs/zones/zone_tracks_mapped_camA.csv` | 204 | 134 | 458 |
| camB | `outputs/zones/zone_tracks_mapped_camB.csv` | 183 | 270 | 495 |

`other` etiketi; kısa, hareketsiz, ters yönde görünen, sınırda kalan veya Faz 3 eşleştirmesinde ana aday yapılmayacak track’leri temsil etmektedir.

### 6.4 ROI odaklı videolar

Faz 1 annotated videoları tüm görüntüdeki araçları gösteriyordu. Faz 2’de ayrıca ROI odaklı videolar üretilmiştir:

- `outputs/videos/zone_annotated_camA.mp4`
- `outputs/videos/zone_annotated_camB.mp4`

Bu videolarda:

- ROI dışı alan karartılmıştır,
- zone poligonu çizilmiştir,
- yalnızca `foot_point` noktası ROI içinde olan araçlar kutulanmıştır.

Bu çıktı, çizilen ROI’nin doğru araçları yakalayıp yakalamadığını gözle kontrol etmek için kullanılmıştır.

---

## 7. Kabul Edilen Sınırlamalar

ROI çizimi sonrasında bazı sınır durumlarında, sayılmak istenmeyen bazı araçların da zone içine girdiği gözlemlenmiştir.

Bu durumun nedenleri:

- kamera perspektifinin eğik olması,
- araç kutusunun her zaman zemindeki gerçek temas noktasını birebir temsil etmemesi,
- `foot_point` yaklaşımının tahmini olması,
- ROI sınırına çok yakın geçen araçların kararsız davranması,
- videoda iki yönlü trafik akışının aynı geniş bölge içinde yer alması.

Bu hatalar minimuma indirilmeye çalışılmıştır; ancak tamamen sıfırlanması beklenmemektedir. Bu nedenle Faz 3’te eşleştirme yapılırken yalnızca zone bilgisi değil, zaman penceresi, hareket yönü, sınıf ve ek skorlar birlikte kullanılacaktır.

---

## 8. Brief ile Uyum Değerlendirmesi

Brief’te Faz 2 için temel beklenti şuydu:

```text
zones_*.yaml + zone_events_*.csv üretilecek;
her araç timestamp, camera_id, track_id, zone_id/park_id ve zemin noktasıyla ilişkilendirilecek.
```

Bu beklenti Senaryo B’ye uyarlanarak karşılanmıştır.

Senaryo A’daki park yeri doluluğu mantığı yerine Senaryo B’de **geçiş/ROI bölgeleri** kullanılmıştır. Bu nedenle `park_id` merkezli `occupancy_snapshot_<cam>.csv` çıktısı bu aşamada üretilmemiştir. Bunun yerine Faz 3 için daha uygun olan `zone_tracks_mapped_<cam>.csv` dosyaları hazırlanmıştır.

Bu tercih proje senaryosunun otoparktan trafik kamera handoff problemine kaydırılmış olmasıyla uyumludur.

---

## 9. Faz 2 Sonuç Kararı

Faz 2 teknik olarak tamamlanmıştır.

Tamamlanan işler:

1. camA ve camB için ROI/zone bölgeleri çizildi.
2. `tracks_<cam>.jsonl` kayıtları zone olaylarına dönüştürüldü.
3. `enter` / `exit` olayları üretildi.
4. Her zone-track için hareket yönü ve süre özeti çıkarıldı.
5. Üst/alt şerit mantığıyla fiziksel hareket etiketleri üretildi.
6. ROI odaklı kontrol videoları oluşturuldu.
7. Minimal sınır hataları kabul edilebilir düzeyde değerlendirildi.

Bu nedenle proje **Faz 3 — kameralar arası aday eşleştirme ve `match_id` üretimi** aşamasına geçmeye hazırdır.

---

## 10. Faz 3’e Geçiş İçin Hazır Girdiler

Faz 3’te kullanılacak ana girdiler:

- `outputs/zones/zone_tracks_mapped_camA.csv`
- `outputs/zones/zone_tracks_mapped_camB.csv`
- `configs/direction_mapping.yaml`
- `configs/zones_camA.yaml`
- `configs/zones_camB.yaml`
- `outputs/tracks/tracks_camA.jsonl`
- `outputs/tracks/tracks_camB.jsonl`

Faz 3’te ilk hedef, `camA_to_camB` ve `camB_to_camA` hareket etiketlerine göre kameralar arası aday eşleşmeleri üretmektir.

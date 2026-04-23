# II. Proje Ödevi Gereksinim Eşlemesi

Kaynak PDF: `../course_materials/pdf/yzm304_proje2_2526.pdf`

## Proje Başlığı

`2_YZM304-CNN-Image-Classification`

## Durum

Bu klasör PDF yönergesindeki ikinci proje gereksinimlerine göre tamamlandı. Çalıştırılabilir deney hattı `src/cnn_image_project.py` içinde, rapor ve sonuç yorumları `README.md` içinde, metrik/figür/model çıktıları `reports/` altında bulunur.

## Gereksinimler

| PDF Gereksinimi | Bu Projede Karşılığı | Durum |
| --- | --- | --- |
| Görüntü veri seti kullanılmalı | `sklearn.datasets.load_digits`, `1x8x8`, 10 sınıflı el yazısı rakam görüntüleri | Tamamlandı |
| Veri ön işleme yapılmalı | `0-1` ölçekleme, kanal boyutu ekleme, train mean/std ile normalizasyon, stratified split | Tamamlandı |
| İlk model LeNet-5 benzeri açık CNN sınıfı olmalı | `LeNetLikeCNN`: `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear` | Tamamlandı |
| İkinci model aynı temel hiperparametrelerle iyileştirilmeli | `ImprovedCNN`: aynı conv/pooling yapı + `BatchNorm2d` + `Dropout` | Tamamlandı |
| İlk iki model aynı train-test setiyle eğitilmeli | Tüm CNN modelleri aynı `random_state=42` split ile eğitildi/test edildi | Tamamlandı |
| Üçüncü model yaygın CNN mimarisi olmalı | `VGGStyleSmallCNN`: VGG blok mantığı küçük görüntülere uyarlandı | Tamamlandı |
| Cross entropy loss kullanılabilir | `torch.nn.CrossEntropyLoss()` | Tamamlandı |
| Optimizer, learning rate, epoch, batch size gerekçelendirilmeli | README Methods içinde gerekçeli tablo var | Tamamlandı |
| Hibrit model için CNN özellikleri `.npy` olarak çıkarılmalı | `reports/models/train_features.npy`, `train_labels.npy`, `test_features.npy`, `test_labels.npy` | Tamamlandı |
| `.npy` veri boyutu ve uzunluğu yazdırılmalı | `feature_shapes.json` ve README tablosu: train `(1437, 64)`, test `(360, 64)` | Tamamlandı |
| Klasik ML modeli CNN özellikleri ile eğitilmeli | `SVC(kernel="rbf", C=10.0, gamma="scale")` | Tamamlandı |
| Tam CNN ile hibrit model karşılaştırılmalı | `ImprovedCNN` tam CNN sonucu ile `hybrid_improved_cnn_features_svc` karşılaştırıldı | Tamamlandı |
| Rapor IMRAD ve References başlıkları içermeli | README: Introduction, Methods, Results, Discussion, References | Tamamlandı |
| Sonuçlar tablo/grafik/confusion matrix ile verilmeli | `model_comparison.*`, `learning_curves.png`, `*_confusion_matrix.png` | Tamamlandı |

## Üretilen Sonuçlar

| Model | Test Accuracy | F1 Macro | Öğrenilebilir Parametre |
| --- | ---: | ---: | ---: |
| `lenet_like_cnn` | 0.9306 | 0.9299 | 5750 |
| `improved_batchnorm_dropout_cnn` | 0.9667 | 0.9661 | 5794 |
| `vgg_style_small_cnn` | 0.9583 | 0.9572 | 29722 |
| `hybrid_improved_cnn_features_svc` | 0.9750 | 0.9747 | 0 |

## Kontrol Komutu

```bash
.venv/bin/python 2_YZM304-CNN-Image-Classification/src/cnn_image_project.py
```

Bu komut metrikleri, confusion matrix görsellerini, learning curve grafiğini, CNN ağırlıklarını ve `.npy` feature dosyalarını yeniden üretir.

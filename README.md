# YZM304-LAB

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi için hazırlanmış proje teslim deposudur. Ödevler ayrı klasörlerde, ders kaynakları ise ayrı `course_materials/` klasöründe tutulur.

## Teslim Klasörleri

| Klasör | İçerik |
| --- | --- |
| `1_YZM304-Banknote-MLP` | Banknote authentication veri seti üzerinde manuel MLP, `scikit-learn` ve `PyTorch` karşılaştırmaları |
| `2_YZM304-CNN-Image-Classification` | Görüntü verisi üzerinde LeNet benzeri CNN, BatchNorm/Dropout CNN, VGG tarzı CNN ve CNN feature extraction + SVC hibrit model |
| `course_materials` | Proje PDF yönergeleri ve ders PPTX dosyaları |

## 1. Proje Özeti

`1_YZM304-Banknote-MLP` klasörü, laboratuvarda kurulan MLP modelini tekrar üretilebilir hale getirir. Aynı veri ayrımı ve aynı başlangıç ağırlıklarıyla manuel, `scikit-learn` ve `PyTorch` backendleri karşılaştırılır. Sonuç tabloları, confusion matrix görselleri ve split özetleri klasör içindeki `artifacts/` altında bulunur.

Detaylı rapor: `1_YZM304-Banknote-MLP/README.md`

## 2. Proje Özeti

`2_YZM304-CNN-Image-Classification` klasörü, `yzm304_proje2_2526.pdf` yönergesine göre hazırlanmış CNN tabanlı görüntü sınıflandırma çalışmasını içerir. Veri seti olarak `sklearn.datasets.load_digits` kullanılmıştır.

| Model | Test Accuracy | F1 Macro |
| --- | ---: | ---: |
| `lenet_like_cnn` | 0.9306 | 0.9299 |
| `improved_batchnorm_dropout_cnn` | 0.9667 | 0.9661 |
| `vgg_style_small_cnn` | 0.9583 | 0.9572 |
| `hybrid_improved_cnn_features_svc` | 0.9750 | 0.9747 |

Detaylı rapor: `2_YZM304-CNN-Image-Classification/README.md`

## Çalıştırma

1. proje ana deney hattı:

```bash
PYTHONPATH=src python -m banknote_mlp.experiment
```

2. proje ana deney hattı:

```bash
.venv/bin/python 2_YZM304-CNN-Image-Classification/src/cnn_image_project.py
```

## Not

`PROJECT_MEMORY.md` yerel çalışma hafızasıdır ve GitHub teslimine dahil edilmez. Sınav notları için kullanılan `NOtlar/` klasörü de bu proje teslim kapsamına eklenmemiştir.

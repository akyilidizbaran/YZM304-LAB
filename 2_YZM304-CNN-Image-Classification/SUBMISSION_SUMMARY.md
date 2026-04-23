# II. Proje Teslim Özeti

## Proje

`2_YZM304-CNN-Image-Classification`

## PDF Gereksinim Durumu

`yzm304_proje2_2526.pdf` içindeki görüntü verisi, ön işleme, açık yazılmış CNN sınıfları, geliştirilmiş CNN, yaygın CNN mimarisi, CNN feature extraction, `.npy` kayıtları, klasik ML hibrit model, tam CNN karşılaştırması, IMRAD README ve sonuç görselleri gereksinimleri tamamlandı.

## Kullanılan Veri

- Veri seti: `sklearn.datasets.load_digits`
- Görüntü şekli: `1x8x8`
- Sınıf sayısı: `10`
- Toplam örnek: `1797`
- Test örneği: `360`

## Modeller

| Model | Rol |
| --- | --- |
| `LeNetLikeCNN` | LeNet-5 benzeri temel CNN |
| `ImprovedCNN` | BatchNorm + Dropout eklenmiş CNN |
| `VGGStyleSmallCNN` | VGG blok mantığına dayalı yaygın mimari tipi CNN |
| `hybrid_improved_cnn_features_svc` | CNN feature extraction + SVC hibrit model |

## Sonuç

| Model | Test Accuracy | F1 Macro |
| --- | ---: | ---: |
| `lenet_like_cnn` | 0.9306 | 0.9299 |
| `improved_batchnorm_dropout_cnn` | 0.9667 | 0.9661 |
| `vgg_style_small_cnn` | 0.9583 | 0.9572 |
| `hybrid_improved_cnn_features_svc` | 0.9750 | 0.9747 |

En iyi sonuç `hybrid_improved_cnn_features_svc` modelinde elde edildi. Tam CNN karşılaştırması için `ImprovedCNN` referans alındı.

## Çalıştırma

```bash
.venv/bin/python 2_YZM304-CNN-Image-Classification/src/cnn_image_project.py
```

## Ana Çıktılar

- `reports/metrics/model_comparison.csv`
- `reports/metrics/model_comparison.json`
- `reports/metrics/data_summary.json`
- `reports/figures/learning_curves.png`
- `reports/figures/*_confusion_matrix.png`
- `reports/models/*.pt`
- `reports/models/*.npy`

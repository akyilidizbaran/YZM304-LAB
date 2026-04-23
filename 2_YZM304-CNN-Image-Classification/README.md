# 2_YZM304-CNN-Image-Classification

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi I. Proje Modülü II. Proje Ödevi için hazırlanmış CNN tabanlı görüntü sınıflandırma çalışmasıdır. Rapor `yzm304_proje2_2526.pdf` yönergesindeki IMRAD yapısına göre düzenlenmiştir.

## Introduction

Bu projenin amacı, evrişimli sinir ağları ile görüntü verilerinden özellik çıkarma ve sınıflandırma yapmayı deneysel olarak incelemektir. Çalışmada dört model karşılaştırılmıştır: LeNet-5 benzeri temel CNN, aynı ana hiperparametreleri koruyup BatchNorm ve Dropout eklenen geliştirilmiş CNN, VGG blok mantığına dayanan yaygın mimari tipi CNN ve CNN özellikleri üzerinde çalışan hibrit `SVC` modeli.

Veri seti olarak `sklearn.datasets.load_digits` kullanılmıştır. Bu veri seti MNIST benzeri, tek kanallı, `8x8` boyutlu, `10` sınıflı el yazısı rakam görüntülerinden oluşan küçük bir benchmark görüntü veri setidir. Küçük boyutlu olması, eğitim hattının hızlı ve tekrar üretilebilir şekilde çalışmasını sağlar.

## Methods

### Veri Seti ve Ön İşleme

Kullanılan veri seti `1797` adet gri seviye rakam görüntüsü içerir. Görüntüler `1x8x8` tensör formatına dönüştürülmüş, piksel değerleri `0-1` aralığına ölçeklenmiş ve eğitim setinden hesaplanan ortalama/standart sapma ile normalize edilmiştir.

Veri ayrımı sabit `random_state=42` ile yapılmıştır:

| Bölüm | Örnek Sayısı | Kullanım |
| --- | ---: | --- |
| Train | 1149 | CNN ağırlık güncellemesi |
| Validation | 288 | Eğitim sırasında doğrulama grafikleri |
| Train+Validation | 1437 | Hibrit model feature extraction eğitimi |
| Test | 360 | Tüm modeller için nihai karşılaştırma |

### Model 1: LeNet Benzeri CNN

İlk model `LeNetLikeCNN` sınıfıdır. Yapı açık olarak PyTorch ile yazılmıştır ve `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten` ve tam bağlantılı sınıflandırıcı katmanlarını içerir. Bu model temel CNN davranışını ölçmek için başlangıç modeli olarak kullanılmıştır.

### Model 2: BatchNorm + Dropout CNN

İkinci model `ImprovedCNN` sınıfıdır. İlk modeldeki ana evrişim hiperparametreleri korunmuştur: iki evrişim bloğu, aynı kanal sayıları, `3x3` kernel ve aynı pooling yapısı. Fark olarak evrişim bloklarına `BatchNorm2d`, sınıflandırıcı öncesine `Dropout(p=0.35)` eklenmiştir. Amaç, normalizasyon ve rastgele sönümlemenin genelleme performansına etkisini aynı train-test ayrımı üzerinde ölçmektir.

### Model 3: VGG Tarzı CNN

Üçüncü model `VGGStyleSmallCNN` sınıfıdır. `torchvision` bağımlılığı gerektirmeden VGG mimarisinin temel fikri olan ardışık küçük `3x3` evrişim blokları ve pooling yapısı küçük `8x8` görüntülere uyarlanmıştır. Bu model literatürde yaygın CNN mimarisi mantığını küçük veri seti üzerinde temsil eder.

### Model 4: Hibrit CNN Feature Extraction + SVC

Hibrit modelde `ImprovedCNN.extract_features` çıktıları kullanılmıştır. CNN sınıflandırıcı kısmı yerine bu özellik vektörleri `.npy` dosyalarına kaydedilmiş, ardından `SVC(kernel="rbf", C=10.0, gamma="scale")` ile klasik makine öğrenmesi modeli eğitilmiştir.

Üretilen feature dosyaları:

| Dosya | Shape |
| --- | ---: |
| `reports/models/train_features.npy` | `(1437, 64)` |
| `reports/models/train_labels.npy` | `(1437,)` |
| `reports/models/test_features.npy` | `(360, 64)` |
| `reports/models/test_labels.npy` | `(360,)` |

PDF yönergesindeki tam CNN ile hibrit model karşılaştırması için `ImprovedCNN`, hibrit modele karşılık gelen tam CNN referansı olarak kullanılmıştır. Bu nedenle ayrıca beşinci bağımsız CNN modeli eklenmemiştir.

### Eğitim Ayarları

| Ayar | Değer | Gerekçe |
| --- | ---: | --- |
| `batch_size` | `64` | Küçük veri setinde kararlı mini-batch eğitimi sağlar |
| `epochs` | `18` | Modellerin yakınsaması için yeterli, aşırı uzun değil |
| `learning_rate` | `0.001` | Adam için güvenli başlangıç değeri |
| `optimizer` | `Adam` | Küçük CNN modellerinde hızlı ve kararlı optimizasyon |
| `loss` | `CrossEntropyLoss` | Çok sınıflı sınıflandırma için standart kayıp |
| `random_state` | `42` | Tekrar üretilebilir split ve sonuçlar |

## Results

Deneyler CPU üzerinde çalıştırılmıştır. Nihai test sonuçları aşağıdadır:

| Model | Tür | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Öğrenilebilir Parametre |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lenet_like_cnn` | CNN | 0.9306 | 0.9372 | 0.9298 | 0.9299 | 5750 |
| `improved_batchnorm_dropout_cnn` | CNN | 0.9667 | 0.9685 | 0.9663 | 0.9661 | 5794 |
| `vgg_style_small_cnn` | CNN | 0.9583 | 0.9577 | 0.9578 | 0.9572 | 29722 |
| `hybrid_improved_cnn_features_svc` | CNN features + SVC | 0.9750 | 0.9751 | 0.9748 | 0.9747 | 0 |

Üretilen ana çıktı dosyaları:

| Çıktı | Yol |
| --- | --- |
| Model karşılaştırma tablosu | `reports/metrics/model_comparison.csv` |
| JSON metrik özeti | `reports/metrics/model_comparison.json` |
| Veri özeti | `reports/metrics/data_summary.json` |
| Öğrenme eğrileri | `reports/figures/learning_curves.png` |
| Confusion matrix görselleri | `reports/figures/*_confusion_matrix.png` |
| CNN ağırlıkları | `reports/models/*.pt` |
| Feature/label `.npy` dosyaları | `reports/models/*.npy` |

## Discussion

Temel LeNet benzeri CNN, düşük parametre sayısına rağmen güçlü bir başlangıç sonucu vermiştir. BatchNorm ve Dropout eklenen ikinci CNN, aynı ana mimari korunmasına rağmen test accuracy değerini `0.9306` seviyesinden `0.9667` seviyesine çıkarmıştır; bu sonuç ek katmanların genelleme performansını iyileştirdiğini gösterir.

VGG tarzı model daha fazla parametreye sahiptir ve `0.9583` accuracy ile güçlü sonuç üretmiştir; ancak bu küçük veri setinde geliştirilmiş daha kompakt CNN'i geçememiştir. Bu durum, daha karmaşık mimarinin her zaman daha iyi sonuç vermediğini ve veri boyutunun model kapasitesiyle birlikte düşünülmesi gerektiğini gösterir.

En yüksek test başarımı `0.9750` accuracy ile hibrit `CNN features + SVC` modelinde elde edilmiştir. Bu sonuç, CNN'in öğrendiği 64 boyutlu özelliklerin klasik bir SVM sınıflandırıcısı için yeterince ayrıştırıcı olduğunu gösterir. Tam CNN ile hibrit model karşılaştırmasında hibrit yaklaşım bu veri seti için daha başarılıdır; fakat daha büyük ve daha çeşitli görüntü veri setlerinde uçtan uca CNN mimarileri avantaj kazanabilir.

## Reproducibility

Kök dizinden çalıştırma:

```bash
.venv/bin/python 2_YZM304-CNN-Image-Classification/src/cnn_image_project.py
```

Proje klasöründen çalıştırma:

```bash
../../.venv/bin/python src/cnn_image_project.py
```

Bağımlılıklar:

```bash
../../.venv/bin/pip install -r requirements.txt
```

Script çalışınca `reports/figures`, `reports/metrics` ve `reports/models` altındaki metrik, grafik, ağırlık ve `.npy` dosyalarını yeniden üretir.

## References

- YZM304 II. Proje Ödevi yönergesi: `../course_materials/pdf/yzm304_proje2_2526.pdf`
- PyTorch: [https://pytorch.org](https://pytorch.org)
- scikit-learn digits dataset: [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- scikit-learn SVC: [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Klasör Yapısı

```text
2_YZM304-CNN-Image-Classification/
├── README.md
├── REQUIREMENTS_FROM_ASSIGNMENT.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
│   └── cnn_image_project.py
└── reports/
    ├── figures/
    ├── metrics/
    └── models/
```

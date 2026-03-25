# YZM304-Banknote-MLP

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi için hazırlanan birinci proje deposu. Çalışma, laboratuvarda geliştirilen MLP örneğini tekrar üretilebilir bir repo yapısına taşır ve aynı veri ayrımı, aynı başlangıç ağırlıkları ve aynı tam-batch SGD hattı üzerinde manuel, `scikit-learn` ve `PyTorch` karşılaştırmalarını içerir.

## Introduction

Amaç, banknote authentication veri seti üzerinde kurulan temel MLP modelini sadece çalıştırmak değil, PDF yönergesindeki deneyleri izlenebilir artefaktlarla tekrar üretilebilir hale getirmektir. Bu sürümde üç kritik boşluk kapatıldı:

- backendler arası aynı başlangıç ağırlıkları `data/weights/` altında kalıcı artefakt olarak üretildi,
- `PyTorch` eşleniği eklendi,
- veri miktarı etkisi için `%50 / %75 / %100` eğitim fraksiyonu deneyleri eklendi.

## Methods

### Veri Seti

- Veri kaynağı: `data/raw/banknote_authentication.csv`
- Problem tipi: ikili sınıflandırma
- Özellikler: `variance`, `skewness`, `curtosis`, `entropy`
- Hedef değişken: `class`
- Toplam örnek sayısı: `1372`
- Toplam sınıf dağılımı: `762` gerçek banknot, `610` sahte banknot

### Deney Ayarları

- Veri karıştırma: `random_state=42`
- Eğitim/doğrulama/test ayrımı: `878 / 220 / 274`
- Eğitim fraksiyonları: `0.50`, `0.75`, `1.00`
- Fraksiyon yapısı: sınıf dengeli ve iç içe geçmiş alt kümeler
- Giriş boyutu: `4`
- Çıkış boyutu: `1`
- Aktivasyonlar: gizli katmanlarda `tanh`, çıkışta `sigmoid`
- Kayıp fonksiyonu: binary cross-entropy
- Optimizasyon: momentum kapalı, tam-batch SGD
- Sınıf tahmin eşiği: `0.5`
- Seçim kuralı: en yüksek doğrulama doğruluğu, eşitlikte daha düşük adım sayısı, sonra daha düşük parametre sayısı

### Ortak Split ve Ağırlık Artefaktları

- Split manifesti: `data/splits/split_manifest.json`
- Tek gizli katman başlangıç ağırlıkları: `data/weights/4-6-1.npz`
- Derin model başlangıç ağırlıkları: `data/weights/4-10-6-1.npz`
- Ağırlık metadataları: `data/weights/4-6-1.json`, `data/weights/4-10-6-1.json`

Bu dosyalar, manuel model, `scikit-learn MLPClassifier` ve `PyTorch` modelinin aynı parametrelerden başlamasını sağlar. Standartlaştırılmış backend karşılaştırmaları tam olarak bu ortak artefaktlar üzerinden yürütülür.

### Repo Yapısı

```text
.
├── data/raw/                         # Ham veri seti
├── data/splits/                      # Sabit train/val/test ve fraksiyon manifesti
├── data/weights/                     # Backendler arası paylaşılan başlangıç ağırlıkları
├── docs/assignment/                  # Ödev yönergesi
├── notebooks/                        # Laboratuvar notebook'u
├── reports/                          # Grafik, metrik ve split çıktıları
├── src/banknote_mlp/                 # Modüler deney kodu
├── tests/                            # Birim testleri
├── README.md
├── requirements.txt
└── requirements-notebook.txt
```

### Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m banknote_mlp.experiment
```

Notebook ile çalışmak için:

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/one_hidden_layer_mlp.ipynb
```

### Uygulanan Deneyler

- `manual_raw_baseline`: ham veri, tek gizli katman (`6`), `500` adım
- `manual_standardized_baseline`: standardize veri, tek gizli katman (`6`), `1000` adım, `learning_rate=0.03`
- `sklearn_standardized_baseline`: standardize veri, tek gizli katman (`6`), `1000` adım, `learning_rate=0.03`
- `pytorch_standardized_baseline`: standardize veri, tek gizli katman (`6`), `1000` adım, `learning_rate=0.03`
- `manual_regularized_deeper_data50`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`, eğitim verisinin `%50`si
- `manual_regularized_deeper_data75`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`, eğitim verisinin `%75`i
- `manual_regularized_deeper_data100`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`, eğitim verisinin tamamı
- `sklearn_regularized_deeper`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`
- `pytorch_regularized_deeper`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`

### Üretilen Artefaktlar

- Tüm deneyler: `reports/metrics/experiment_comparison.csv`
- Backend eşlenik karşılaştırması: `reports/metrics/backend_comparison.csv`
- Markdown özet: `reports/metrics/experiment_summary.md`
- JSON özet: `reports/metrics/experiment_summary.json`
- Split özeti: `reports/models/data_split_summary.json`
- Öğrenme eğrileri: `reports/figures/learning_curves.png`
- Karmaşıklık matrisi görselleri: `reports/figures/*_confusion_matrix.png`

## Results

Veri, stratified sabit bir ayrımla `878` eğitim, `220` doğrulama ve `274` test örneğine bölündü. Fraksiyon deneylerinde eğitim örnek sayıları sırasıyla `439`, `658` ve `878` oldu.

### Backend Karşılaştırması

| Konfigürasyon | Backend | Val Acc | Test Acc | Test F1 |
| --- | --- | ---: | ---: | ---: |
| Standardized baseline | `manual` | `0.9636` | `0.9745` | `0.9719` |
| Standardized baseline | `sklearn` | `0.9636` | `0.9745` | `0.9719` |
| Standardized baseline | `pytorch` | `0.9636` | `0.9745` | `0.9719` |
| Deeper + L2 (`100%` train) | `manual` | `0.9591` | `0.9708` | `0.9680` |
| Deeper + L2 (`100%` train) | `sklearn` | `0.9591` | `0.9708` | `0.9680` |
| Deeper + L2 (`100%` train) | `pytorch` | `0.9591` | `0.9708` | `0.9680` |

Bu tabloda üç backendin de aynı split ve aynı başlangıç ağırlıklarıyla birebir aynı sayılara ulaştığı görülüyor.

### Veri Miktarı Etkisi

| Deney | Train Fraction | Val Acc | Test Acc | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| `manual_regularized_deeper_data50` | `0.50` | `0.9591` | `0.9745` | `0.9719` |
| `manual_regularized_deeper_data75` | `0.75` | `0.9591` | `0.9708` | `0.9680` |
| `manual_regularized_deeper_data100` | `1.00` | `0.9591` | `0.9708` | `0.9680` |

### Genel Sonuç

- Seçim kuralına göre en iyi deney: `manual_raw_baseline`
- En yüksek doğrulama doğruluğu: `0.9727`
- En yüksek test doğruluğu: `0.9745`
- Aynı en yüksek test doğruluğunu veren deneyler: `manual_standardized_baseline`, `sklearn_standardized_baseline`, `pytorch_standardized_baseline`, `manual_regularized_deeper_data50`

## Discussion

En önemli teknik sonuç, backendler arası karşılaştırmanın artık yaklaşık değil birebir kurulmuş olmasıdır. Aynı split manifesti, aynı `.npz` ağırlık artefaktları ve aynı tam-batch SGD düzeni kullanıldığında manuel, `scikit-learn` ve `PyTorch` sonuçları eşleşti. Bu, karşılaştırmanın rastgele başlangıçtan veya farklı eğitim akışlarından etkilenmediğini gösterir.

Veri miktarı deneyi daha ilginç bir tablo verdi. Derin ve L2 regülarize modelde `%50` eğitim fraksiyonu, doğrulama doğruluğunu artırmadan testte en yüksek sonuca ortak oldu. Bu sonucu tek bir split üzerinde gördüğümüz için dikkatli yorumlamak gerekir; yine de model kapasitesi, regülarizasyon ve örnek seçiminin etkileşimini göstermesi açısından rapora dahil edilmesi gereken anlamlı bir deneydir.

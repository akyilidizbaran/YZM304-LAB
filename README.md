# YZM304-Banknote-MLP

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi için hazırlanan birinci proje deposu. Bu repo, laboratuvarda geliştirilen tek gizli katmanlı MLP çalışmasını banknote authentication veri seti üzerinde düzenli, tekrar üretilebilir ve GitHub'a uygun bir yapıya taşımak için hazırlandı.

## Introduction

Bu çalışmanın amacı, banknot doğrulama veri seti üzerinde laboratuvarda kurulan iki katmanlı MLP modelini temel alarak eğitim, test ve iyileştirme deneylerini sistematik hale getirmektir. Proje yönergesine göre tek gizli katmanlı temel model korunmuş, doğrulama seti eklenmiş, daha derin bir manuel model ve `scikit-learn MLPClassifier` karşılaştırması aynı veri ayrımı üzerinde koşturulmuştur.

## Methods

### Veri Seti

- Veri kaynağı: `data/raw/banknote_authentication.csv`
- Problem tipi: ikili sınıflandırma
- Özellikler: `variance`, `skewness`, `curtosis`, `entropy`
- Hedef değişken: `class`

### Deney Ayarları

Bu bölüm, mevcut laboratuvar notebook'undan çıkarılan başlangıç ayarlarının modüler deney hattına taşınmış halini içerir.

- Veri karıştırma: `random_state=42`
- Eğitim/doğrulama/test ayrımı: `0.64 / 0.16 / 0.20`
- Giriş boyutu: `4`
- Çıkış boyutu: `1`
- Gizli katman sayısı: `1`
- Başlangıç gizli nöron sayısı: `6`
- Başlangıç eğitim adımı: `500`
- Öğrenme oranı: `0.01`
- Aktivasyonlar: gizli katmanda `tanh`, çıkışta `sigmoid`
- Kayıp fonksiyonu: binary cross-entropy
- Optimizasyon yaklaşımı: SGD tabanlı parametre güncellemesi
- Sınıf tahmin eşiği: `0.5`
- Veri standardizasyonu: deney bazlı açık/kapalı
- Seçim kuralı: en yüksek doğrulama doğruluğu, eşitlikte daha düşük adım sayısı

### Repo Yapısı

```text
.
├── data/raw/                         # Ham veri seti
├── docs/assignment/                 # Ödev yönergesi
├── notebooks/                       # Laboratuvar notebook'u ve deney notları
├── reports/                         # Grafik, metrik ve veri ayrımı çıktıları
├── src/banknote_mlp/                # Yeniden kullanılabilir Python modülleri
├── tests/                           # Temel birim testleri
├── PROJECT_MEMORY.md                # Kalıcı proje hafızası
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

Notebook ile çalışmak istersen:

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/one_hidden_layer_mlp.ipynb
```

### Uygulanan Deneyler

- `manual_raw_baseline`: ham veri, tek gizli katman (`6` nöron), `500` adım
- `manual_standardized_baseline`: standardize veri, tek gizli katman (`6` nöron), `1000` adım, `learning_rate=0.03`
- `manual_regularized_deeper`: standardize veri, iki gizli katman (`10-6`), `1000` adım, `L2=0.001`
- `sklearn_standardized_baseline`: standardize veri, `MLPClassifier(hidden_layer_sizes=(6,), solver="sgd")`, `1000` iterasyon

### Üretilen Artefaktlar

- Karşılaştırma tablosu: `reports/metrics/experiment_comparison.csv`
- Deney özeti: `reports/metrics/experiment_summary.md`
- JSON özet: `reports/metrics/experiment_summary.json`
- Veri ayrımı özeti: `reports/models/data_split_summary.json`
- Öğrenme eğrileri: `reports/figures/learning_curves.png`
- Karmaşıklık matrisi görselleri: `reports/figures/*_confusion_matrix.png`

## Results

Toplam `1372` örnek, `4` özellik ve sınıf dağılımı `762 / 610` olacak şekilde kullanıldı. Veri, `878` eğitim, `220` doğrulama ve `274` test örneğine ayrıldı. Deney sonuçları aşağıdadır:

| Deney | Doğrulama Doğruluğu | Test Doğruluğu | Test F1 |
| --- | ---: | ---: | ---: |
| `manual_raw_baseline` | `0.9727` | `0.9635` | `0.9576` |
| `manual_standardized_baseline` | `0.9636` | `0.9745` | `0.9719` |
| `manual_regularized_deeper` | `0.9591` | `0.9708` | `0.9680` |
| `sklearn_standardized_baseline` | `0.9591` | `0.9708` | `0.9677` |

Seçim kuralına göre en yüksek doğrulama doğruluğunu verdiği için `manual_raw_baseline` en iyi deney olarak işaretlendi. Saf test başarımı açısından ise `manual_standardized_baseline` en yüksek sonucu verdi.

## Discussion

Sonuçlar, aynı veri ayrımı altında standardizasyon ve daha uzun eğitim süresinin test başarımını yükselttiğini gösteriyor. Bununla birlikte doğrulama setinde en güçlü sonuç ham veriyle çalışan temel manuel modelden geldi; bu da küçük veri koşullarında seçilen validasyon kesitinin model sıralamasını etkileyebildiğini gösteriyor. Derinleştirilmiş ve L2 regülarize edilmiş manuel model ile `scikit-learn` tabanlı model, tek gizli katmanlı standardize manuel modele oldukça yakın sonuç verdi. Sıradaki mantıklı genişletme, aynı çatıya `PyTorch` karşılaştırması, mini-batch eğitim ve ek regularization teknikleri eklemek olacaktır.

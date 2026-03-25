# YZM304-Banknote-MLP

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi için hazırlanan birinci proje deposu. Bu repo, laboratuvarda geliştirilen tek gizli katmanlı MLP çalışmasını banknote authentication veri seti üzerinde düzenli, tekrar üretilebilir ve GitHub'a uygun bir yapıya taşımak için hazırlandı.

## Introduction

Bu çalışmanın amacı, banknot doğrulama veri seti üzerinde laboratuvarda kurulan iki katmanlı MLP modelini temel alarak eğitim, test ve iyileştirme deneylerini sistematik hale getirmektir. Proje yönergesine göre tek gizli katmanlı temel model korunacak, overfitting ve underfitting davranışları incelenecek, ek model varyasyonları ve kütüphane tabanlı yeniden yazımlar için uygun altyapı oluşturulacaktır.

## Methods

### Veri Seti

- Veri kaynağı: `data/raw/banknote_authentication.csv`
- Problem tipi: ikili sınıflandırma
- Özellikler: `variance`, `skewness`, `curtosis`, `entropy`
- Hedef değişken: `class`

### Başlangıç Deney Ayarları

Bu bölüm, mevcut laboratuvar notebook'undan çıkarılan başlangıç ayarlarını içerir.

- Veri karıştırma: `random_state=42`
- Eğitim/test ayrımı: `test_size=0.20`, `stratify=y`
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

### Repo Yapısı

```text
.
├── data/raw/                         # Ham veri seti
├── docs/assignment/                 # Ödev yönergesi
├── notebooks/                       # Laboratuvar notebook'u ve deney notları
├── reports/                         # Grafik, metrik ve model çıktıları
├── src/banknote_mlp/                # Yeniden kullanılabilir Python modülleri
├── tests/                           # İleride eklenecek testler
├── PROJECT_MEMORY.md                # Kalıcı proje hafızası
├── README.md
└── requirements.txt
```

### Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/one_hidden_layer_mlp.ipynb
```

## Results

Şu aşamada depoda, laboratuvarda geliştirilen temel notebook ve veri seti düzenli klasör yapısına taşınmıştır. Sonuç tabloları, karışıklık matrisi görselleri ve alternatif modeller bu iskelet üzerinde ilerleyen adımlarda eklenecektir.

## Discussion

Projede sıradaki teknik adım, notebook içerisindeki temel MLP akışını tekrar kullanılabilir Python modüllerine bölmek ve ardından doğrulama seti, regülarizasyon, çok katmanlı varyantlar ve `scikit-learn` ya da `PyTorch` karşılıklarını eklemektir. Bu repo yapısı, PDF yönergesindeki IMRAD dokümantasyon, metrik raporlama ve GitHub üzerinden teslim gereksinimlerini karşılayacak başlangıç zemini olarak hazırlanmıştır.


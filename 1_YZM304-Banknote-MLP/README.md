# 1_YZM304-Banknote-MLP

Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği `YZM304 Derin Öğrenme` dersi I. Proje Modülü I. Proje Ödevi için hazırlanmış proje teslim özetidir. Çalışmanın ana deposu: [https://github.com/akyilidizbaran/YZM304-LAB](https://github.com/akyilidizbaran/YZM304-LAB)

Bu çalışma, banknote authentication veri seti üzerinde laboratuvarda kurulan tek gizli katmanlı MLP modelini tekrar üretilebilir bir proje yapısına taşır. Manuel MLP, `scikit-learn MLPClassifier` ve `PyTorch` karşılıkları aynı veri ayrımı, aynı başlangıç ağırlıkları ve aynı tam-batch SGD yaklaşımı ile karşılaştırılır.

## Introduction

Projenin amacı, ikili sınıflandırma problemi üzerinde temel MLP modelini yalnızca çalıştırmak değil, model eğitimi, validasyon, test, optimizasyon ve karşılaştırma adımlarını izlenebilir hale getirmektir. Veri seti banknot görüntülerinden çıkarılmış istatistiksel özelliklerden oluşur ve hedef değişken banknotun gerçek ya da sahte olduğunu gösterir.

Bu raporda üç ana araştırma sorusu ele alınır:

- Laboratuvardaki tek gizli katmanlı MLP modeli bu veri setinde hangi performansı verir?
- Standardizasyon, daha derin mimari, L2 regülarizasyonu ve veri miktarı performansı nasıl etkiler?
- Manuel, `scikit-learn` ve `PyTorch` uygulamaları aynı split, ağırlık ve hiperparametrelerle adil biçimde eşleşir mi?

## Methods

### Veri Seti

- Veri dosyası: `../data/raw/banknote_authentication.csv`
- Problem tipi: ikili sınıflandırma
- Özellikler: `variance`, `skewness`, `curtosis`, `entropy`
- Hedef değişken: `class`
- Toplam örnek sayısı: `1372`
- Sınıf dağılımı: `762` gerçek banknot, `610` sahte banknot

### Veri Ayrımı ve Ön İşleme

- Rastgelelik: `random_state=42`
- Eğitim / doğrulama / test ayrımı: `878 / 220 / 274`
- Veri miktarı deneyleri: eğitim setinin `%50`, `%75`, `%100` alt kümeleri
- Standardizasyon: standardize deneylerde eğitim setinden öğrenilen ortalama ve standart sapma kullanıldı
- Sınıf tahmin eşiği: `0.5`

### Modeller

Temel manuel model, tek gizli katmanlı MLP mimarisidir. Giriş boyutu `4`, çıkış boyutu `1`, gizli katman aktivasyonu `tanh`, çıkış aktivasyonu `sigmoid` olarak ayarlandı. Kayıp fonksiyonu binary cross-entropy, optimizer ise momentum kapalı tam-batch SGD olarak kullanıldı.

Karşılaştırılan deneyler:

| Deney | Backend | Standartlaştırma | Gizli Katman | Adım | Ek Ayar |
| --- | --- | --- | --- | ---: | --- |
| `manual_raw_baseline` | manual | Hayır | `6` | `500` | temel model |
| `manual_standardized_baseline` | manual | Evet | `6` | `1000` | `learning_rate=0.03` |
| `sklearn_standardized_baseline` | sklearn | Evet | `6` | `1000` | aynı ağırlık ve split |
| `pytorch_standardized_baseline` | pytorch | Evet | `6` | `1000` | aynı ağırlık ve split |
| `manual_regularized_deeper_data50` | manual | Evet | `10-6` | `1000` | `L2=0.001`, `%50` veri |
| `manual_regularized_deeper_data75` | manual | Evet | `10-6` | `1000` | `L2=0.001`, `%75` veri |
| `manual_regularized_deeper_data100` | manual | Evet | `10-6` | `1000` | `L2=0.001`, `%100` veri |
| `sklearn_regularized_deeper` | sklearn | Evet | `10-6` | `1000` | aynı ağırlık ve split |
| `pytorch_regularized_deeper` | pytorch | Evet | `10-6` | `1000` | aynı ağırlık ve split |

### Tekrar Üretilebilirlik

Ortak split ve ağırlık artefaktları ana repoda tutulur:

- Split manifesti: `../data/splits/split_manifest.json`
- Tek gizli katman ağırlıkları: `../data/weights/4-6-1.npz`
- Derin model ağırlıkları: `../data/weights/4-10-6-1.npz`
- Backend karşılaştırması: `artifacts/metrics/backend_comparison.csv`
- Tüm deney tablosu: `artifacts/metrics/experiment_comparison.csv`

Çalıştırma:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m banknote_mlp.experiment
```

## Results

### Backend Karşılaştırması

| Konfigürasyon | Backend | Val Acc | Test Acc | Test F1 |
| --- | --- | ---: | ---: | ---: |
| Standardized baseline | `manual` | `0.9636` | `0.9745` | `0.9719` |
| Standardized baseline | `sklearn` | `0.9636` | `0.9745` | `0.9719` |
| Standardized baseline | `pytorch` | `0.9636` | `0.9745` | `0.9719` |
| Deeper + L2 (`100%` train) | `manual` | `0.9591` | `0.9708` | `0.9680` |
| Deeper + L2 (`100%` train) | `sklearn` | `0.9591` | `0.9708` | `0.9680` |
| Deeper + L2 (`100%` train) | `pytorch` | `0.9591` | `0.9708` | `0.9680` |

Üç backend aynı split, aynı başlangıç ağırlıkları ve aynı tam-batch SGD mantığı ile eşleşti. Bu nedenle sonuç farkı framework rastgeleliğinden değil, gerçekten aynı deney koşullarından elde edildi.

### Veri Miktarı Etkisi

| Deney | Train Fraction | Val Acc | Test Acc | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| `manual_regularized_deeper_data50` | `0.50` | `0.9591` | `0.9745` | `0.9719` |
| `manual_regularized_deeper_data75` | `0.75` | `0.9591` | `0.9708` | `0.9680` |
| `manual_regularized_deeper_data100` | `1.00` | `0.9591` | `0.9708` | `0.9680` |

### Genel Sonuçlar

- Seçim kuralına göre en iyi deney: `manual_raw_baseline`
- En yüksek doğrulama doğruluğu: `0.9727`
- En yüksek test doğruluğu: `0.9745`
- Aynı en yüksek test doğruluğunu veren deneyler: `manual_standardized_baseline`, `sklearn_standardized_baseline`, `pytorch_standardized_baseline`, `manual_regularized_deeper_data50`
- Karmaşıklık matrisi görselleri: `artifacts/figures/*_confusion_matrix.png`
- Öğrenme eğrileri: `artifacts/figures/learning_curves.png`

## Discussion

Sonuçlar, banknote authentication veri setinin temel MLP ile yüksek başarıya ulaşabildiğini gösterir. Ham veriyle çalışan `manual_raw_baseline`, validasyon doğruluğu açısından seçim kuralına göre en iyi modeldir. Buna karşılık standardize tek gizli katmanlı modeller test doğruluğunda daha yüksek sonuca ulaşmıştır.

Backend karşılaştırması projenin en kritik kısmıdır. Manuel, `scikit-learn` ve `PyTorch` uygulamaları aynı başlangıç ağırlıkları, aynı split ve aynı optimizasyon hattı ile birebir aynı metrikleri üretmiştir. Bu durum, model davranışının framework farkından bağımsız olarak doğrulandığını gösterir.

Veri miktarı deneyinde `%50` eğitim fraksiyonu test başarımında en yüksek skora ortak olmuştur. Bu sonuç tek split üzerinde değerlendirildiği için genellenebilirlik açısından dikkatli yorumlanmalıdır. Gelecek çalışmada çoklu split, cross-validation, mini-batch eğitim ve batch normalization deneyleri eklenebilir.

## References

- YZM304 I. Proje Ödevi yönergesi: `../course_materials/pdf/yzm304_proje_odevi1_2526.pdf`
- Proje deposu: [https://github.com/akyilidizbaran/YZM304-LAB](https://github.com/akyilidizbaran/YZM304-LAB)
- `scikit-learn MLPClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- PyTorch: [https://pytorch.org](https://pytorch.org)

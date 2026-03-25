# PROJECT_MEMORY

## 0) TL;DR (En güncel durum)

* Şu an ne yapıyoruz? `YZM304-Banknote-MLP` için modüler deney hattısı kuruldu, temel deneyler üretildi ve rapor artefaktları repo içine yazıldı.
* Son değişiklik neydi? Manuel MLP, veri ayrımı, metrik üretimi, sklearn karşılaştırması ve deney script'i eklendi; `reports/` altında sonuçlar üretildi.
* Bir sonraki net adım ne? Notebook'u yeni modülleri kullanacak şekilde sadeleştirmek ve istenirse `PyTorch` eşleniğini eklemek.

## 1) Proje Amacı ve Kapsam

* Amaç: Banknote authentication veri seti üzerinde laboratuvarda kurulan tek gizli katmanlı MLP çalışmasını tekrar üretilebilir proje yapısına taşımak ve PDF yönergesindeki deneyleri bu temel üstünde yürütmek.
* Kapsam içi: Veri seti organizasyonu, notebook düzenleme, IMRAD biçimli README, deney sonuç klasörleri, manuel MLP ve `scikit-learn` tabanlı yeniden yazım.
* Kapsam dışı: Bu aşamada `PyTorch` implementasyonu ve tüm ileri düzenlileştirme tekniklerinin tamamlanması.

## 2) Non-negotiables / Kırmızı Çizgiler

* Repo yapısı tekrar üretilebilir olmalı.
* README, PDF yönergesine uygun olarak IMRAD formatında tutulmalı.
* Veri, notebook, kaynak kod ve sonuçlar ayrı klasörlerde tutulmalı.
* Laboratuvar temel modeli korunmalı; sonraki deneyler bunun üstüne inşa edilmeli.

## 3) Mimari Özet

* Bileşenler: Ham veri seti, laboratuvar notebook'u, Python kaynak kodu için `src/`, deney çıktıları için `reports/`, temel testler için `tests/`.
* Veri akışı: `data/raw` içindeki CSV, `src/banknote_mlp/data.py` ile ayrılıyor; manuel ve sklearn modelleri `src/banknote_mlp/` altında eğitiliyor; deney çıktıları `reports/` altında tutuluyor.
* Önemli dizinler/modüller: `data/raw/`, `docs/assignment/`, `notebooks/`, `src/banknote_mlp/`, `reports/`, `tests/`.

## 4) Konvansiyonlar ve Standartlar

* Kod stili / lint / format: Şimdilik sade Python ve notebook akışı; ileride modüler Python dosyalarında PEP 8 izlenecek.
* Branch/commit yaklaşımı: Ana çalışma `main` üzerinde ilk kurulum; anlamlı küçük commit'ler tercih edilmeli.
* İsimlendirme/klasör düzeni: Veri için `data/raw`, yönerge için `docs/assignment`, deney çıktıları için `reports`, kaynak kod için `src/banknote_mlp`.

## 5) Kurulum & Çalıştırma

* Gereksinimler: Python 3, `venv`.
* Komutlar:
  * `python3 -m venv .venv`
  * `source .venv/bin/activate`
  * `pip install -r requirements.txt`
  * `PYTHONPATH=src python -m banknote_mlp.experiment`
  * `pip install -r requirements-notebook.txt`
  * `jupyter notebook notebooks/one_hidden_layer_mlp.ipynb`
* Ortam değişkenleri (sadece İSİMLER): `PYTHONPATH`
* Lokal geliştirme notları: Hafif deney kurulumu `requirements.txt`; notebook için ek bağımlılıklar `requirements-notebook.txt`.

## 6) Decision Log (append-only)

* 2026-03-25 — Karar: Proje adı `YZM304-Banknote-MLP` olarak kullanılacak. | Gerekçe: Kullanıcı tarafından açıkça belirtildi. | Etki: README başlığı, proje hafızası ve GitHub teslim dili bu adla hizalanacak. | Alternatifler: Yok.
* 2026-03-25 — Karar: Repo yapısı `data`, `docs`, `notebooks`, `reports`, `src`, `tests` ekseninde kurulacak. | Gerekçe: PDF yönergesi uygun klasör hiyerarşisi ve tekrar üretilebilirlik istiyor. | Etki: Mevcut veri, notebook ve yönerge dosyaları taşınacak; yeni modüller bu yapı altında geliştirilecek. | Alternatifler: Tüm dosyaları kökte tutmak.
* 2026-03-25 — Karar: Çalıştırılabilir deney hattı manuel MLP + `scikit-learn MLPClassifier` karşılaştırması şeklinde modülerleştirilecek. | Gerekçe: PDF, laboratuvar modelinin yeniden yazımı ve kütüphane tabanlı karşılaştırma istiyor. | Etki: `src/banknote_mlp/` altında veri, model, metrik ve deney modülleri eklendi. | Alternatifler: Tüm mantığı notebook içinde bırakmak.
* 2026-03-25 — Karar: Hafif çalışma ortamı için `requirements.txt`, notebook ekleri için `requirements-notebook.txt` kullanılacak. | Gerekçe: Jupyter bağımlılıklarını ana deney hattısından ayırmak kurulumu hızlandırıyor. | Etki: CLI deney komutu daha hızlı kuruluyor; notebook kullanımı ayrı dosyada korunuyor. | Alternatifler: Tüm bağımlılıkları tek dosyada tutmak.

## 7) Milestones / Dönüm Noktaları (append-only)

* 2026-03-25 — Milestone: PDF yönergesi analiz edildi. | Sonuç: Teslim için gereken klasör yapısı, README biçimi ve başlangıç hiperparametreleri netleştirildi.
* 2026-03-25 — Milestone: İlk repo iskeleti kuruldu ve GitHub'a pushlandı. | Sonuç: `data/`, `docs/`, `notebooks/`, `reports/`, `src/` ve `tests/` yapısı oluşturuldu; uzak repo ile `main` dalı eşitlendi.
* 2026-03-25 — Milestone: Modüler deney hattı çalıştırıldı. | Sonuç: Manuel MLP, derin varyant ve sklearn karşılaştırması için metrikler, karmaşıklık matrisleri ve öğrenme eğrileri üretildi.

## 8) Yapılanlar

* [x] PDF yönergesi metin olarak çıkarıldı ve gereksinimler belirlendi.
* [x] Mevcut veri seti ve laboratuvar notebook'u tespit edildi.
* [x] Dosya yapısı organize edildi.
* [x] Uzak repoya ilk push yapıldı.
* [x] Modüler veri ayrımı, manuel MLP ve sklearn deney hattısı yazıldı.
* [x] Deney çıktıları `reports/` altında üretildi.

## 9) Yapılacaklar (Next)

* [ ] Notebook'u `src/banknote_mlp` modüllerini kullanacak şekilde sadeleştir.
* [ ] `PyTorch` ile eşdeğer model karşılaştırmasını ekle.
* [ ] Overfitting-underfitting yorumunu README içinde daha ayrıntılı yaz.
* [ ] İstenirse mini-batch ya da batch normalization deneyi ekle.

## 10) Bilinen Sorunlar / Teknik Borç / Riskler

* `scikit-learn` tarafında aynı başlangıç ağırlıklarını manuel model ile birebir eşlemek pratik değil; karşılaştırma mimari ve optimizasyon ayarları üzerinden yaklaşık tutuluyor.
* Notebook henüz modüler Python kodunu kullanacak şekilde yeniden yazılmadı; şu an referans laboratuvar çalışması olarak duruyor.

## 11) Notlar ve Tuzaklar (Pitfalls)

* Notebook veri yolu repo yapısına göre `../data/raw/banknote_authentication.csv` olarak düzeltildi; fakat eğitim mantığı hala notebook içinde kopya halde duruyor.
* Uzak GitHub deposu `https://github.com/akyilidizbaran/YZM304-LAB` adresine `main` dalı ile bağlandı.
* `reports/metrics/experiment_summary.md` seçim kuralına göre en iyi modeli verir; test doğruluğu en yüksek model farklı olabilir.

### Güncelleme Kaydı

* Son güncelleme: 2026-03-25

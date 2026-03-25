# PROJECT_MEMORY

## 0) TL;DR (En güncel durum)

* Şu an ne yapıyoruz? `YZM304-Banknote-MLP` için temel repo iskeleti kuruldu ve GitHub'a ilk push yapıldı.
* Son değişiklik neydi? Veri, notebook ve PDF yönergesi klasör hiyerarşisine taşındı; README, `requirements.txt` ve başlangıç Python paketi eklendi.
* Bir sonraki net adım ne? Temel notebook akışını modüler Python koduna ayırmak ve PDF'teki deney genişletmelerini uygulamak.

## 1) Proje Amacı ve Kapsam

* Amaç: Banknote authentication veri seti üzerinde laboratuvarda kurulan tek gizli katmanlı MLP çalışmasını tekrar üretilebilir proje yapısına taşımak ve PDF yönergesindeki deneyleri bu temel üstünde yürütmek.
* Kapsam içi: Veri seti organizasyonu, notebook düzenleme, IMRAD biçimli README, deney sonuç klasörleri, ileride eklenecek manuel MLP ve kütüphane tabanlı yeniden yazımlar.
* Kapsam dışı: Bu aşamada tam deney sonuçlarının tamamlanması ve ileri model varyantlarının tümünün uygulanması.

## 2) Non-negotiables / Kırmızı Çizgiler

* Repo yapısı tekrar üretilebilir olmalı.
* README, PDF yönergesine uygun olarak IMRAD formatında tutulmalı.
* Veri, notebook, kaynak kod ve sonuçlar ayrı klasörlerde tutulmalı.
* Laboratuvar temel modeli korunmalı; sonraki deneyler bunun üstüne inşa edilmeli.

## 3) Mimari Özet

* Bileşenler: Ham veri seti, laboratuvar notebook'u, Python kaynak kodu için `src/`, deney çıktıları için `reports/`.
* Veri akışı: `data/raw` içindeki CSV notebook ve ileride eklenecek Python modülleri tarafından okunacak; deney çıktıları `reports/` altında tutulacak.
* Önemli dizinler/modüller: `data/raw/`, `docs/assignment/`, `notebooks/`, `src/banknote_mlp/`, `reports/`.

## 4) Konvansiyonlar ve Standartlar

* Kod stili / lint / format: Şimdilik sade Python ve notebook akışı; ileride modüler Python dosyalarında PEP 8 izlenecek.
* Branch/commit yaklaşımı: Ana çalışma `main` üzerinde ilk kurulum; anlamlı küçük commit'ler tercih edilmeli.
* İsimlendirme/klasör düzeni: Veri için `data/raw`, yönerge için `docs/assignment`, deney çıktıları için `reports`, kaynak kod için `src/banknote_mlp`.

## 5) Kurulum & Çalıştırma

* Gereksinimler: Python 3, `venv`, Jupyter Notebook.
* Komutlar:
  * `python3 -m venv .venv`
  * `source .venv/bin/activate`
  * `pip install -r requirements.txt`
  * `jupyter notebook notebooks/one_hidden_layer_mlp.ipynb`
* Ortam değişkenleri (sadece İSİMLER): Yok.
* Lokal geliştirme notları: Notebook veri yolunun repo içi klasör yapısına göre korunması gerekir.

## 6) Decision Log (append-only)

* 2026-03-25 — Karar: Proje adı `YZM304-Banknote-MLP` olarak kullanılacak. | Gerekçe: Kullanıcı tarafından açıkça belirtildi. | Etki: README başlığı, proje hafızası ve GitHub teslim dili bu adla hizalanacak. | Alternatifler: Yok.
* 2026-03-25 — Karar: Repo yapısı `data`, `docs`, `notebooks`, `reports`, `src`, `tests` ekseninde kurulacak. | Gerekçe: PDF yönergesi uygun klasör hiyerarşisi ve tekrar üretilebilirlik istiyor. | Etki: Mevcut veri, notebook ve yönerge dosyaları taşınacak; yeni modüller bu yapı altında geliştirilecek. | Alternatifler: Tüm dosyaları kökte tutmak.

## 7) Milestones / Dönüm Noktaları (append-only)

* 2026-03-25 — Milestone: PDF yönergesi analiz edildi. | Sonuç: Teslim için gereken klasör yapısı, README biçimi ve başlangıç hiperparametreleri netleştirildi.
* 2026-03-25 — Milestone: İlk repo iskeleti kuruldu ve GitHub'a pushlandı. | Sonuç: `data/`, `docs/`, `notebooks/`, `reports/`, `src/` ve `tests/` yapısı oluşturuldu; uzak repo ile `main` dalı eşitlendi.

## 8) Yapılanlar

* [x] PDF yönergesi metin olarak çıkarıldı ve gereksinimler belirlendi.
* [x] Mevcut veri seti ve laboratuvar notebook'u tespit edildi.
* [x] Dosya yapısı organize edildi.
* [x] Uzak repoya ilk push yapıldı.

## 9) Yapılacaklar (Next)

* [ ] Notebook içindeki temel MLP akışını modüler Python koduna taşı.
* [ ] Doğrulama seti, overfitting-underfitting analizi ve alternatif model deneyleri ekle.
* [ ] Karışıklık matrisi ve temel metrik çıktılarını `reports/` altında üret.
* [ ] `scikit-learn` veya `PyTorch` ile eşdeğer model karşılaştırmasını ekle.

## 10) Bilinen Sorunlar / Teknik Borç / Riskler

* Mevcut notebook doğrudan kök dizindeki dosya adına bağımlı; taşınmadan sonra veri yolu düzeltilmezse kırılır.
* Deney çıktıları ve alternatif model implementasyonları henüz eklenmedi.

## 11) Notlar ve Tuzaklar (Pitfalls)

* Notebook'taki veri okuma satırı `BankNote_Authentication.csv` ismini kullanıyor; gerçek dosya adı `banknote_authentication.csv`.
* Uzak GitHub deposu `https://github.com/akyilidizbaran/YZM304-LAB` adresine `main` dalı ile bağlandı.

### Güncelleme Kaydı

* Son güncelleme: 2026-03-25

# PROJECT_MEMORY

## 0) TL;DR (En güncel durum)

* Şu an ne yapıyoruz? `YZM304-Banknote-MLP` için PDF yönergesine uygun repo iskeleti kuruluyor ve GitHub'a hazırlanıyor.
* Son değişiklik neydi? Proje yönergesi incelendi; veri seti, laboratuvar notebook'u ve teslim beklentileri çıkarıldı.
* Bir sonraki net adım ne? Dosyaları `data/`, `notebooks/`, `docs/` ve `src/` altında düzenleyip uzak repoya ilk push'u yapmak.

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

## 8) Yapılanlar

* [x] PDF yönergesi metin olarak çıkarıldı ve gereksinimler belirlendi.
* [x] Mevcut veri seti ve laboratuvar notebook'u tespit edildi.
* [ ] Dosya yapısı organize edildi.
* [ ] Uzak repoya ilk push yapıldı.

## 9) Yapılacaklar (Next)

* [ ] Mevcut CSV dosyasını `data/raw/` altına taşı.
* [ ] Notebook'u `notebooks/` altına taşı ve veri yolunu güncelle.
* [ ] PDF yönergesini `docs/assignment/` altında sakla.
* [ ] Kaynak kod için `src/banknote_mlp/` başlangıç modüllerini ekle.
* [ ] Git deposunu başlat, commit oluştur ve `https://github.com/akyilidizbaran/YZM304-LAB` adresine push et.

## 10) Bilinen Sorunlar / Teknik Borç / Riskler

* Mevcut notebook doğrudan kök dizindeki dosya adına bağımlı; taşınmadan sonra veri yolu düzeltilmezse kırılır.
* Deney çıktıları ve alternatif model implementasyonları henüz eklenmedi.

## 11) Notlar ve Tuzaklar (Pitfalls)

* Notebook'taki veri okuma satırı `BankNote_Authentication.csv` ismini kullanıyor; gerçek dosya adı `banknote_authentication.csv`.
* Uzak GitHub deposu şu an boş görünüyor; ilk push doğrudan bu çalışma dizininden yapılabilir.

### Güncelleme Kaydı

* Son güncelleme: 2026-03-25


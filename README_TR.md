# Scrubber Design Simulator — Streamlit

Bu klasör, scrubber hesap motorunu web tabanlı bir arayüzde çalıştırmak için hazırlanmıştır.

## Dosyalar

- `app.py`: Streamlit kullanıcı arayüzü
- `scrubber_model.py`: mühendislik hesap motoru
- `requirements.txt`: gerekli Python paketleri
- `setup_and_run.bat`: Windows'ta ilk kurulum + çalıştırma
- `run_app.bat`: kurulum tamamlandıktan sonraki hızlı çalıştırma

## En kolay Windows kurulumu

1. Python 3.11 veya 3.12 kur.
2. Bu klasörü bilgisayarına çıkar.
3. `setup_and_run.bat` dosyasına çift tıkla.
4. Komut penceresi sanal ortam oluşturur, gerekli paketleri kurar ve Streamlit'i başlatır.
5. Tarayıcı otomatik açılmazsa terminalde yazan `http://localhost:8501` adresini tarayıcıda aç.

## Manuel kurulum

Komut İstemi veya PowerShell'i bu klasörde aç:

```bash
python -m venv .venv
```

Windows CMD:

```bash
.venv\Scripts\activate
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Paketleri kur:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Uygulamayı başlat:

```bash
python -m streamlit run app.py
```

Durdurmak için terminalde `Ctrl+C`.

## Not

Model, konsantrasyon girişini şu anda `mg VOC/Nm³` olarak yorumlar. Tesis TOC analizörü `mgC/Nm³` veya `ppmv` raporluyorsa önce uygun dönüşüm yapılmalıdır.

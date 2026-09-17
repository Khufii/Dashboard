# 🥚 Deteksi Kesegaran Kuning Telur Puyuh

Aplikasi GUI (Streamlit) — bagian dari produk luaran skripsi *"Klasifikasi
dan Estimasi Umur Simpan Kuning Telur Puyuh Menggunakan EfficientNet-XGBoost
dengan Bayesian Optimization"*.

Fitur:
- Unggah citra kuning telur puyuh
- Prediksi kelas kualitas: **Segar / Sedang / Menurun**
- Estimasi lama telur sudah disimpan & perkiraan sisa umur simpan (hari)

---

## 1. Jalankan di komputer lokal

```bash
git clone https://github.com/<username-kamu>/<nama-repo>.git
cd <nama-repo>
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

> Tanpa file model di folder `models/`, aplikasi tetap bisa dijalankan
> dalam **mode DEMO** (hasil acak, untuk pratinjau tampilan saja).
> Lihat `models/README.md` untuk cara memasang model asli hasil training
> (`pipeline.py`).

---

## 2. Push ke GitHub

```bash
git init
git add .
git commit -m "Inisialisasi aplikasi deteksi kesegaran kuning telur puyuh"
git branch -M main
git remote add origin https://github.com/<username-kamu>/<nama-repo>.git
git push -u origin main
```

Jika file model (`.h5` / `.json`) berukuran besar (>25 MB), gunakan
[Git LFS](https://git-lfs.com/) agar tidak melebihi batas GitHub:

```bash
git lfs install
git lfs track "models/*.h5" "models/*.json"
git add .gitattributes
```

---

## 3. Deploy ke Streamlit Community Cloud (integrasi GitHub)

1. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
2. Klik **"New app"**.
3. Pilih repository, branch (`main`), dan file utama: `app.py`.
4. Klik **Deploy**.

Setelah itu, setiap kali kamu `git push` perubahan ke branch `main`,
Streamlit Cloud akan otomatis re-deploy aplikasi (CI/CD sederhana bawaan
Streamlit Cloud, tidak perlu setup GitHub Actions tambahan).

---

## 4. Struktur folder

```
.
├── app.py              # tampilan utama (GUI)
├── utils.py             # pemuatan model & fungsi prediksi
├── requirements.txt
├── models/
│   ├── README.md
│   ├── efficientnet_feature_extractor.h5   (ditambahkan setelah training)
│   ├── xgb_classifier.json                  (ditambahkan setelah training)
│   └── xgb_regressor.json                   (ditambahkan setelah training)
└── README.md
```

## 5. Menyesuaikan estimasi umur simpan

Ambang umur simpan maksimum (hari) per kondisi penyimpanan diatur di
`utils.py` pada variabel `MAX_SHELF_LIFE_DAYS`. Sesuaikan dengan temuan
penelitian kamu (hari rata-rata sampel mencapai kelas "Menurun" pada
masing-masing kondisi suhu ruang / refrigerator).

# UTS Praktikum Pembelajaran Mesin
## Implementasi Klasifikasi Decision Tree dengan Resampling pada Data Bank Customer Churn

Aplikasi web (Flask) yang **menampilkan presentasi seluruh tahapan pipeline machine learning**:
dataset, preprocessing, transformation, split data, resampling (SMOTE & RUS), klasifikasi
Decision Tree, hingga evaluasi — lengkap dengan penjelasan *apa, untuk apa, dan kenapa*
untuk setiap tahap, serta visualisasi dari **output aktual program**.

**Kelompok — D4 Teknik Informatika, Universitas Airlangga:**
1. M. Arviansyah Desta Andini — 434241003
2. Marsha Adhia Camilla — 434241005
3. Robithoh Ahmad — 434241014

---

## 1. Persiapan (Sekali Saja)

### Prasyarat
- **Python 3.10 atau lebih baru** — cek dengan perintah:
  ```
  python --version
  ```

### Install library yang dibutuhkan
Buka terminal / command prompt di folder ini, lalu jalankan:
```
pip install -r requirements.txt
```
Perintah ini akan menginstal: Flask, pandas, numpy, scikit-learn, imbalanced-learn, dan matplotlib.

> Alternatif manual:
> ```
> pip install flask pandas numpy scikit-learn imbalanced-learn matplotlib
> ```

---

## 2. Menjalankan Aplikasi

```
python app.py
```

Tunggu hingga muncul di terminal:
```
* Running on http://127.0.0.1:5000
```

Lalu buka browser (Chrome/Edge/Firefox) dan akses:

### **http://localhost:5000**

> Matikan aplikasi dengan menekan `Ctrl + C` di terminal.

---

## 3. Halaman yang Ditampilkan

| Bagian | Isi |
|---|---|
| **Hero** | Ringkasan proyek + anggota kelompok |
| **Dataset** | 12 variabel (fitur & target), distribusi kelas churn, churn rate per negara/gender |
| **Tahapan** | Penjelasan 6 tahap: Preprocessing → Split 80:20 → Transformation → Resampling (SMOTE & RUS) → Decision Tree → Evaluasi |
| **Evaluasi** | Confusion Matrix, accuracy, precision, recall, F1-Score, dan kurva ROC SMOTE vs RUS |

Semua angka dan grafik dihitung **langsung dari output program** saat aplikasi dijalankan —
bukan angka yang diketik manual.

---

## 4. Struktur File

```
├── app.py                                  # Aplikasi Flask — jalankan FILE INI
├── preprocessing.py                        # Tahap 1: cek missing/duplikat/outlier (Arviansyah)
├── transformation.py                       # Tahap 2-3: split 80:20 + One-Hot Encoding (Arviansyah)
├── resampling.py                           # Tahap 4: SMOTE & RUS (Marsha)
├── model.py                                # Tahap 5: training Decision Tree (Marsha)
├── evaluation.py                           # Tahap 6: evaluasi + kurva ROC (Robithoh)
│
├── Bank Customer Churn Prediction.csv      # Dataset (10.000 baris)
├── requirements.txt                        # Daftar library yang perlu di-install
├── README.md                               # Panduan ini
│
├── confusion_matrix_smote.png              # Output evaluasi (dibuat evaluation.py)
├── confusion_matrix_rus.png                # Output evaluasi (dibuat evaluation.py)
├── hasil_evaluasi.csv                      # Output evaluasi (dibuat evaluation.py)
│
└── static/                                 # Seluruh tampilan web (satu folder utuh)
    ├── index.html                          # Halaman presentasi
    ├── css/style.css                       # Desain (tema bank biru)
    ├── js/app.js                           # Animasi & interaksi
    └── img/
        ├── confusion_matrix_smote.png      # Salinan untuk tampilan web
        ├── confusion_matrix_rus.png
        └── roc_curve.png                   # Kurva ROC (dibuat evaluation.py)
```

---

## 5. Pertanyaan Umum (FAQ)

**Q: `python app.py` error `No module named 'flask'` (atau module lain)?**
A: Library belum terinstal. Jalankan lagi `pip install -r requirements.txt`.

**Q: Error `Address already in use` / port 5000 dipakai?**
A: Ada aplikasi lain yang masih berjalan di port yang sama. Tutup instansi sebelumnya
(tutup terminalnya, atau `Ctrl + C`), lalu jalankan ulang.

**Q: Aplikasi berjalan lambat saat pertama dibuka?**
A: Normal. Saat pertama kali dijalankan, aplikasi memuat dataset, melakukan split,
resampling (SMOTE & RUS), dan melatih dua pohon keputusan. Proses ini hanya sekali
saat startup, setelahnya halaman tampil cepat.

**Q: Apakah butuh internet?**
A: Tidak wajib. Hanya font tampilan yang dimuat dari Google Fonts — jika offline,
font otomatis diganti dengan font bawaan sistem, isi halaman tetap utuh.

**Q: Bisakah gambar evaluasi dibuat ulang?**
A: Bisa. Jalankan `python evaluation.py` — ia menghitung ulang seluruh evaluasi dan
menyimpan ulang confusion matrix, `hasil_evaluasi.csv`, dan `static/img/roc_curve.png`.

**Q: Mau ganti port (misalnya 5000 dipakai aplikasi lain)?**
A: Ubah baris terakhir `app.py`:
```python
app.run(debug=True, port=8080)  # ganti 5000 -> 8080
```
lalu akses `http://localhost:8080`.

---

## 6. Catatan Penting

- **Urutan program** mengikuti ketentuan UTS: input → preprocessing → transformation →
  split → resampling (hanya pada data training) → Decision Tree → testing → evaluasi.
- `app.py` **tidak mengubah logika** file tahapan mana pun — ia hanya mengimpornya dan
  menampilkan hasilnya.
- Folder `static/` berdiri sendiri: bila diunduh dari Google Drive, `index.html` di
  dalamnya tetap bisa dibuka langsung di browser (tanpa Flask), hanya bagian visualisasi
  interaktif yang tidak aktif.

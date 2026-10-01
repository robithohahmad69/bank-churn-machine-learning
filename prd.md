PRD: Dashboard Evaluasi Decision Tree (Prediksi Churn Nasabah Bank)
Versi: 1.0 (draf) Penyusun: Robithoh (bagian Decision Tree + Evaluasi) Proyek: UTS Praktikum Pembelajaran Mesin, kelompok Arviansyah, Marsha, Robithoh Teknologi utama: Python, Flask, scikit-learn, imbalanced-learn

1. Latar Belakang
Kelompok sedang mengerjakan klasifikasi churn nasabah bank dengan dataset Bank Customer Churn Prediction (10.000 baris, 12 kolom, target churn). Alur pekerjaan sudah terbagi: preprocessing dan transformation oleh Arviansyah, resampling (SMOTE dan RandomUnderSampler) serta pelatihan Decision Tree oleh Marsha, dan evaluasi oleh Robithoh.

Saat ini hasil evaluasi hanya muncul di terminal melalui evaluasi.py. Hasil seperti itu sulit dibaca ulang, sulit dipresentasikan, dan tidak mudah dibandingkan sekilas. Dashboard web berbasis Flask dibuat agar hasil yang sama dapat dilihat dalam tampilan yang rapi dan interaktif, tanpa mengubah logika pemodelan yang sudah ada.

2. Tujuan
Tujuan utama produk ini adalah menampilkan seluruh evaluasi wajib dari panduan, yaitu Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score, dalam satu halaman web. Tujuan tambahannya adalah membandingkan dua skenario (SMOTE dan RUS) secara berdampingan, serta mempermudah presentasi dan penyusunan laporan.

Indikator keberhasilan: semua angka di web identik dengan output evaluasi.py pada run yang sama, dan seluruh halaman inti dapat dibuka tanpa error dari satu perintah python app.py.

3. Prinsip Data (dari panduan tugas)
Tiga aturan berikut tidak boleh dilanggar oleh aplikasi ini.

Data testing (X_test_final, y_test) tidak boleh di-resampling.
Model dilatih pada data training hasil resampling, lalu diuji pada data testing asli.
Semua angka evaluasi harus berasal dari hasil program aktual, tidak diketik manual dan tidak diperkirakan.
Konsekuensinya, aplikasi web tidak menyimpan angka tetap di HTML. Semua nilai dihitung dari y_test dan prediksi model saat aplikasi dijalankan.

4. Pengguna dan Skenario
Pengguna utama adalah anggota kelompok yang menyiapkan laporan dan presentasi, serta dosen atau asisten praktikum yang menilai hasil. Skenario utamanya: pengguna menjalankan aplikasi, membuka dashboard, membandingkan SMOTE dan RUS, lalu mengambil gambar atau angka untuk laporan.

5. Ruang Lingkup
Termasuk dalam versi 1.0

Dashboard ringkasan metrik
Halaman confusion matrix untuk SMOTE dan RUS
Halaman perbandingan SMOTE vs RUS
Halaman distribusi kelas (sebelum dan sesudah resampling)
Halaman tabel prediksi dengan filter prediksi salah
Opsional (dikerjakan bila waktu cukup)

Halaman feature importance
Halaman "Tentang" berisi alur proyek dan pembagian tugas
Di luar lingkup

Form prediksi churn untuk data nasabah baru (membutuhkan penyimpanan model dan encoder, dan tidak diminta panduan)
Pelatihan ulang model dari antarmuka web
Login, database, dan fitur multi-pengguna
Penerapan (deployment) ke server publik
6. Kebutuhan Fungsional
ID	Kebutuhan	Prioritas
F-01	Menampilkan Accuracy, Precision, Recall, F1-Score untuk SMOTE dan RUS dalam bentuk kartu.	Wajib
F-02	Menampilkan confusion matrix (TN, FP, FN, TP) untuk masing-masing metode, dengan label kelas yang jelas.	Wajib
F-03	Menampilkan tabel dan grafik batang perbandingan metrik SMOTE vs RUS.	Wajib
F-04	Menyatakan dengan jelas bahwa kelas positif adalah churn (1) dan metrik dihitung terhadap kelas tersebut.	Wajib
F-05	Menampilkan distribusi kelas training sebelum resampling, setelah SMOTE, dan setelah RUS, serta distribusi data testing.	Penting
F-06	Menampilkan tabel y_test, y_pred_smote, y_pred_rus dengan pagination dan filter "hanya prediksi salah".	Penting
F-07	Menyediakan navigasi menu yang konsisten di semua halaman.	Wajib
F-08	Menyediakan tombol unduh hasil_evaluasi.csv dan gambar confusion matrix untuk laporan.	Penting
F-09	Menampilkan feature importance dari masing-masing model.	Opsional
F-10	Menampilkan halaman alur proyek dan nama anggota kelompok.	Opsional
7. Rincian Halaman
Dashboard (/). Halaman pembuka berisi dua baris kartu metrik, satu baris untuk SMOTE dan satu baris untuk RUS, masing-masing memuat Accuracy, Precision, Recall, dan F1-Score. Di bawahnya ada catatan singkat tentang kelas positif, jumlah data testing, dan nama model (Decision Tree dengan random_state=42).

Confusion Matrix (/confusion-matrix). Dua heatmap berdampingan. Setiap kotak menampilkan angka dan namanya (TN, FP, FN, TP), ditambah satu paragraf penjelasan cara membacanya. Misalnya, FN adalah nasabah yang sebenarnya churn tetapi diprediksi tidak churn.

Perbandingan (/perbandingan). Tabel metrik SMOTE vs RUS dan grafik batang. Ada ruang catatan interpretasi yang diisi manual oleh kelompok setelah angka final diperoleh. Interpretasi tidak ditulis otomatis oleh aplikasi karena kesimpulan "mana yang lebih baik" bergantung pada tujuan bisnis.

Distribusi Data (/distribusi). Tabel dan grafik jumlah kelas 0 dan 1 pada training sebelum resampling, setelah SMOTE, setelah RUS, dan pada testing. Informasi ini membuktikan bahwa resampling hanya diterapkan pada data training.

Tabel Prediksi (/prediksi). Tabel berhalaman berisi label aktual dan prediksi dua model. Filter "hanya prediksi salah" membantu menelusuri kasus FP dan FN.

8. Kebutuhan Non-Fungsional
Konsistensi: angka di web harus sama dengan output terminal pada run yang sama. Gunakan random_state yang sudah ada dan jangan mengubah parameter model tanpa kesepakatan kelompok.
Performa: model dilatih sekali saat aplikasi dinyalakan, lalu hasilnya disimpan di memori. Halaman tidak boleh melatih ulang model pada setiap permintaan.
Kemudahan jalan: satu perintah (python app.py) cukup untuk menjalankan aplikasi di komputer lokal.
Keterbacaan: tampilan bersih dan responsif dengan Bootstrap, sehingga tidak bergantung pada kemampuan desain manual.
Bahasa: seluruh teks antarmuka dalam Bahasa Indonesia.
9. Arsitektur dan Alur Data
Bank Customer Churn Prediction.csv
        ↓
transformation.py  → X_train_final, X_test_final, y_train, y_test
        ↓
resampling.py      → X_train_smote / X_train_rus
        ↓
model.py           → model_smote, model_rus, y_pred_smote, y_pred_rus
        ↓
evaluasi_service.py (baru) → menghitung metrik dan confusion matrix
        ↓
app.py (Flask)     → rute → template HTML (Jinja2) → browser
Logika perhitungan metrik dipisahkan ke satu modul (evaluasi_service.py) yang dipakai bersama oleh evaluasi.py dan app.py, supaya tidak ada dua versi rumus yang bisa berbeda hasil. File milik Arviansyah dan Marsha tidak diubah; aplikasi hanya meng-import variabel dari model.py dan resampling.py.

Catatan teknis: karena model.py melatih model saat di-import, aplikasi cukup meng-import sekali di bagian atas app.py. Bila pelatihan terasa lambat saat debug, Flask mode debug dengan reloader dapat mengimpor ulang modul. Gunakan use_reloader=False bila itu mengganggu.

10. Usulan Struktur Folder
proyek/
├── Bank Customer Churn Prediction.csv
├── preprocessing.py        (Arviansyah)
├── transformation.py       (Arviansyah)
├── resampling.py           (Marsha)
├── model.py                (Marsha)
├── evaluasi.py             (Robithoh, versi terminal)
├── evaluasi_service.py     (Robithoh, logika metrik bersama)
├── app.py                  (Robithoh, aplikasi Flask)
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── confusion_matrix.html
│   ├── perbandingan.html
│   ├── distribusi.html
│   └── prediksi.html
└── static/
    ├── css/
    └── img/
Seluruh file Python harus berada dalam satu folder karena resampling.py, model.py, dan transformation.py saling meng-import dengan nama modul biasa.

11. Daftar Rute
Rute	Fungsi	Halaman
/	Ringkasan metrik	dashboard
/confusion-matrix	Heatmap SMOTE dan RUS	confusion matrix
/perbandingan	Tabel dan grafik perbandingan	perbandingan
/distribusi	Distribusi kelas	distribusi
/prediksi	Tabel prediksi dengan filter	prediksi
/unduh/hasil.csv	Unduh hasil evaluasi	file CSV
12. Kriteria Penerimaan
python app.py berjalan tanpa error dan halaman / terbuka di browser.
Accuracy, Precision, Recall, F1-Score, dan keempat angka confusion matrix di web sama persis dengan output evaluasi.py.
Data testing tidak mengalami resampling, dan halaman distribusi menunjukkan hal ini.
Semua menu pada navigasi dapat dibuka dan tidak menghasilkan halaman kosong.
Tidak ada angka evaluasi yang tertulis permanen di file HTML.
13. Risiko dan Mitigasi
Risiko	Dampak	Mitigasi
Nama file CSV tidak cocok (Bank_Customer... vs Bank Customer...).	Aplikasi gagal membaca data.	Samakan nama file dengan yang dibaca transformation.py, atau sepakati perubahan dengan Arviansyah.
Versi library di laptop anggota berbeda.	Angka bisa sedikit berbeda antar komputer.	Catat versi di requirements.txt dan gunakan hasil dari satu komputer untuk laporan.
Hasil bergantung pada satu kali split (random_state=42).	Kesimpulan mungkin tidak stabil.	Cantumkan keterbatasan ini di laporan.
Panduan menulis "SMOTE + RUS", sedangkan kode memisahkannya.	Interpretasi dosen bisa berbeda.	Sudah diputuskan memakai versi Marsha; konfirmasi ke dosen bila ragu.
Pelatihan model saat import memperlambat start-up.	Aplikasi terasa lambat dinyalakan.	Latih sekali di awal, simpan hasil di memori.
14. Rencana Tahapan Pengerjaan
Tahap 1, fondasi: buat evaluasi_service.py dan pastikan angkanya sama dengan evaluasi.py. Buat app.py dengan satu rute / dan base.html.
Tahap 2, halaman inti: dashboard, confusion matrix, dan perbandingan.
Tahap 3, halaman pendukung: distribusi data dan tabel prediksi.
Tahap 4, penyempurnaan: tombol unduh, tampilan Bootstrap yang rapi, dan (opsional) feature importance serta halaman Tentang.
Tahap 5, verifikasi: cocokkan semua angka dengan terminal dan jalankan seluruh kriteria penerimaan.
15. Asumsi dan Pertanyaan Terbuka
Dosen menerima tampilan web sebagai pelengkap, dengan output terminal tetap tersedia.
Aplikasi hanya dijalankan secara lokal (belum ada kebutuhan hosting).
Apakah kelompok ingin menambahkan halaman feature importance atau form prediksi? Keputusan ini belum ada.
Versi Flask, scikit-learn, dan imbalanced-learn yang dipakai kelompok belum diketahui. Periksa dengan pip list, dan lihat dokumentasi resmi masing-masing pustaka bila ada perbedaan perilaku.
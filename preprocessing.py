# =============================================================================
# preprocessing.py
# UTS Praktikum Pembelajaran Mesin
# Kelompok:
#   1. M. Arviansyah Desta Andini  - 434241003
#   2. Marsha Adhia Camilla         - 434241005
#   3. Robithoh Ahmad               - 434241014
#
# Bagian: Preprocessing (dikerjakan oleh Arviansyah)
# =============================================================================

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# 1. MEMBACA DATASET
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Bank Customer Churn Prediction.csv")

print("=" * 60)
print("PREPROCESSING - Bank Customer Churn Prediction")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset berhasil dibaca: {DATASET_PATH}")

# ---------------------------------------------------------------------------
# 2. INFORMASI DASAR DATASET
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("INFORMASI DASAR DATASET")
print("=" * 60)
print(f"Jumlah baris : {df.shape[0]}")
print(f"Jumlah kolom : {df.shape[1]}")
print("\nNama kolom:")
for col in df.columns:
    print(f"  - {col}")
print("\nTipe data setiap kolom:")
print(df.dtypes.to_string())

# ---------------------------------------------------------------------------
# 3. PENGECEKAN MISSING VALUE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PENGECEKAN MISSING VALUE")
print("=" * 60)
missing = df.isnull().sum()
print(missing.to_string())
total_missing = int(missing.sum())
print(f"\nTotal missing value: {total_missing}")
if total_missing == 0:
    print("-> Tidak ada missing value. Tidak perlu penanganan.")
else:
    print("-> Ditemukan missing value. Perlu penanganan.")

# ---------------------------------------------------------------------------
# 4. PENGECEKAN DUPLIKASI
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PENGECEKAN DUPLIKASI")
print("=" * 60)
jumlah_duplikasi = int(df.duplicated().sum())
print(f"Jumlah baris duplikasi: {jumlah_duplikasi}")
if jumlah_duplikasi == 0:
    print("-> Tidak ada duplikasi data. Tidak perlu penanganan.")
else:
    print(f"-> Ditemukan {jumlah_duplikasi} duplikasi. Menghapus...")
    df = df.drop_duplicates()
    print(f"   Baris setelah penghapusan: {df.shape[0]}")

# ---------------------------------------------------------------------------
# 5. PENGECEKAN OUTLIER (Metode IQR)
# ---------------------------------------------------------------------------
# Kolom numerik kontinu yang dicek outlier.
# Dikecualikan: customer_id (identifier), credit_card & active_member (biner 0/1),
# products_number (diskrit 1-4), churn (target).
print("\n" + "=" * 60)
print("PENGECEKAN OUTLIER (Metode IQR)")
print("=" * 60)

KOLOM_CEK_OUTLIER = ["credit_score", "age", "tenure", "balance", "estimated_salary"]

def cek_outlier_iqr(dataframe, kolom):
    """Menghitung jumlah outlier menggunakan metode IQR."""
    Q1 = dataframe[kolom].quantile(0.25)
    Q3 = dataframe[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas  = Q3 + 1.5 * IQR
    jumlah = int(((dataframe[kolom] < batas_bawah) | (dataframe[kolom] > batas_atas)).sum())
    return jumlah, batas_bawah, batas_atas

total_outlier = 0
print("Kolom                Batas Bawah   Batas Atas   Jml Outlier")
print("-" * 62)
for kol in KOLOM_CEK_OUTLIER:
    jml, bb, ba = cek_outlier_iqr(df, kol)
    total_outlier += jml
    print(f"{kol:<20} {bb:>12.2f}  {ba:>12.2f}  {jml:>10}")

print(f"\nTotal outlier terdeteksi: {total_outlier} baris")

# Catatan penanganan outlier:
# - Outlier pada 'age' (usia 63-92) adalah nasabah lansia yang valid secara domain.
# - Outlier pada 'credit_score' (<383) adalah skor kredit rendah, tetap data valid.
# - Decision Tree tidak sensitif terhadap outlier maupun skala fitur.
# - Menghapus outlier berisiko mengurangi informasi penting, sehingga TIDAK dihapus.
print("\nCatatan:")
print("  Outlier yang ditemukan adalah nilai valid secara domain bisnis perbankan.")
print("  Decision Tree tidak sensitif terhadap outlier.")
print("  Outlier TIDAK dihapus agar data tetap lengkap.")

# Distribusi products_number (kolom diskrit, dicek terpisah)
print("\n--- Distribusi products_number (kolom diskrit, nilai 1-4) ---")
print(df["products_number"].value_counts().sort_index().to_string())
print("  Nilai 1-4 valid dalam domain perbankan.")

# ---------------------------------------------------------------------------
# 6. DISTRIBUSI TARGET
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DISTRIBUSI TARGET (churn)")
print("=" * 60)
dist_target = df["churn"].value_counts()
print(dist_target.to_string())
total = df.shape[0]
pct0 = round(dist_target[0] / total * 100, 1)
pct1 = round(dist_target[1] / total * 100, 1)
print(f"\n  churn=0 (tidak churn): {dist_target[0]} ({pct0}%)")
print(f"  churn=1 (churn)      : {dist_target[1]} ({pct1}%)")
print("  -> Dataset tidak seimbang. Akan ditangani di tahap resampling (Marsha).")

# ---------------------------------------------------------------------------
# 7. MEMISAHKAN FITUR (X) DAN TARGET (y)
# ---------------------------------------------------------------------------
# customer_id adalah identifier unik nasabah, BUKAN fitur prediktif.
# Tidak diikutsertakan dalam pemodelan.
print("\n" + "=" * 60)
print("MEMISAHKAN FITUR (X) DAN TARGET (y)")
print("=" * 60)

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print(f"Fitur (X) - {X.shape[1]} kolom:")
print("  " + str(X.columns.tolist()))
print(f"\nTarget (y): kolom 'churn'")
print(f"  Shape X : {X.shape}")
print(f"  Shape y : {y.shape}")

# ---------------------------------------------------------------------------
# 8. RINGKASAN HASIL PREPROCESSING
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RINGKASAN HASIL PREPROCESSING")
print("=" * 60)
print(f"  Missing value      : {total_missing} (tidak ada)")
print(f"  Duplikasi          : {jumlah_duplikasi} (tidak ada)")
print(f"  Outlier terdeteksi : {total_outlier} baris (tidak dihapus, valid secara domain)")
print(f"  Total data akhir   : {df.shape[0]} baris")
print(f"  Fitur untuk model  : {X.shape[1]} kolom (tanpa customer_id dan churn)")
print()
print("Preprocessing selesai. X dan y siap diteruskan ke transformation.py")
print("=" * 60)

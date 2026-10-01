# =============================================================================
# transformation.py
# UTS Praktikum Pembelajaran Mesin
# Kelompok:
#   1. M. Arviansyah Desta Andini  - 434241003
#   2. Marsha Adhia Camilla         - 434241005
#   3. Robithoh Ahmad               - 434241014
#
# Bagian: Split Data + Transformation (dikerjakan oleh Arviansyah)
#
# Alur:
#   Preprocessing (preprocessing.py)
#        -> Split 80% Training / 20% Testing   <- di file ini
#        -> Transformation (One-Hot Encoding)  <- di file ini
#        -> Resampling SMOTE + RUS             <- Marsha
#        -> Decision Tree + Evaluasi           <- Robithoh
#
# Catatan mengapa split diletakkan di sini:
#   Transformation harus di-fit HANYA pada data training.
#   Oleh karena itu split harus dilakukan SEBELUM transformation.
#   Menempatkan keduanya dalam satu file membuat alur lebih jelas dan aman.
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------------------------
# LANGKAH 1: Jalankan ulang preprocessing untuk mendapatkan X dan y
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Bank Customer Churn Prediction.csv")

print("=" * 60)
print("TRANSFORMATION - Bank Customer Churn Prediction")
print("=" * 60)

# Membaca dataset
df = pd.read_csv(DATASET_PATH)

# customer_id adalah identifier, bukan fitur prediktif
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print(f"Data dimuat: {X.shape[0]} baris, {X.shape[1]} fitur")

# ---------------------------------------------------------------------------
# LANGKAH 2: SPLIT DATA 80% TRAINING / 20% TESTING
# ---------------------------------------------------------------------------
# stratify=y memastikan proporsi kelas churn tetap terjaga di training dan testing.
# random_state=42 digunakan agar hasil split dapat direproduksi.
print("\n" + "=" * 60)
print("SPLIT DATA 80% TRAINING / 20% TESTING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Total data       : {len(X)} baris")
print(f"Data training    : {len(X_train)} baris ({len(X_train)/len(X)*100:.1f}%)")
print(f"Data testing     : {len(X_test)} baris ({len(X_test)/len(X)*100:.1f}%)")
print()
print("Distribusi target pada training:")
print(y_train.value_counts().to_string())
print()
print("Distribusi target pada testing:")
print(y_test.value_counts().to_string())

# ---------------------------------------------------------------------------
# LANGKAH 3: IDENTIFIKASI KOLOM KATEGORIKAL DAN NUMERIK
# ---------------------------------------------------------------------------
# Diidentifikasi berdasarkan tipe data aktual dari dataset (bukan asumsi).
print("\n" + "=" * 60)
print("IDENTIFIKASI TIPE KOLOM")
print("=" * 60)

# Kolom dengan tipe object/string = kategorikal
kolom_kategorikal = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

# Kolom numerik = semua kolom selain kategorikal
kolom_numerik = X_train.select_dtypes(exclude=["object", "str"]).columns.tolist()

print(f"Kolom kategorikal ({len(kolom_kategorikal)} kolom): {kolom_kategorikal}")
print(f"Kolom numerik     ({len(kolom_numerik)} kolom): {kolom_numerik}")

# ---------------------------------------------------------------------------
# LANGKAH 4: TRANSFORMATION - ONE-HOT ENCODING
# ---------------------------------------------------------------------------
# One-Hot Encoding digunakan untuk fitur kategorikal (country, gender).
# Fitur numerik dibiarkan apa adanya (passthrough) karena Decision Tree
# tidak memerlukan normalisasi/standardisasi.
#
# PENTING:
#   - Encoder di-FIT hanya menggunakan data TRAINING
#   - Data TESTING hanya di-TRANSFORM (tidak di-fit ulang)
#   Ini mencegah data leakage dari testing ke training.
print("\n" + "=" * 60)
print("TRANSFORMATION (One-Hot Encoding)")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
         kolom_kategorikal)
    ],
    remainder="passthrough"  # kolom numerik diteruskan tanpa perubahan
)

# FIT menggunakan data training SAJA
preprocessor.fit(X_train)
print("Encoder di-fit menggunakan data training.")

# TRANSFORM data training
X_train_transformed = preprocessor.transform(X_train)

# TRANSFORM data testing (hanya transform, tidak fit ulang)
X_test_transformed = preprocessor.transform(X_test)

print("Data training dan testing berhasil ditransformasi.")

# ---------------------------------------------------------------------------
# LANGKAH 5: MENDAPATKAN NAMA FITUR HASIL ENCODING
# ---------------------------------------------------------------------------
# Nama fitur hasil OHE penting agar tahap berikutnya (Robithoh) mudah
# memahami struktur data yang diterima.
ohe_feature_names = preprocessor.named_transformers_["ohe"].get_feature_names_out(kolom_kategorikal).tolist()
all_feature_names = ohe_feature_names + kolom_numerik

print(f"\nFitur hasil transformation ({len(all_feature_names)} kolom):")
for nama in all_feature_names:
    print(f"  - {nama}")

# Konversi ke DataFrame agar lebih mudah digunakan oleh tahap berikutnya
X_train_final = pd.DataFrame(X_train_transformed, columns=all_feature_names)
X_test_final  = pd.DataFrame(X_test_transformed,  columns=all_feature_names)

# Reset index agar konsisten
X_train_final = X_train_final.reset_index(drop=True)
X_test_final  = X_test_final.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# ---------------------------------------------------------------------------
# LANGKAH 6: VERIFIKASI KONSISTENSI TRAINING DAN TESTING
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFIKASI KONSISTENSI TRAINING DAN TESTING")
print("=" * 60)
print(f"Shape X_train_final : {X_train_final.shape}")
print(f"Shape X_test_final  : {X_test_final.shape}")
print(f"Shape y_train       : {y_train.shape}")
print(f"Shape y_test        : {y_test.shape}")
print()

# Pastikan nama kolom sama persis antara training dan testing
kolom_sama = list(X_train_final.columns) == list(X_test_final.columns)
print(f"Nama kolom training == testing : {kolom_sama}")
print(f"Jumlah kolom training          : {X_train_final.shape[1]}")
print(f"Jumlah kolom testing           : {X_test_final.shape[1]}")

# ---------------------------------------------------------------------------
# LANGKAH 7: RINGKASAN HASIL TRANSFORMATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RINGKASAN HASIL TRANSFORMATION")
print("=" * 60)
print(f"  Split data          : 80% training ({len(X_train)} baris) / 20% testing ({len(X_test)} baris)")
print(f"  Stratify            : Ya (proporsi kelas dipertahankan)")
print(f"  Encoding            : One-Hot Encoding pada {kolom_kategorikal}")
print(f"  drop='first'        : Ya (menghindari dummy variable trap)")
print(f"  Numerik             : Diteruskan tanpa perubahan (passthrough)")
print(f"  Fitur sebelum OHE   : {X.shape[1]} kolom")
print(f"  Fitur setelah OHE   : {len(all_feature_names)} kolom")
print(f"  Konsistensi kolom   : {kolom_sama}")
print()
print("Transformation selesai.")
print("Output yang tersedia untuk tahap berikutnya:")
print("  X_train_final  -> fitur training (sudah di-encode)")
print("  X_test_final   -> fitur testing (sudah di-encode)")
print("  y_train        -> label training")
print("  y_test         -> label testing")
print("  all_feature_names -> nama kolom hasil transformation")
print()
print("Catatan untuk Marsha (Resampling):")
print("  Gunakan X_train_final dan y_train sebagai input SMOTE + RUS.")
print("  JANGAN lakukan resampling pada X_test_final / y_test.")
print()
print("Catatan untuk Robithoh (Decision Tree + Evaluasi):")
print("  Setelah resampling, gunakan data hasil resampling untuk training.")
print("  Gunakan X_test_final dan y_test untuk testing dan evaluasi.")
print("=" * 60)

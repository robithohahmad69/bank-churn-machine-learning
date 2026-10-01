# =============================================================================
# app.py — Antarmuka presentasi UTS Praktikum Pembelajaran Mesin
# Implementasi Klasifikasi Decision Tree dengan Resampling
# pada Data Bank Customer Churn Prediction
#
# Menjelaskan dataset (variabel & target) dan setiap tahap pipeline:
# apa, untuk apa, dan kenapa — dengan angka & VISUALISASI dari OUTPUT
# AKTUAL program, bukan angka yang diketik manual.
#
# Catatan penting:
#   File tahapan (preprocessing.py, transformation.py, resampling.py,
#   model.py, evaluation.py) TIDAK diubah sama sekali. Semua visualisasi
#   di sini dihitung dari objek yang sudah dihasilkan pipeline (preprocessor,
#   y_train_smote, model_smote.tree_, y_test, dst.) atau dihitung ulang
#   dari dataset mentah dengan logika yang sama persis.
# =============================================================================

import io
import os
import contextlib

import pandas as pd
from flask import Flask, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Muat pipeline (preprocessing -> transformation -> resampling -> model).
# Import model.py lebih dulu: ia me-chain resampling & transformation dengan
# stdout yang di-redirect, sehingga modul sudah ter-cache tanpa mem-print.
# ---------------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()):
    import model  # noqa: F401  (menjalankan seluruh pipeline)
    from model import model_smote, model_rus
    from transformation import (
        preprocessor,
        all_feature_names,
        X_train_final,
        X_test_final,
        y_train,
    )
    from resampling import y_train_smote, y_train_rus, format_distribusi

# Template + css/js/img digabung dalam SATU folder "static"
# agar mudah di-upload ke Google Drive sebagai satu paket.
# static_url_path="" membuat css/js/img dilayani dari root ("/css/..."),
# sehingga path relatif di index.html berfungsi baik via Flask maupun
# saat file dibuka langsung dari folder hasil download Drive.
app = Flask(
    __name__,
    template_folder="static",
    static_folder="static",
    static_url_path="",
)

# ---------------------------------------------------------------------------
# Statistik dataset — dihitung dari output aktual
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "Bank Customer Churn Prediction.csv"))

churn_counts = df["churn"].value_counts()
DATASET_STATS = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "features": len(all_feature_names),
    "n_tetap": int(churn_counts.get(0, 0)),
    "n_churn": int(churn_counts.get(1, 0)),
    "pct_tetap": round(churn_counts.get(0, 0) / df.shape[0] * 100, 2),
    "pct_churn": round(churn_counts.get(1, 0) / df.shape[0] * 100, 2),
}

# ---------------------------------------------------------------------------
# VISUALISASI 1 — IQR outlier per kolom (logika sama dengan preprocessing.py)
# ---------------------------------------------------------------------------
KOLOM_CEK_OUTLIER = ["credit_score", "age", "tenure", "balance", "estimated_salary"]
iqr_rows = []
for kol in KOLOM_CEK_OUTLIER:
    Q1 = df[kol].quantile(0.25)
    Q3 = df[kol].quantile(0.75)
    IQR = Q3 - Q1
    bb, ba = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n = int(((df[kol] < bb) | (df[kol] > ba)).sum())
    iqr_rows.append({
        "kolom": kol,
        "bawah": round(float(bb), 1),
        "atas": round(float(ba), 1),
        "outlier": n,
    })
IQR_MAX = max(r["outlier"] for r in iqr_rows) or 1

# ---------------------------------------------------------------------------
# VISUALISASI 2 — churn rate per negara & gender (EDA dari dataset)
# ---------------------------------------------------------------------------
def churn_rate_group(kolom):
    g = df.groupby(kolom)["churn"].agg(["mean", "size"]).reset_index()
    g = g.sort_values("mean", ascending=False)
    return [
        {"label": str(r[kolom]), "rate": round(float(r["mean"]) * 100, 1), "n": int(r["size"])}
        for _, r in g.iterrows()
    ]

CHURN_BY_COUNTRY = churn_rate_group("country")
CHURN_BY_GENDER = churn_rate_group("gender")
CHURN_RATE_MAX = max(
    [r["rate"] for r in CHURN_BY_COUNTRY] + [r["rate"] for r in CHURN_BY_GENDER]
)

# ---------------------------------------------------------------------------
# Struktur variabel dataset (sesuai template laporan / panduan UTS)
# ---------------------------------------------------------------------------
VARIABLES = [
    ("1",  "customer_id",       "Identifier",  "Nomor unik nasabah — bukan fitur prediktif, dikeluarkan dari pemodelan."),
    ("2",  "credit_score",      "Fitur",       "Skor kredit nasabah (numerik)."),
    ("3",  "country",           "Fitur kategorikal", "Negara domisili: France, Spain, Germany."),
    ("4",  "gender",            "Fitur kategorikal", "Jenis kelamin nasabah: Male / Female."),
    ("5",  "age",               "Fitur",       "Usia nasabah (numerik)."),
    ("6",  "tenure",            "Fitur",       "Lama menjadi nasabah, dalam tahun (0–10)."),
    ("7",  "balance",           "Fitur",       "Saldo rekening (numerik)."),
    ("8",  "products_number",   "Fitur",       "Jumlah produk bank yang dimiliki (1–4, diskrit)."),
    ("9",  "credit_card",       "Fitur",       "Memiliki kartu kredit (biner 0/1)."),
    ("10", "active_member",     "Fitur",       "Status keanggotaan aktif (biner 0/1)."),
    ("11", "estimated_salary",  "Fitur",       "Estimasi gaji nasabah (numerik)."),
    ("12", "churn",             "Target",      "Kelas yang diprediksi: 1 = nasabah keluar (churn), 0 = bertahan."),
]

# ---------------------------------------------------------------------------
# Statistik split & resampling — dihitung dari output aktual pipeline
# ---------------------------------------------------------------------------
c0, p0, c1, p1, total = format_distribusi(y_train)
S0, SP0, S1, SP1, S_total = format_distribusi(y_train_smote)
R0, RP0, R1, RP1, R_total = format_distribusi(y_train_rus)

RESAMPLING_ROWS = [
    {"metode": "Training sebelum resampling", "k0": c0, "p0": f"{p0:.2f}", "k1": c1, "p1": f"{p1:.2f}", "total": total},
    {"metode": "SMOTE (oversampling)",        "k0": S0, "p0": f"{SP0:.2f}", "k1": S1, "p1": f"{SP1:.2f}", "total": S_total},
    {"metode": "RUS (undersampling)",         "k0": R0, "p0": f"{RP0:.2f}", "k1": R1, "p1": f"{RP1:.2f}", "total": R_total},
]

# VISUALISASI 3 — bar keseimbangan kelas sebelum/sesudah resampling
RESAMPLING_BARS = [
    {"label": "Sebelum", "p0": p0, "p1": p1},
    {"label": "SMOTE",   "p0": SP0, "p1": SP1},
    {"label": "RUS",     "p0": RP0, "p1": RP1},
]

SPLIT_ROWS = [
    {"pembagian": "Training", "jumlah": len(X_train_final), "persen": "80%"},
    {"pembagian": "Testing",  "jumlah": len(X_test_final),  "persen": "20%"},
]

EVALUASI = pd.read_csv(os.path.join(BASE_DIR, "hasil_evaluasi.csv")).to_dict("records")

# ---------------------------------------------------------------------------
# VISUALISASI 4 — anatomi pohon keputusan (dari objek sklearn, tanpa retrain)
# ---------------------------------------------------------------------------
def tree_anatomi(tree_model, feature_names):
    t = tree_model.tree_
    root_feature = feature_names[int(t.feature[0])]
    root_threshold = float(t.threshold[0])
    return {
        "depth": int(t.max_depth),
        "nodes": int(t.node_count),
        "leaves": int(t.n_leaves),
        "root_fitur": root_feature,
        "root_ambang": round(root_threshold, 2),
    }

ANATOMI_SMOTE = tree_anatomi(model_smote, all_feature_names)
ANATOMI_RUS = tree_anatomi(model_rus, all_feature_names)

# Feature importance dari model SMOTE (pohon yang sudah dilatih)
importances = sorted(
    zip(all_feature_names, model_smote.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)
FEATURE_IMPORTANCE = [
    {"nama": n, "nilai": round(float(v), 4)} for n, v in importances
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        stats=DATASET_STATS,
        variables=VARIABLES,
        iqr_rows=iqr_rows,
        iqr_max=IQR_MAX,
        churn_country=CHURN_BY_COUNTRY,
        churn_gender=CHURN_BY_GENDER,
        churn_rate_max=CHURN_RATE_MAX,
        split_rows=SPLIT_ROWS,
        resampling_rows=RESAMPLING_ROWS,
        resampling_bars=RESAMPLING_BARS,
        anatomi_smote=ANATOMI_SMOTE,
        anatomi_rus=ANATOMI_RUS,
        importance=FEATURE_IMPORTANCE,
        evaluasi=EVALUASI,
    )


if __name__ == "__main__":
    print("Menjalankan aplikasi di http://localhost:5000")
    app.run(debug=True, port=5000)

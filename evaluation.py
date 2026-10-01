# =============================================================================
# evaluasi.py
# UTS Praktikum Pembelajaran Mesin
# Bagian: Decision Tree + Evaluasi (dikerjakan oleh Robithoh)
#
# Alur:
#   transformation.py -> resampling.py -> model.py -> evaluasi.py (file ini)
#
# File ini TIDAK melatih ulang model dan TIDAK melakukan resampling.
# Ia memakai y_test dan prediksi dari model.py (model dilatih pada data hasil
# resampling, diuji pada X_test_final asli tanpa resampling).
# Semua angka dihitung langsung dari hasil program, bukan diketik manual.
# =============================================================================

import io
import contextlib
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # simpan gambar ke file; ganti/hapus baris ini jika ingin plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import dari model.py (output print dari file lain disembunyikan)
with contextlib.redirect_stdout(io.StringIO()):
    from model import y_test, y_pred_smote, y_pred_rus

# Kelas positif = churn (1), karena fokus kasus ini adalah mendeteksi nasabah churn.
HASIL = {
    "SMOTE": y_pred_smote,
    "RUS": y_pred_rus,
}

ringkasan = []

for nama, y_pred in HASIL.items():
    print("=" * 60)
    print(f"EVALUASI DECISION TREE - {nama}")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("Confusion Matrix (baris = aktual, kolom = prediksi):")
    print(pd.DataFrame(
        cm,
        index=["Aktual 0 (tidak churn)", "Aktual 1 (churn)"],
        columns=["Prediksi 0", "Prediksi 1"],
    ).to_string())
    print(f"\nTN = {tn} | FP = {fp} | FN = {fn} | TP = {tp}")

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}  (kelas churn = 1)")
    print(f"Recall    : {rec:.4f}  (kelas churn = 1)")
    print(f"F1-Score  : {f1:.4f}  (kelas churn = 1)")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Tidak churn (0)", "Churn (1)"], digits=4,
    ))

    # Simpan gambar confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Tidak churn", "Churn"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"Confusion Matrix - Decision Tree ({nama})")
    plt.tight_layout()
    path_gambar = os.path.join(BASE_DIR, f"confusion_matrix_{nama.lower()}.png")
    fig.savefig(path_gambar, dpi=150)
    plt.close(fig)
    print(f"Gambar disimpan: {path_gambar}\n")

    ringkasan.append({
        "Metode": nama,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    })

print("=" * 60)
print("PERBANDINGAN SMOTE vs RUS (kelas churn = 1)")
print("=" * 60)
df_ringkasan = pd.DataFrame(ringkasan).set_index("Metode").round(4)
print(df_ringkasan.to_string())
df_ringkasan.to_csv(os.path.join(BASE_DIR, "hasil_evaluasi.csv"))
print("\nTabel disimpan ke hasil_evaluasi.csv")

# =============================================================================
# KURVA ROC — SMOTE vs RUS (kelas positif = churn / 1)
#
# ROC butuh PROBABILITAS prediksi (bukan label 0/1), sehingga model dan
# X_test_final di-import dari file tahapan — tidak ada perhitungan yang
# diubah, kurva murni dari output aktual model.
# Gambar disimpan ke static/img/roc_curve.png agar ikut dalam satu folder
# web (untuk upload Google Drive).
# =============================================================================
with contextlib.redirect_stdout(io.StringIO()):
    from model import model_smote, model_rus
    from transformation import X_test_final
from sklearn.metrics import roc_curve, auc

prob_smote = model_smote.predict_proba(X_test_final)[:, 1]
prob_rus = model_rus.predict_proba(X_test_final)[:, 1]

plt.figure(figsize=(6, 5))
for nama, y_score in [("SMOTE", prob_smote), ("RUS", prob_rus)]:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{nama} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Tebakan acak (AUC = 0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Kurva ROC — Kelas Positif Churn (1)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

IMG_DIR = os.path.join(BASE_DIR, "static", "img")
os.makedirs(IMG_DIR, exist_ok=True)
path_roc = os.path.join(IMG_DIR, "roc_curve.png")
plt.savefig(path_roc, dpi=150)
plt.close()
print(f"Gambar disimpan: {path_roc}")
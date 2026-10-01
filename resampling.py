import os
import sys
import io
import contextlib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

with contextlib.redirect_stdout(io.StringIO()):
    from transformation import (
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        all_feature_names,
    )

smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_final, y_train)

rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_final, y_train)


def format_distribusi(y):
    counts = y.value_counts().sort_index()
    c0, c1 = int(counts.get(0, 0)), int(counts.get(1, 0))
    total = c0 + c1
    p0 = (c0 / total * 100) if total > 0 else 0.0
    p1 = (c1 / total * 100) if total > 0 else 0.0
    return c0, p0, c1, p1, total


def tampilkan_ringkasan():
    print("Resampling\n")

    c0, p0, c1, p1, total = format_distribusi(y_train)
    print("Training sebelum resampling")
    print(f"Class 0: {c0} ({p0:.2f}%)")
    print(f"Class 1: {c1} ({p1:.2f}%)")
    print(f"Total: {total}\n")

    s0, sp0, s1, sp1, s_total = format_distribusi(y_train_smote)
    print("SMOTE")
    print(f"Class 0: {s0} ({sp0:.2f}%)")
    print(f"Class 1: {s1} ({sp1:.2f}%)")
    print(f"Total: {s_total}\n")

    r0, rp0, r1, rp1, r_total = format_distribusi(y_train_rus)
    print("RUS")
    print(f"Class 0: {r0} ({rp0:.2f}%)")
    print(f"Class 1: {r1} ({rp1:.2f}%)")
    print(f"Total: {r_total}")


if __name__ == "__main__":
    tampilkan_ringkasan()

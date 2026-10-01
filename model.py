import os
import sys
import io
import contextlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

with contextlib.redirect_stdout(io.StringIO()):
    from resampling import (
        X_train_smote,
        y_train_smote,
        X_train_rus,
        y_train_rus,
        X_test_final,
        y_test,
    )

model_smote = DecisionTreeClassifier(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test_final)

model_rus = DecisionTreeClassifier(random_state=42)
model_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = model_rus.predict(X_test_final)

df_prediksi = pd.DataFrame({
    "y_test": y_test,
    "y_pred_smote": y_pred_smote,
    "y_pred_rus": y_pred_rus
})


def tampilkan_ringkasan():
    print("Decision Tree\n")

    print("SMOTE")
    print("Training selesai")
    print(f"Jumlah prediksi: {len(y_pred_smote)}\n")

    print("RUS")
    print("Training selesai")
    print(f"Jumlah prediksi: {len(y_pred_rus)}")


if __name__ == "__main__":
    tampilkan_ringkasan()

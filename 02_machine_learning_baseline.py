# ============================================================
# STEP 2: MACHINE LEARNING BASELINE
# Project: AI-Powered Banking Customer Support
#
# Model:
#     TF-IDF + Logistic Regression
#
# Purpose:
#     Establish a traditional NLP baseline before
#     Transformer/BERT model development.
# ============================================================

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "train_clean.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "test_clean.csv"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_logistic_regression.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "ml_baseline_metrics.json"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "ml_classification_report.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "ml_confusion_matrix.csv"
)


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. START
# ============================================================

print("=" * 70)
print("STEP 2: MACHINE LEARNING BASELINE")
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)


# ============================================================
# 4. CHECK FILES
# ============================================================

print("\n[1/9] Checking input files...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_PATH}"
    )

if not os.path.exists(LABEL_MAPPING_PATH):
    raise FileNotFoundError(
        f"Label mapping not found:\n{LABEL_MAPPING_PATH}"
    )

print("Training dataset found.")
print("Test dataset found.")
print("Label mapping found.")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n[2/9] Loading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

with open(
    LABEL_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as file:
    label_mapping = json.load(file)

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Number of classes:", label_mapping["num_classes"])


# ============================================================
# 6. PREPARE TEXT AND LABELS
# ============================================================

print("\n[3/9] Preparing text and labels...")

X = train_df["text"].astype(str)
y = train_df["label"].astype(int)

X_test = test_df["text"].astype(str)
y_test = test_df["label"].astype(int)

print("Training samples:", len(X))
print("Official test samples:", len(X_test))
print("Number of classes:", y.nunique())


# ============================================================
# 7. STRATIFIED TRAIN/VALIDATION SPLIT
# ============================================================

print("\n[4/9] Creating stratified validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training split:", len(X_train))
print("Validation split:", len(X_val))
print("Official test set:", len(X_test))

print("\nThe official test set is being kept separate.")


# ============================================================
# 8. TF-IDF VECTORIZATION
# ============================================================

print("\n[5/9] Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_val_tfidf = vectorizer.transform(X_val)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature matrix:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)
print("Test:", X_test_tfidf.shape)

print(
    "Number of learned TF-IDF features:",
    len(vectorizer.get_feature_names_out())
)


# ============================================================
# 9. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n[6/9] Training Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    C=4.0,
    solver="saga",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Logistic Regression training completed.")


# ============================================================
# 10. VALIDATION PREDICTIONS
# ============================================================

print("\n[7/9] Evaluating on validation data...")

val_predictions = model.predict(
    X_val_tfidf
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_f1_weighted = f1_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_f1_macro = f1_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

print("\nVALIDATION RESULTS")
print("-" * 70)

print(f"Accuracy          : {val_accuracy:.4f}")
print(f"Weighted Precision: {val_precision:.4f}")
print(f"Weighted Recall   : {val_recall:.4f}")
print(f"Weighted F1       : {val_f1_weighted:.4f}")
print(f"Macro F1          : {val_f1_macro:.4f}")


# ============================================================
# 11. VALIDATION CLASSIFICATION REPORT
# ============================================================

class_names = [
    label_mapping["id2label"][str(i)]
    for i in range(label_mapping["num_classes"])
]

classification_report_text = classification_report(
    y_val,
    val_predictions,
    labels=list(range(label_mapping["num_classes"])),
    target_names=class_names,
    zero_division=0
)

print("\nVALIDATION CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report_text)


# ============================================================
# 12. VALIDATION CONFUSION MATRIX
# ============================================================

val_confusion_matrix = confusion_matrix(
    y_val,
    val_predictions,
    labels=list(range(label_mapping["num_classes"]))
)

confusion_df = pd.DataFrame(
    val_confusion_matrix,
    index=class_names,
    columns=class_names
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\n[8/9] Saving validation results...")

metrics = {
    "model": "TF-IDF + Logistic Regression",

    "dataset": {
        "total_training_rows": int(len(train_df)),
        "train_split_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "official_test_rows": int(len(X_test)),
        "num_classes": int(y.nunique())
    },

    "tfidf": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
        "max_features": 50000,
        "num_features": int(
            len(vectorizer.get_feature_names_out())
        )
    },

    "validation_metrics": {
        "accuracy": float(val_accuracy),
        "weighted_precision": float(val_precision),
        "weighted_recall": float(val_recall),
        "weighted_f1": float(val_f1_weighted),
        "macro_f1": float(val_f1_macro)
    }
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-Powered Banking Customer Support\n"
    )

    file.write(
        "TF-IDF + Logistic Regression Baseline\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        classification_report_text
    )


confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# 14. SAVE MODEL AND VECTORIZER
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print("Model saved:", MODEL_PATH)
print("Vectorizer saved:", VECTORIZER_PATH)


# ============================================================
# 15. TEST SET IS NOT USED FOR TRAINING
# ============================================================

print("\n[9/9] Final dataset separation check...")

print(
    "Training rows used:",
    len(X_train)
)

print(
    "Validation rows used:",
    len(X_val)
)

print(
    "Official test rows kept separate:",
    len(X_test)
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nBASELINE MODEL")
print("-" * 70)

print("TF-IDF + Logistic Regression")

print("\nVALIDATION PERFORMANCE")
print("-" * 70)

print(f"Accuracy           : {val_accuracy:.4f}")
print(f"Weighted Precision : {val_precision:.4f}")
print(f"Weighted Recall    : {val_recall:.4f}")
print(f"Weighted F1        : {val_f1_weighted:.4f}")
print(f"Macro F1           : {val_f1_macro:.4f}")

print("\nFILES CREATED")
print("-" * 70)

print(MODEL_PATH)
print(VECTORIZER_PATH)
print(METRICS_PATH)
print(REPORT_PATH)
print(CONFUSION_MATRIX_PATH)

print("\n" + "=" * 70)
print("READY FOR STEP 3: TRANSFORMER / BERT MODEL")
print("=" * 70)
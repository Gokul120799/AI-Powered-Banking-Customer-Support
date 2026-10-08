# ============================================================
# STEP 4 - FINAL MODEL EVALUATION
# AI-Powered Banking Customer Support
# ============================================================
#
# PURPOSE:
#   Evaluate the selected final DistilBERT model on the
#   official test dataset.
#
# IMPORTANT:
#   - NO TRAINING is performed.
#   - Official test data is used ONLY for final evaluation.
#   - Uses the best Experiment 2 checkpoint.
#   - Also evaluates the ML baseline on the SAME official
#     test dataset.
#
# EXPERIMENT 2:
#   Validation Accuracy : 88.95%
#   Weighted F1         : 88.76%
#   Macro F1            : 87.74%
#
# OUTPUTS:
#
# results/
# ├── step3_exp2/
# │   └── validation_metrics_failed_backup.json
# │
# └── evaluation/
#     ├── final_metrics.json
#     ├── ml_vs_transformer_comparison.json
#     ├── transformer_classification_report.txt
#     ├── ml_classification_report.txt
#     ├── transformer_predictions.csv
#     ├── ml_predictions.csv
#     ├── confusion_matrix.csv
#     ├── confusion_matrix.png
#     ├── normalized_confusion_matrix.csv
#     └── normalized_confusion_matrix.png
#
# ============================================================


import os
import json
import warnings

import numpy as np
import pandas as pd
import torch
import joblib

from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# 1. BASE PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ============================================================
# 2. DATASET PATHS
# ============================================================

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "test_clean.csv"
)


LABEL_MAPPING_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)


# ============================================================
# 3. FINAL TRANSFORMER MODEL
# ============================================================

TRANSFORMER_MODEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints_step3_exp2",
    "best_model"
)


# ============================================================
# 4. ML BASELINE FILES
# ============================================================

ML_MODEL_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints",
    "tfidf_logistic_regression.joblib"
)


TFIDF_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints",
    "tfidf_vectorizer.joblib"
)


# ============================================================
# 5. RESULT DIRECTORIES
# ============================================================

STEP3_EXP2_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "step3_exp2"
)


EVALUATION_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "evaluation"
)


os.makedirs(
    STEP3_EXP2_RESULTS_DIR,
    exist_ok=True
)


os.makedirs(
    EVALUATION_RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 6. HISTORICAL FAILED JSON BACKUP
# ============================================================
#
# The original Experiment 2 training completed successfully.
# Only the final JSON serialization failed because a NumPy
# bool_ value was passed to json.dump().
#
# This record preserves that event for future documentation.
#
# ============================================================


failed_json_backup = {

    "status":
        "FAILED_TO_SAVE",

    "stage":
        "Experiment 2 final validation metrics",

    "error_type":
        "TypeError",

    "error_message":
        "Object of type bool_ is not JSON serializable",

    "cause":
        "A NumPy bool_ value was passed to json.dump().",

    "training_status":
        "COMPLETED",

    "best_validation_accuracy":
        0.8895,

    "best_validation_accuracy_percent":
        88.95,

    "best_validation_weighted_f1":
        0.8876,

    "best_validation_weighted_f1_percent":
        88.76,

    "best_validation_macro_f1":
        0.8774,

    "best_validation_macro_f1_percent":
        87.74,

    "best_model_path":
        "dataset/checkpoints_step3_exp2/best_model",

    "official_test_used":
        False,

    "resolution":
        (
            "Converted the NumPy boolean expression "
            "to a native Python bool using bool(...)."
        ),

    "historical_note":
        (
            "Training and best-model saving succeeded. "
            "Only final JSON serialization failed."
        )
}


FAILED_JSON_BACKUP_FILE = os.path.join(
    STEP3_EXP2_RESULTS_DIR,
    "validation_metrics_failed_backup.json"
)


with open(
    FAILED_JSON_BACKUP_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        failed_json_backup,
        f,
        indent=4
    )


# ============================================================
# 7. EVALUATION SETTINGS
# ============================================================

MAX_LENGTH = 64

BATCH_SIZE = 16


# ============================================================
# 8. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 75)
print("STEP 4 - FINAL MODEL EVALUATION")
print("=" * 75)

print(
    f"Device:              {device}"
)

print(
    f"Official test file:  {TEST_FILE}"
)

print(
    f"Transformer model:   {TRANSFORMER_MODEL_DIR}"
)

print(
    f"ML model:            {ML_MODEL_FILE}"
)

print("=" * 75)


# ============================================================
# 9. VERIFY REQUIRED FILES
# ============================================================

print()
print("Checking required files...")
print("-" * 75)


required_files = {

    "Official test dataset":
        TEST_FILE,

    "Label mapping":
        LABEL_MAPPING_FILE,

    "Transformer model":
        TRANSFORMER_MODEL_DIR,

    "ML model":
        ML_MODEL_FILE,

    "TF-IDF vectorizer":
        TFIDF_FILE
}


for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )

    print(
        f"[OK] {name}"
    )


print()
print("[OK] All required files are available.")


print()
print(
    "[OK] Historical failed JSON preserved at:"
)

print(
    FAILED_JSON_BACKUP_FILE
)


# ============================================================
# 10. LOAD OFFICIAL LABEL MAPPING
# ============================================================

print()
print("Loading official label mapping...")


with open(
    LABEL_MAPPING_FILE,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)


if "id2label" in label_mapping:

    id2label = {

        int(k): v

        for k, v
        in label_mapping["id2label"].items()
    }


elif "label2id" in label_mapping:

    id2label = {

        int(v): k

        for k, v
        in label_mapping["label2id"].items()
    }


else:

    raise ValueError(
        "label_mapping.json does not contain "
        "'id2label' or 'label2id'."
    )


label2id = {

    name: idx

    for idx, name
    in id2label.items()
}


NUM_LABELS = len(
    id2label
)


print(
    f"Number of classes: {NUM_LABELS}"
)


if NUM_LABELS != 77:

    raise ValueError(
        f"Expected 77 classes but found "
        f"{NUM_LABELS}."
    )


# ============================================================
# 11. LOAD OFFICIAL TEST DATA
# ============================================================

print()
print("Loading official test dataset...")


test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"Original test shape: {test_df.shape}"
)


required_columns = {

    "text",
    "category",
    "label"
}


missing_columns = (
    required_columns
    -
    set(test_df.columns)
)


if missing_columns:

    raise ValueError(
        f"Missing test columns: "
        f"{missing_columns}"
    )


test_df = test_df.dropna(
    subset=[
        "text",
        "category",
        "label"
    ]
).copy()


test_df["text"] = (
    test_df["text"]
    .astype(str)
)


test_df["label"] = (
    test_df["label"]
    .astype(int)
)


print(
    f"Test samples after validation: "
    f"{len(test_df)}"
)


# ============================================================
# 12. VERIFY TEST LABELS
# ============================================================

test_labels = set(
    test_df["label"].unique()
)


mapping_labels = set(
    id2label.keys()
)


if not test_labels.issubset(
    mapping_labels
):

    raise ValueError(
        "Official test data contains labels "
        "not present in label_mapping.json."
    )


print(
    f"Test labels verified: "
    f"{len(test_labels)} classes"
)


# ============================================================
# 13. TRANSFORMER DATASET
# ============================================================

class BankingTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.texts = (
            dataframe["text"]
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .tolist()
        )

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(self):

        return len(
            self.texts
        )


    def __getitem__(
        self,
        idx
    ):

        text = self.texts[idx]

        label = self.labels[idx]


        encoding = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt"
        )


        item = {

            key:
                value.squeeze(0)

            for key, value
            in encoding.items()
        }


        item["labels"] = torch.tensor(

            label,

            dtype=torch.long
        )


        return item


# ============================================================
# 14. LOAD FINAL DISTILBERT MODEL
# ============================================================

print()
print("Loading final DistilBERT model...")


tokenizer = (
    DistilBertTokenizerFast
    .from_pretrained(
        TRANSFORMER_MODEL_DIR
    )
)


transformer_model = (
    DistilBertForSequenceClassification
    .from_pretrained(
        TRANSFORMER_MODEL_DIR
    )
)


transformer_model.to(
    device
)


transformer_model.eval()


print(
    "[OK] Final Transformer model loaded."
)


print(
    f"Model classes: "
    f"{transformer_model.config.num_labels}"
)


if transformer_model.config.num_labels != NUM_LABELS:

    raise ValueError(
        "Model class count does not match "
        "official label mapping."
    )


# ============================================================
# 15. CREATE TEST DATALOADER
# ============================================================

test_dataset = BankingTestDataset(

    test_df,

    tokenizer,

    MAX_LENGTH
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


print(
    f"Test batches: "
    f"{len(test_loader)}"
)


# ============================================================
# 16. TRANSFORMER PREDICTION
# ============================================================

print()
print("=" * 75)
print("TRANSFORMER OFFICIAL TEST EVALUATION")
print("=" * 75)


transformer_predictions = []

transformer_true_labels = []

transformer_confidences = []

transformer_top3_predictions = []

transformer_top3_confidences = []


with torch.no_grad():

    for batch_index, batch in enumerate(

        test_loader,

        start=1
    ):

        batch_inputs = {

            key:
                value.to(device)

            for key, value
            in batch.items()

            if key != "labels"
        }


        labels = (
            batch["labels"]
            .to(device)
        )


        outputs = transformer_model(

            **batch_inputs
        )


        logits = outputs.logits


        probabilities = torch.softmax(

            logits,

            dim=1
        )


        confidences, predictions = torch.max(

            probabilities,

            dim=1
        )


        top3_values, top3_indices = torch.topk(

            probabilities,

            k=3,

            dim=1
        )


        transformer_predictions.extend(

            predictions
            .cpu()
            .numpy()
            .tolist()
        )


        transformer_true_labels.extend(

            labels
            .cpu()
            .numpy()
            .tolist()
        )


        transformer_confidences.extend(

            confidences
            .cpu()
            .numpy()
            .tolist()
        )


        transformer_top3_predictions.extend(

            top3_indices
            .cpu()
            .numpy()
            .tolist()
        )


        transformer_top3_confidences.extend(

            top3_values
            .cpu()
            .numpy()
            .tolist()
        )


        if (

            batch_index == 1

            or batch_index % 20 == 0

            or batch_index == len(
                test_loader
            )
        ):

            print(

                f"Processed batches: "
                f"{batch_index}/"
                f"{len(test_loader)}"
            )


# ============================================================
# 17. TRANSFORMER METRICS
# ============================================================

transformer_accuracy = accuracy_score(

    transformer_true_labels,

    transformer_predictions
)


(
    transformer_precision,
    transformer_recall,
    transformer_f1,
    _
) = precision_recall_fscore_support(

    transformer_true_labels,

    transformer_predictions,

    average="weighted",

    zero_division=0
)


(
    transformer_macro_precision,
    transformer_macro_recall,
    transformer_macro_f1,
    _
) = precision_recall_fscore_support(

    transformer_true_labels,

    transformer_predictions,

    average="macro",

    zero_division=0
)


print()
print("-" * 75)
print("TRANSFORMER RESULTS")
print("-" * 75)


print(

    f"Accuracy:           "
    f"{transformer_accuracy:.4f} "
    f"({transformer_accuracy * 100:.2f}%)"
)


print(

    f"Weighted Precision: "
    f"{transformer_precision:.4f} "
    f"({transformer_precision * 100:.2f}%)"
)


print(

    f"Weighted Recall:    "
    f"{transformer_recall:.4f} "
    f"({transformer_recall * 100:.2f}%)"
)


print(

    f"Weighted F1:        "
    f"{transformer_f1:.4f} "
    f"({transformer_f1 * 100:.2f}%)"
)


print(

    f"Macro Precision:    "
    f"{transformer_macro_precision:.4f}"
)


print(

    f"Macro Recall:       "
    f"{transformer_macro_recall:.4f}"
)


print(

    f"Macro F1:           "
    f"{transformer_macro_f1:.4f} "
    f"({transformer_macro_f1 * 100:.2f}%)"
)


# ============================================================
# 18. CONFIDENCE ANALYSIS
# ============================================================

confidence_array = np.array(

    transformer_confidences,

    dtype=float
)


mean_confidence = float(

    np.mean(
        confidence_array
    )
)


median_confidence = float(

    np.median(
        confidence_array
    )
)


minimum_confidence = float(

    np.min(
        confidence_array
    )
)


maximum_confidence = float(

    np.max(
        confidence_array
    )
)


high_confidence_count = int(

    np.sum(
        confidence_array >= 0.80
    )
)


medium_confidence_count = int(

    np.sum(

        (
            confidence_array >= 0.60
        )
        &
        (
            confidence_array < 0.80
        )
    )
)


low_confidence_count = int(

    np.sum(

        confidence_array < 0.60
    )
)


total_predictions = int(

    len(
        confidence_array
    )
)


print()
print("-" * 75)
print("PREDICTION CONFIDENCE")
print("-" * 75)


print(

    f"Mean confidence:    "
    f"{mean_confidence:.4f}"
)


print(

    f"Median confidence:  "
    f"{median_confidence:.4f}"
)


print(

    f"Minimum confidence: "
    f"{minimum_confidence:.4f}"
)


print(

    f"Maximum confidence: "
    f"{maximum_confidence:.4f}"
)


print(

    f"High confidence:    "
    f"{high_confidence_count} "
    f"("
    f"{high_confidence_count / total_predictions * 100:.2f}%"
    f")"
)


print(

    f"Medium confidence:  "
    f"{medium_confidence_count} "
    f"("
    f"{medium_confidence_count / total_predictions * 100:.2f}%"
    f")"
)


print(

    f"Low confidence:     "
    f"{low_confidence_count} "
    f"("
    f"{low_confidence_count / total_predictions * 100:.2f}%"
    f")"
)


# ============================================================
# 19. TRANSFORMER CLASSIFICATION REPORT
# ============================================================

transformer_report = classification_report(

    transformer_true_labels,

    transformer_predictions,

    labels=list(
        range(NUM_LABELS)
    ),

    target_names=[

        id2label[i]

        for i in range(NUM_LABELS)
    ],

    zero_division=0
)


transformer_report_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "transformer_classification_report.txt"
)


with open(

    transformer_report_file,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        transformer_report
    )


print()
print(
    "[OK] Transformer classification report saved."
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

print()
print(
    "Generating confusion matrix..."
)


cm = confusion_matrix(

    transformer_true_labels,

    transformer_predictions,

    labels=list(
        range(NUM_LABELS)
    )
)


cm_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "confusion_matrix.csv"
)


cm_dataframe = pd.DataFrame(

    cm,

    index=[

        id2label[i]

        for i in range(NUM_LABELS)
    ],

    columns=[

        id2label[i]

        for i in range(NUM_LABELS)
    ]
)


cm_dataframe.to_csv(
    cm_file
)


# ============================================================
# 21. CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(24, 22)
)


plt.imshow(

    cm,

    interpolation="nearest",

    aspect="auto"
)


plt.title(

    "DistilBERT Confusion Matrix - Official Test Set"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.colorbar()


plt.xticks(

    range(NUM_LABELS),

    [
        id2label[i]
        for i in range(NUM_LABELS)
    ],

    rotation=90,

    fontsize=6
)


plt.yticks(

    range(NUM_LABELS),

    [
        id2label[i]
        for i in range(NUM_LABELS)
    ],

    fontsize=6
)


plt.tight_layout()


cm_png = os.path.join(

    EVALUATION_RESULTS_DIR,

    "confusion_matrix.png"
)


plt.savefig(

    cm_png,

    dpi=200,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 22. NORMALIZED CONFUSION MATRIX
# ============================================================

row_sums = cm.sum(
    axis=1,
    keepdims=True
)


cm_normalized = np.divide(

    cm.astype(float),

    row_sums,

    out=np.zeros_like(
        cm,
        dtype=float
    ),

    where=row_sums != 0
)


normalized_cm_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "normalized_confusion_matrix.csv"
)


normalized_cm_dataframe = pd.DataFrame(

    cm_normalized,

    index=[

        id2label[i]

        for i in range(NUM_LABELS)
    ],

    columns=[

        id2label[i]

        for i in range(NUM_LABELS)
    ]
)


normalized_cm_dataframe.to_csv(

    normalized_cm_file
)


plt.figure(

    figsize=(24, 22)
)


plt.imshow(

    cm_normalized,

    interpolation="nearest",

    aspect="auto"
)


plt.title(

    "Normalized DistilBERT Confusion Matrix"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.colorbar()


plt.xticks(

    range(NUM_LABELS),

    [
        id2label[i]
        for i in range(NUM_LABELS)
    ],

    rotation=90,

    fontsize=6
)


plt.yticks(

    range(NUM_LABELS),

    [
        id2label[i]
        for i in range(NUM_LABELS)
    ],

    fontsize=6
)


plt.tight_layout()


normalized_cm_png = os.path.join(

    EVALUATION_RESULTS_DIR,

    "normalized_confusion_matrix.png"
)


plt.savefig(

    normalized_cm_png,

    dpi=200,

    bbox_inches="tight"
)


plt.close()


print(
    "[OK] Confusion matrices saved."
)


# ============================================================
# 23. SAVE TRANSFORMER PREDICTIONS
# ============================================================

transformer_prediction_df = pd.DataFrame({

    "text":
        test_df["text"].values,

    "true_label_id":
        transformer_true_labels,

    "true_category":

        [
            id2label[i]

            for i
            in transformer_true_labels
        ],

    "predicted_label_id":
        transformer_predictions,

    "predicted_category":

        [
            id2label[i]

            for i
            in transformer_predictions
        ],

    "confidence":
        transformer_confidences
})


transformer_prediction_df["correct"] = (

    transformer_prediction_df[
        "true_label_id"
    ]

    ==

    transformer_prediction_df[
        "predicted_label_id"
    ]
)


# ------------------------------------------------------------
# Add top-3 predictions
# ------------------------------------------------------------

transformer_prediction_df[
    "top_1_category"
] = [

    id2label[int(
        row[0]
    )]

    for row
    in transformer_top3_predictions
]


transformer_prediction_df[
    "top_2_category"
] = [

    id2label[int(
        row[1]
    )]

    for row
    in transformer_top3_predictions
]


transformer_prediction_df[
    "top_3_category"
] = [

    id2label[int(
        row[2]
    )]

    for row
    in transformer_top3_predictions
]


transformer_prediction_df[
    "top_1_confidence"
] = [

    float(
        row[0]
    )

    for row
    in transformer_top3_confidences
]


transformer_prediction_df[
    "top_2_confidence"
] = [

    float(
        row[1]
    )

    for row
    in transformer_top3_confidences
]


transformer_prediction_df[
    "top_3_confidence"
] = [

    float(
        row[2]
    )

    for row
    in transformer_top3_confidences
]


transformer_prediction_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "transformer_predictions.csv"
)


transformer_prediction_df.to_csv(

    transformer_prediction_file,

    index=False
)


print(
    "[OK] Transformer predictions saved."
)


# ============================================================
# 24. LOAD ML BASELINE
# ============================================================

print()
print("=" * 75)
print("ML BASELINE - OFFICIAL TEST EVALUATION")
print("=" * 75)


print()
print(
    "Loading TF-IDF vectorizer..."
)


tfidf_vectorizer = joblib.load(

    TFIDF_FILE
)


print(
    "Loading Logistic Regression model..."
)


ml_model = joblib.load(

    ML_MODEL_FILE
)


print(
    "[OK] ML baseline loaded."
)


# ============================================================
# 25. ML BASELINE PREDICTIONS
# ============================================================

test_texts = (

    test_df["text"]
    .tolist()
)


test_labels_ml = (

    test_df["label"]
    .astype(int)
    .tolist()
)


print()
print(
    "Transforming official test data..."
)


X_test_tfidf = (

    tfidf_vectorizer
    .transform(
        test_texts
    )
)


print(

    f"TF-IDF test matrix shape: "
    f"{X_test_tfidf.shape}"
)


ml_predictions = (

    ml_model
    .predict(
        X_test_tfidf
    )
)


# ============================================================
# 26. ML METRICS
# ============================================================

ml_accuracy = accuracy_score(

    test_labels_ml,

    ml_predictions
)


(
    ml_precision,
    ml_recall,
    ml_f1,
    _
) = precision_recall_fscore_support(

    test_labels_ml,

    ml_predictions,

    average="weighted",

    zero_division=0
)


(
    ml_macro_precision,
    ml_macro_recall,
    ml_macro_f1,
    _
) = precision_recall_fscore_support(

    test_labels_ml,

    ml_predictions,

    average="macro",

    zero_division=0
)


print()
print("-" * 75)
print("ML BASELINE RESULTS")
print("-" * 75)


print(

    f"Accuracy:           "
    f"{ml_accuracy:.4f} "
    f"({ml_accuracy * 100:.2f}%)"
)


print(

    f"Weighted Precision: "
    f"{ml_precision:.4f} "
    f"({ml_precision * 100:.2f}%)"
)


print(

    f"Weighted Recall:    "
    f"{ml_recall:.4f} "
    f"({ml_recall * 100:.2f}%)"
)


print(

    f"Weighted F1:        "
    f"{ml_f1:.4f} "
    f"({ml_f1 * 100:.2f}%)"
)


print(

    f"Macro F1:           "
    f"{ml_macro_f1:.4f} "
    f"({ml_macro_f1 * 100:.2f}%)"
)


# ============================================================
# 27. ML CLASSIFICATION REPORT
# ============================================================

ml_report = classification_report(

    test_labels_ml,

    ml_predictions,

    labels=list(
        range(NUM_LABELS)
    ),

    target_names=[

        id2label[i]

        for i in range(NUM_LABELS)
    ],

    zero_division=0
)


ml_report_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "ml_classification_report.txt"
)


with open(

    ml_report_file,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        ml_report
    )


# ============================================================
# 28. SAVE ML PREDICTIONS
# ============================================================

ml_prediction_df = pd.DataFrame({

    "text":
        test_df["text"].values,

    "true_label_id":
        test_labels_ml,

    "true_category":

        [
            id2label[i]

            for i
            in test_labels_ml
        ],

    "predicted_label_id":
        ml_predictions,

    "predicted_category":

        [
            id2label[int(i)]

            for i
            in ml_predictions
        ]
})


ml_prediction_df["correct"] = (

    ml_prediction_df[
        "true_label_id"
    ]

    ==

    ml_prediction_df[
        "predicted_label_id"
    ]
)


ml_prediction_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "ml_predictions.csv"
)


ml_prediction_df.to_csv(

    ml_prediction_file,

    index=False
)


print(
    "[OK] ML predictions saved."
)


# ============================================================
# 29. ML VS TRANSFORMER COMPARISON
# ============================================================

accuracy_improvement = (

    transformer_accuracy
    -
    ml_accuracy
)


f1_improvement = (

    transformer_f1
    -
    ml_f1
)


comparison = {

    "test_samples":
        int(len(test_df)),

    "number_of_classes":
        int(NUM_LABELS),


    "ml_baseline": {

        "model":
            "TF-IDF + Logistic Regression",

        "accuracy":
            float(ml_accuracy),

        "accuracy_percent":
            float(
                ml_accuracy * 100
            ),

        "weighted_precision":
            float(ml_precision),

        "weighted_recall":
            float(ml_recall),

        "weighted_f1":
            float(ml_f1),

        "macro_f1":
            float(ml_macro_f1)
    },


    "transformer": {

        "model":
            "DistilBERT",

        "checkpoint":
            str(
                TRANSFORMER_MODEL_DIR
            ),

        "accuracy":
            float(
                transformer_accuracy
            ),

        "accuracy_percent":
            float(
                transformer_accuracy * 100
            ),

        "weighted_precision":
            float(
                transformer_precision
            ),

        "weighted_recall":
            float(
                transformer_recall
            ),

        "weighted_f1":
            float(
                transformer_f1
            ),

        "macro_f1":
            float(
                transformer_macro_f1
            ),

        "mean_confidence":
            float(
                mean_confidence
            )
    },


    "transformer_minus_ml": {

        "accuracy_difference":
            float(
                accuracy_improvement
            ),

        "accuracy_difference_percentage_points":
            float(
                accuracy_improvement * 100
            ),

        "weighted_f1_difference":
            float(
                f1_improvement
            ),

        "weighted_f1_difference_percentage_points":
            float(
                f1_improvement * 100
            )
    }
}


comparison_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "ml_vs_transformer_comparison.json"
)


with open(

    comparison_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        comparison,

        f,

        indent=4
    )


# ============================================================
# 30. FINAL METRICS JSON
# ============================================================

final_metrics = {

    "evaluation_stage":
        "Final official test evaluation",

    "official_test_used":
        True,

    "test_samples":
        int(len(test_df)),

    "number_of_classes":
        int(NUM_LABELS),


    "transformer": {

        "model":
            "DistilBERT",

        "model_path":
            str(
                TRANSFORMER_MODEL_DIR
            ),

        "accuracy":
            float(
                transformer_accuracy
            ),

        "accuracy_percent":
            float(
                transformer_accuracy * 100
            ),

        "weighted_precision":
            float(
                transformer_precision
            ),

        "weighted_recall":
            float(
                transformer_recall
            ),

        "weighted_f1":
            float(
                transformer_f1
            ),

        "macro_precision":
            float(
                transformer_macro_precision
            ),

        "macro_recall":
            float(
                transformer_macro_recall
            ),

        "macro_f1":
            float(
                transformer_macro_f1
            ),

        "mean_confidence":
            float(
                mean_confidence
            ),

        "median_confidence":
            float(
                median_confidence
            ),

        "minimum_confidence":
            float(
                minimum_confidence
            ),

        "maximum_confidence":
            float(
                maximum_confidence
            )
    },


    "confidence_distribution": {

        "high_threshold":
            0.80,

        "medium_threshold":
            0.60,

        "high_count":
            int(
                high_confidence_count
            ),

        "medium_count":
            int(
                medium_confidence_count
            ),

        "low_count":
            int(
                low_confidence_count
            ),

        "high_percentage":
            float(

                high_confidence_count
                /
                total_predictions
                *
                100
            ),

        "medium_percentage":
            float(

                medium_confidence_count
                /
                total_predictions
                *
                100
            ),

        "low_percentage":
            float(

                low_confidence_count
                /
                total_predictions
                *
                100
            )
    },


    "ml_baseline": {

        "model":
            "TF-IDF + Logistic Regression",

        "accuracy":
            float(
                ml_accuracy
            ),

        "accuracy_percent":
            float(
                ml_accuracy * 100
            ),

        "weighted_precision":
            float(
                ml_precision
            ),

        "weighted_recall":
            float(
                ml_recall
            ),

        "weighted_f1":
            float(
                ml_f1
            ),

        "macro_f1":
            float(
                ml_macro_f1
            )
    },


    "comparison": {

        "accuracy_difference":
            float(
                accuracy_improvement
            ),

        "accuracy_difference_percentage_points":
            float(
                accuracy_improvement * 100
            ),

        "weighted_f1_difference":
            float(
                f1_improvement
            ),

        "weighted_f1_difference_percentage_points":
            float(
                f1_improvement * 100
            )
    }
}


metrics_file = os.path.join(

    EVALUATION_RESULTS_DIR,

    "final_metrics.json"
)


with open(

    metrics_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        final_metrics,

        f,

        indent=4
    )


# ============================================================
# 31. FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("FINAL OFFICIAL TEST EVALUATION COMPLETE")
print("=" * 75)


print()
print("TRANSFORMER - DISTILBERT")
print("-" * 50)


print(

    f"Accuracy:        "
    f"{transformer_accuracy * 100:.2f}%"
)


print(

    f"Weighted F1:     "
    f"{transformer_f1 * 100:.2f}%"
)


print(

    f"Macro F1:        "
    f"{transformer_macro_f1 * 100:.2f}%"
)


print(

    f"Mean Confidence:"
    f" {mean_confidence * 100:.2f}%"
)


print()
print("ML BASELINE")
print("-" * 50)


print(

    f"Accuracy:        "
    f"{ml_accuracy * 100:.2f}%"
)


print(

    f"Weighted F1:     "
    f"{ml_f1 * 100:.2f}%"
)


print()
print("IMPROVEMENT")
print("-" * 50)


print(

    f"Accuracy difference: "
    f"{accuracy_improvement * 100:+.2f} "
    f"percentage points"
)


print(

    f"Weighted F1 difference: "
    f"{f1_improvement * 100:+.2f} "
    f"percentage points"
)


print()
print("RESULT FILES")
print("-" * 50)


print(
    "\nHistorical failed JSON:"
)

print(
    FAILED_JSON_BACKUP_FILE
)


print(
    "\nFinal metrics:"
)

print(
    metrics_file
)


print(
    "\nML vs Transformer:"
)

print(
    comparison_file
)


print(
    "\nTransformer report:"
)

print(
    transformer_report_file
)


print(
    "\nML report:"
)

print(
    ml_report_file
)


print(
    "\nTransformer predictions:"
)

print(
    transformer_prediction_file
)


print(
    "\nML predictions:"
)

print(
    ml_prediction_file
)


print(
    "\nConfusion matrix:"
)

print(
    cm_png
)


print(
    "\nNormalized confusion matrix:"
)

print(
    normalized_cm_png
)


print()
print("=" * 75)
print("IMPORTANT")
print("=" * 75)


print(
    "Official test data was used ONLY for final evaluation."
)


print(
    "No training or model tuning was performed."
)


print(
    "Selected model: Experiment 2 best_model"
)


print(
    "Historical Experiment 2 JSON failure was preserved."
)


print("=" * 75)
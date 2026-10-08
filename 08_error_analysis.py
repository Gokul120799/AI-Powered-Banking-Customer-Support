# ============================================================
# Step 8A: Error & Confusion Analysis
# AI-Powered Banking Customer Support
# ============================================================
# PURPOSE
# Analyze the already-generated OFFICIAL TEST predictions.
# IMPORTANT: This script DOES NOT train or retrain the model.
# It is for diagnosis only. Do not use its findings to tune the
# model on the official test set.
# ============================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "evaluation"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = EVAL_DIR / "transformer_predictions.csv"
TEST_FILE = PROCESSED_DIR / "test_clean.csv"
MAPPING_FILE = PROCESSED_DIR / "label_mapping.json"

print("=" * 70)
print("STEP 8A — ERROR & CONFUSION ANALYSIS")
print("=" * 70)
print("No model training/retraining will be performed.")
print()

# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------
if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_FILE}\n"
        "Run src\\04_evaluate_model.py first."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

if not MAPPING_FILE.exists():
    raise FileNotFoundError(f"Label mapping not found: {MAPPING_FILE}")

pred = pd.read_csv(PREDICTIONS_FILE)
test = pd.read_csv(TEST_FILE)

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# Normalize mapping in case keys are strings.
id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = {v: int(k) for k, v in id2label.items()}

print(f"Prediction rows: {len(pred):,}")
print(f"Official test rows: {len(test):,}")
print(f"Number of official classes: {len(id2label)}")
print(f"Prediction columns: {list(pred.columns)}")
print()

# ------------------------------------------------------------
# Reconstruct/align prediction table robustly
# ------------------------------------------------------------
# The evaluation script normally saves text, true/predicted category,
# confidence and top-3 information. We support common column names.
def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

true_col = first_existing(pred, ["true_category", "actual_category", "category", "true_label"])
pred_col = first_existing(pred, ["predicted_category", "prediction", "predicted_label"])
conf_col = first_existing(pred, ["confidence", "top1_confidence", "prediction_confidence"])
text_col = first_existing(pred, ["text", "query", "customer_query"])

# If the prediction file does not contain the true category, align it to
# test_clean.csv by row order, which is how 04_evaluate_model.py creates
# official-test predictions.
if true_col is None:
    if len(pred) != len(test):
        raise ValueError(
            "Prediction file has no true-label column and its row count "
            "does not match test_clean.csv. Cannot safely align rows."
        )
    pred["true_category"] = test["category"].astype(str).values
    true_col = "true_category"

if pred_col is None:
    raise ValueError(
        "Could not find predicted-category column in transformer_predictions.csv."
    )

if conf_col is None:
    raise ValueError(
        "Could not find confidence column in transformer_predictions.csv."
    )

if text_col is None and len(pred) == len(test):
    pred["text"] = test["text"].astype(str).values
    text_col = "text"

pred[true_col] = pred[true_col].astype(str)
pred[pred_col] = pred[pred_col].astype(str)
pred[conf_col] = pd.to_numeric(pred[conf_col], errors="coerce")

# ------------------------------------------------------------
# Correct / incorrect flags
# ------------------------------------------------------------
pred["correct"] = pred[true_col] == pred[pred_col]
pred["error_type"] = np.where(pred["correct"], "CORRECT", "MISCLASSIFIED")

if text_col is not None:
    pred["query"] = pred[text_col].astype(str)
else:
    pred["query"] = ""

# ------------------------------------------------------------
# 1. Overall error summary
# ------------------------------------------------------------
total = len(pred)
correct = int(pred["correct"].sum())
incorrect = total - correct
accuracy = correct / total if total else 0.0
error_rate = incorrect / total if total else 0.0

summary = {
    "official_test_rows": int(total),
    "correct_predictions": int(correct),
    "incorrect_predictions": int(incorrect),
    "accuracy": float(accuracy),
    "error_rate": float(error_rate),
    "note": "Diagnostic analysis only; official test set must remain untouched for tuning."
}

with open(OUTPUT_DIR / "error_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("1. OVERALL ERROR SUMMARY")
print(f"   Accuracy:    {accuracy:.4%}")
print(f"   Error rate:  {error_rate:.4%}")
print(f"   Correct:     {correct:,}")
print(f"   Incorrect:   {incorrect:,}")
print()

# ------------------------------------------------------------
# 2. Classification report by class
# ------------------------------------------------------------
labels = sorted(set(pred[true_col]) | set(pred[pred_col]))
report_dict = classification_report(
    pred[true_col],
    pred[pred_col],
    labels=labels,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=True)

# Identify weakest classes by F1 and recall.
class_rows = report_df.loc[
    [x for x in labels if x in report_df.index],
    ["precision", "recall", "f1-score", "support"],
].copy()
class_rows = class_rows.sort_values("f1-score")
class_rows.head(15).to_csv(OUTPUT_DIR / "weakest_15_classes.csv")

print("2. WEAKEST CLASSES")
print(class_rows.head(10).round(4).to_string())
print()

# ------------------------------------------------------------
# 3. Confusion pairs
# ------------------------------------------------------------
wrong = pred.loc[~pred["correct"]].copy()

if len(wrong):
    pairs = (
        wrong.groupby([true_col, pred_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
else:
    pairs = pd.DataFrame(columns=[true_col, pred_col, "count"])

pairs.to_csv(OUTPUT_DIR / "top_confusion_pairs.csv", index=False)

print("3. TOP CONFUSION PAIRS")
if len(pairs):
    print(pairs.head(15).to_string(index=False))
else:
    print("   No misclassified predictions found.")
print()

# ------------------------------------------------------------
# 4. Highest-confidence incorrect predictions
# ------------------------------------------------------------
high_conf_wrong = wrong.sort_values(conf_col, ascending=False).copy()
high_conf_wrong = high_conf_wrong[
    ["query", true_col, pred_col, conf_col]
].head(50)
high_conf_wrong.to_csv(
    OUTPUT_DIR / "high_confidence_incorrect_predictions.csv", index=False
)

print("4. HIGH-CONFIDENCE INCORRECT PREDICTIONS")
if len(high_conf_wrong):
    print(high_conf_wrong.head(10).to_string(index=False))
else:
    print("   None found.")
print()

# ------------------------------------------------------------
# 5. Lowest-confidence correct predictions
# ------------------------------------------------------------
low_conf_correct = pred.loc[pred["correct"]].sort_values(conf_col).copy()
low_conf_correct = low_conf_correct[
    ["query", true_col, pred_col, conf_col]
].head(50)
low_conf_correct.to_csv(
    OUTPUT_DIR / "low_confidence_correct_predictions.csv", index=False
)

print("5. LOW-CONFIDENCE CORRECT PREDICTIONS")
if len(low_conf_correct):
    print(low_conf_correct.head(10).to_string(index=False))
else:
    print("   None found.")
print()

# ------------------------------------------------------------
# 6. Confidence vs correctness
# ------------------------------------------------------------
pred["confidence_percent"] = pred[conf_col] * 100.0

bins = [-np.inf, 0.60, 0.80, 0.90, 0.95, np.inf]
names = ["<60%", "60-80%", "80-90%", "90-95%", ">=95%"]
pred["confidence_band"] = pd.cut(
    pred[conf_col], bins=bins, labels=names, right=False
)

confidence_analysis = (
    pred.groupby("confidence_band", observed=False)
    .agg(
        predictions=("correct", "size"),
        correct=("correct", "sum"),
        incorrect=("correct", lambda x: int((~x).sum())),
        average_confidence=(conf_col, "mean"),
    )
    .reset_index()
)
confidence_analysis["accuracy"] = np.where(
    confidence_analysis["predictions"] > 0,
    confidence_analysis["correct"] / confidence_analysis["predictions"],
    0,
)
confidence_analysis.to_csv(
    OUTPUT_DIR / "confidence_vs_correctness.csv", index=False
)

print("6. CONFIDENCE VS CORRECTNESS")
print(confidence_analysis.round(4).to_string(index=False))
print()

# ------------------------------------------------------------
# 7. Short-query risk analysis
# ------------------------------------------------------------
pred["word_count"] = pred["query"].str.split().str.len()
pred["short_query"] = pred["word_count"] <= 3

short_summary = (
    pred.groupby("short_query")
    .agg(
        queries=("correct", "size"),
        correct=("correct", "sum"),
        average_confidence=(conf_col, "mean"),
    )
    .reset_index()
)
short_summary["accuracy"] = short_summary["correct"] / short_summary["queries"]
short_summary.to_csv(OUTPUT_DIR / "short_query_analysis.csv", index=False)

short_wrong = pred.loc[pred["short_query"] & ~pred["correct"]].copy()
short_wrong[
    ["query", "word_count", true_col, pred_col, conf_col]
].sort_values(conf_col, ascending=False).head(50).to_csv(
    OUTPUT_DIR / "short_query_errors.csv", index=False
)

print("7. SHORT QUERY ANALYSIS")
print(short_summary.round(4).to_string(index=False))
print()

# ------------------------------------------------------------
# 8. Multi-issue / conjunction heuristic analysis
# ------------------------------------------------------------
# This is diagnostic only. It is NOT used to change model predictions.
multiple_markers = [" and ", " also ", " as well as ", ";"]
pred["possible_multiple_issue"] = pred["query"].str.lower().apply(
    lambda q: any(marker in f" {q} " for marker in multiple_markers)
)

multi_summary = (
    pred.groupby("possible_multiple_issue")
    .agg(
        queries=("correct", "size"),
        correct=("correct", "sum"),
        average_confidence=(conf_col, "mean"),
    )
    .reset_index()
)
multi_summary["accuracy"] = multi_summary["correct"] / multi_summary["queries"]
multi_summary.to_csv(OUTPUT_DIR / "multiple_issue_analysis.csv", index=False)

pred.loc[pred["possible_multiple_issue"] & ~pred["correct"],
         ["query", true_col, pred_col, conf_col]].head(50).to_csv(
    OUTPUT_DIR / "multiple_issue_errors.csv", index=False
)

print("8. POSSIBLE MULTI-ISSUE QUERY ANALYSIS")
print(multi_summary.round(4).to_string(index=False))
print()

# ------------------------------------------------------------
# 9. Compact list of candidate risky queries
# ------------------------------------------------------------
# Risk here means "worth reviewing", NOT "definitely wrong".
pred["top2_gap"] = np.nan

# If the evaluation CSV contains a second confidence column, use it.
top2_conf_col = first_existing(
    pred,
    ["top2_confidence", "second_confidence", "top_2_confidence"]
)
if top2_conf_col:
    pred[top2_conf_col] = pd.to_numeric(pred[top2_conf_col], errors="coerce")
    pred["top2_gap"] = pred[conf_col] - pred[top2_conf_col]

risk_reasons = []
for _, row in pred.iterrows():
    reasons = []
    conf = row[conf_col]

    if pd.notna(conf) and conf < 0.60:
        reasons.append("low_confidence")
    if pd.notna(row["top2_gap"]) and row["top2_gap"] < 0.15:
        reasons.append("small_top2_gap")
    if row["short_query"]:
        reasons.append("very_short_query")
    if row["possible_multiple_issue"]:
        reasons.append("possible_multiple_issue")
    if not bool(row["correct"]):
        reasons.append("observed_misclassification")

    risk_reasons.append(", ".join(reasons) if reasons else "none_detected")

pred["diagnostic_risk_reasons"] = risk_reasons

risky = pred[pred["diagnostic_risk_reasons"] != "none_detected"].copy()
risky = risky.sort_values(
    ["correct", conf_col], ascending=[True, False]
)
risky_cols = [
    "query", true_col, pred_col, conf_col,
    "top2_gap", "diagnostic_risk_reasons"
]
risky[[c for c in risky_cols if c in risky.columns]].head(200).to_csv(
    OUTPUT_DIR / "candidate_risky_queries.csv", index=False
)

# ------------------------------------------------------------
# 10. Final machine-readable summary
# ------------------------------------------------------------
final_summary = {
    **summary,
    "weakest_classes_file": "weakest_15_classes.csv",
    "confusion_pairs_file": "top_confusion_pairs.csv",
    "high_confidence_errors_file": "high_confidence_incorrect_predictions.csv",
    "confidence_analysis_file": "confidence_vs_correctness.csv",
    "candidate_risky_queries_file": "candidate_risky_queries.csv",
    "analysis_is_diagnostic_only": True,
    "official_test_used_for_tuning": False,
}

with open(OUTPUT_DIR / "error_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=4)

print("=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)
print(f"Output folder: {OUTPUT_DIR}")
print("No model was retrained or modified.")
print("The official test set remains evaluation-only.")
print()
print("Key files:")
for name in [
    "error_summary.json",
    "per_class_metrics.csv",
    "weakest_15_classes.csv",
    "top_confusion_pairs.csv",
    "high_confidence_incorrect_predictions.csv",
    "low_confidence_correct_predictions.csv",
    "confidence_vs_correctness.csv",
    "short_query_analysis.csv",
    "multiple_issue_analysis.csv",
    "candidate_risky_queries.csv",
    "error_analysis_summary.json",
]:
    print(f"  - {name}")

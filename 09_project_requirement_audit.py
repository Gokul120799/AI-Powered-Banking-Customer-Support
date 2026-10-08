# ============================================================
# AI-POWERED BANKING CUSTOMER SUPPORT
# FINAL PROJECT REQUIREMENT AUDIT
# ============================================================
#
# Purpose:
#   Audit the CLEAN FINAL_PROJECT against the project
#   requirements and available evidence.
#
# Status values:
#   PASS     = sufficient evidence found
#   PARTIAL  = some evidence found, but manual verification needed
#   MISSING  = required evidence/file/result not found
#   N/A      = requirement is optional / not mandatory
#
# IMPORTANT:
#   This script DOES NOT:
#       - train a model
#       - modify the model
#       - modify datasets
#       - modify app.py
#       - evaluate the official test set again
#
#   It only reads project files and creates an audit report.
# ============================================================

import os
import json
import re
from datetime import datetime


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(BASE_DIR, "src")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

MODEL_DIR = os.path.join(
    DATASET_DIR,
    "checkpoints_step3_exp2",
    "best_model"
)

APP_FILE = os.path.join(BASE_DIR, "app.py")

README_FILE = os.path.join(BASE_DIR, "README.md")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")

AUDIT_OUTPUT = os.path.join(
    REPORTS_DIR,
    "project_requirement_audit.json"
)

AUDIT_TEXT_OUTPUT = os.path.join(
    REPORTS_DIR,
    "project_requirement_audit.txt"
)


# ============================================================
# 2. HELPERS
# ============================================================

results = []


def exists(path):
    return os.path.exists(path)


def read_text(path):
    if not exists(path):
        return ""

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            return f.read()

    except Exception:
        return ""


def add_result(
    requirement,
    status,
    evidence,
    details="",
    action=""
):
    results.append(
        {
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "details": details,
            "recommended_action": action,
        }
    )


def count_files(folder):
    if not exists(folder):
        return 0

    total = 0

    for root, dirs, files in os.walk(folder):
        total += len(files)

    return total


def find_files(folder, extensions=None):
    found = []

    if not exists(folder):
        return found

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if extensions:

                if not any(
                    filename.lower().endswith(ext.lower())
                    for ext in extensions
                ):
                    continue

            found.append(
                os.path.join(root, filename)
            )

    return found


def contains_any(text, terms):
    text_lower = text.lower()

    return any(
        term.lower() in text_lower
        for term in terms
    )


def contains_all(text, terms):
    text_lower = text.lower()

    return all(
        term.lower() in text_lower
        for term in terms
    )


def file_has_any(path, terms):
    text = read_text(path)

    if not text:
        return False

    return contains_any(text, terms)


def format_status(status):
    if status == "PASS":
        return "🟢 PASS"

    if status == "PARTIAL":
        return "🟡 PARTIAL"

    if status == "MISSING":
        return "🔴 MISSING"

    return "⚪ N/A"


# ============================================================
# 3. REQUIRED FINAL SOURCE FILES
# ============================================================

required_source_files = [
    "01_data_preprocessing.py",
    "02_machine_learning_baseline.py",
    "03_deep_learning_bert.py",
    "03b_deep_learning_exp2.py",
    "04_evaluate_model.py",
    "05_inference.py",
    "06_confidence_escalation.py",
    "07_generative_ai.py",
    "08_error_analysis.py",
    "08b_risk_detection.py",
]


# ============================================================
# 4. ROOT FILE CHECK
# ============================================================

root_files_ok = all(
    exists(os.path.join(BASE_DIR, filename))
    for filename in [
        "app.py",
        "README.md",
        "requirements.txt",
    ]
)

if root_files_ok:

    add_result(
        "Project structure / core files",
        "PASS",
        "app.py, README.md and requirements.txt",
        "Core final-project files are present."
    )

else:

    missing = [
        filename
        for filename in [
            "app.py",
            "README.md",
            "requirements.txt",
        ]
        if not exists(os.path.join(BASE_DIR, filename))
    ]

    add_result(
        "Project structure / core files",
        "MISSING",
        "Required root files",
        "Missing: " + ", ".join(missing),
        "Restore the missing required file(s)."
    )


# ============================================================
# 5. DATASET REQUIREMENT
# ============================================================

train_raw = os.path.join(
    RAW_DIR,
    "train.csv"
)

test_raw = os.path.join(
    RAW_DIR,
    "test.csv"
)

train_clean = os.path.join(
    PROCESSED_DIR,
    "train_clean.csv"
)

test_clean = os.path.join(
    PROCESSED_DIR,
    "test_clean.csv"
)

label_mapping = os.path.join(
    PROCESSED_DIR,
    "label_mapping.json"
)

dataset_files_ok = all(
    exists(path)
    for path in [
        train_raw,
        test_raw,
        train_clean,
        test_clean,
        label_mapping,
    ]
)

if dataset_files_ok:

    add_result(
        "Dataset and processed data",
        "PASS",
        (
            "dataset/raw/train.csv; "
            "dataset/raw/test.csv; "
            "dataset/processed/train_clean.csv; "
            "dataset/processed/test_clean.csv; "
            "label_mapping.json"
        ),
        "Raw and processed dataset artifacts are present."
    )

else:

    missing = []

    for path in [
        train_raw,
        test_raw,
        train_clean,
        test_clean,
        label_mapping,
    ]:

        if not exists(path):
            missing.append(
                os.path.relpath(path, BASE_DIR)
            )

    add_result(
        "Dataset and processed data",
        "MISSING",
        "Expected dataset artifacts",
        "Missing: " + ", ".join(missing),
        "Restore the missing dataset artifact(s)."
    )


# ============================================================
# 6. LABEL MAPPING / 77 CLASSES
# ============================================================

if exists(label_mapping):

    try:

        with open(
            label_mapping,
            "r",
            encoding="utf-8"
        ) as f:

            mapping = json.load(f)

        if isinstance(mapping, dict):

            # label_mapping.json has three top-level keys. The actual
            # intent count is stored in num_classes or in label2id/id2label.
            if isinstance(mapping.get("num_classes"), int):
                number_of_labels = mapping["num_classes"]
            elif isinstance(mapping.get("label2id"), dict):
                number_of_labels = len(mapping["label2id"])
            elif isinstance(mapping.get("id2label"), dict):
                number_of_labels = len(mapping["id2label"])
            else:
                number_of_labels = 0

        else:

            number_of_labels = 0

        if number_of_labels == 77:

            add_result(
                "77-category intent classification",
                "PASS",
                "dataset/processed/label_mapping.json",
                "Label mapping contains exactly 77 classes."
            )

        else:

            add_result(
                "77-category intent classification",
                "PARTIAL",
                "label_mapping.json",
                f"Detected {number_of_labels} labels instead of 77.",
                "Verify the official 77-class mapping."
            )

    except Exception as e:

        add_result(
            "77-category intent classification",
            "PARTIAL",
            "label_mapping.json",
            f"Could not parse mapping: {e}",
            "Verify label_mapping.json."
        )

else:

    add_result(
        "77-category intent classification",
        "MISSING",
        "label_mapping.json",
        "Label mapping was not found.",
        "Restore the official label mapping."
    )


# ============================================================
# 7. DATA PREPROCESSING
# ============================================================

preprocess_file = os.path.join(
    SRC_DIR,
    "01_data_preprocessing.py"
)

preprocess_text = read_text(preprocess_file)

preprocessing_terms = [
    "dropna",
    "duplicate",
    "text",
    "category",
    "label",
]

if exists(preprocess_file) and contains_all(
    preprocess_text,
    preprocessing_terms
):

    add_result(
        "NLP / data preprocessing",
        "PASS",
        "src/01_data_preprocessing.py",
        "Preprocessing script contains text/category handling, duplicate and missing-value handling."
    )

elif exists(preprocess_file):

    add_result(
        "NLP / data preprocessing",
        "PARTIAL",
        "src/01_data_preprocessing.py",
        "Preprocessing script exists but not all expected preprocessing evidence was detected.",
        "Review preprocessing implementation manually."
    )

else:

    add_result(
        "NLP / data preprocessing",
        "MISSING",
        "src/01_data_preprocessing.py",
        "Preprocessing script not found."
    )


# ============================================================
# 8. EDA
# ============================================================

eda_evidence = []

report_files = find_files(
    REPORTS_DIR
)

for path in report_files:

    text = read_text(path)

    if contains_any(
        text,
        [
            "category",
            "text_length",
            "duplicate",
            "missing",
            "distribution",
        ]
    ):

        eda_evidence.append(
            os.path.relpath(path, BASE_DIR)
        )


if eda_evidence:

    add_result(
        "Exploratory Data Analysis",
        "PASS",
        "; ".join(eda_evidence[:5]),
        "Dataset statistics / EDA evidence was found."
    )

else:

    notebooks = find_files(
        os.path.join(BASE_DIR, "notebooks"),
        [".ipynb"]
    )

    if notebooks:

        add_result(
            "Exploratory Data Analysis",
            "PARTIAL",
            "; ".join(
                os.path.relpath(
                    x,
                    BASE_DIR
                )
                for x in notebooks
            ),
            "Notebook evidence exists, but EDA result artifacts were not automatically confirmed.",
            "Open the EDA notebook and verify required analyses."
        )

    else:

        add_result(
            "Exploratory Data Analysis",
            "MISSING",
            "results/reports or notebooks",
            "No strong EDA evidence was detected.",
            "Include EDA evidence in the final project."
        )


# ============================================================
# 9. ML BASELINE
# ============================================================

ml_file = os.path.join(
    SRC_DIR,
    "02_machine_learning_baseline.py"
)

ml_text = read_text(ml_file)

ml_terms = [
    "TfidfVectorizer",
    "LogisticRegression",
]

ml_results_files = find_files(
    os.path.join(
        RESULTS_DIR,
        "ml"
    )
)

if exists(ml_file) and contains_all(
    ml_text,
    ml_terms
):

    if ml_results_files:

        add_result(
            "ML baseline",
            "PASS",
            (
                "src/02_machine_learning_baseline.py; "
                "results/ml"
            ),
            "TF-IDF + Logistic Regression baseline and result artifacts found."
        )

    else:

        add_result(
            "ML baseline",
            "PARTIAL",
            "src/02_machine_learning_baseline.py",
            "Baseline implementation found but result artifacts were not detected.",
            "Verify and preserve ML baseline metrics."
        )

else:

    add_result(
        "ML baseline",
        "MISSING",
        "src/02_machine_learning_baseline.py",
        "TF-IDF + Logistic Regression evidence not detected.",
        "Restore or verify the ML baseline."
    )


# ============================================================
# 10. DISTILBERT / DEEP LEARNING
# ============================================================

bert_file = os.path.join(
    SRC_DIR,
    "03b_deep_learning_exp2.py"
)

bert_text = read_text(bert_file)

bert_terms = [
    "DistilBertTokenizerFast",
    "DistilBertForSequenceClassification",
    "distilbert-base-uncased",
    "num_labels",
]

bert_exists = exists(bert_file)

model_exists = exists(MODEL_DIR)

if bert_exists and contains_all(
    bert_text,
    bert_terms
) and model_exists:

    add_result(
        "Deep learning / DistilBERT",
        "PASS",
        (
            "src/03b_deep_learning_exp2.py; "
            "dataset/checkpoints_step3_exp2/best_model"
        ),
        "DistilBERT training implementation and final best model are present."
    )

elif bert_exists and model_exists:

    add_result(
        "Deep learning / DistilBERT",
        "PARTIAL",
        (
            "03b_deep_learning_exp2.py; "
            "best_model"
        ),
        "Training script and model exist, but expected transformer implementation terms were not all detected.",
        "Review the final training script."
    )

else:

    add_result(
        "Deep learning / DistilBERT",
        "MISSING",
        "DistilBERT training script / best_model",
        "Final transformer implementation or model is missing.",
        "Restore the final DistilBERT model and script."
    )


# ============================================================
# 11. OFFICIAL EVALUATION
# ============================================================

evaluation_file = os.path.join(
    SRC_DIR,
    "04_evaluate_model.py"
)

evaluation_text = read_text(evaluation_file)

evaluation_terms = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion",
    "confidence",
]

evaluation_results = find_files(
    EVALUATION_DIR
)

if exists(evaluation_file) and contains_all(
    evaluation_text,
    evaluation_terms
) and evaluation_results:

    add_result(
        "Official model evaluation",
        "PASS",
        (
            "src/04_evaluate_model.py; "
            "results/evaluation"
        ),
        "Evaluation code and final evaluation artifacts are present."
    )

elif exists(evaluation_file):

    add_result(
        "Official model evaluation",
        "PARTIAL",
        "src/04_evaluate_model.py",
        "Evaluation script exists but complete evidence was not automatically confirmed.",
        "Verify accuracy, precision, recall, F1, confusion matrix and confidence outputs."
    )

else:

    add_result(
        "Official model evaluation",
        "MISSING",
        "src/04_evaluate_model.py",
        "Official evaluation script was not found."
    )


# ============================================================
# 12. CONFIDENCE ANALYSIS
# ============================================================

confidence_files = find_files(
    RESULTS_DIR
)

confidence_evidence = []

for path in confidence_files:

    filename = os.path.basename(path).lower()

    text = read_text(path).lower()

    if (
        "confidence" in filename
        or "confidence" in text
    ):

        confidence_evidence.append(
            os.path.relpath(path, BASE_DIR)
        )


if confidence_evidence:

    add_result(
        "Prediction confidence",
        "PASS",
        "; ".join(confidence_evidence[:8]),
        "Confidence-related evidence was found in the final results."
    )

else:

    add_result(
        "Prediction confidence",
        "MISSING",
        "results/",
        "No confidence-analysis evidence detected.",
        "Preserve final confidence metrics and analysis."
    )


# ============================================================
# 13. CONFIDENCE / ESCALATION POLICY
# ============================================================

confidence_script = os.path.join(
    SRC_DIR,
    "06_confidence_escalation.py"
)

confidence_script_text = read_text(
    confidence_script
)

escalation_terms = [
    "HUMAN_ESCALATION",
    "VERIFY_WITH_CUSTOMER",
    "AUTOMATED_RESPONSE",
]

if exists(confidence_script) and contains_all(
    confidence_script_text,
    escalation_terms
):

    add_result(
        "Confidence-based escalation",
        "PASS",
        "src/06_confidence_escalation.py",
        "Automated, verification and human-escalation decisions are represented."
    )

elif exists(confidence_script):

    add_result(
        "Confidence-based escalation",
        "PARTIAL",
        "src/06_confidence_escalation.py",
        "Escalation script exists but all operational decision states were not detected.",
        "Review the escalation policy."
    )

else:

    add_result(
        "Confidence-based escalation",
        "MISSING",
        "src/06_confidence_escalation.py",
        "Escalation script not found."
    )


# ============================================================
# 14. RISK / AMBIGUITY DETECTION
# ============================================================

risk_file = os.path.join(
    SRC_DIR,
    "08b_risk_detection.py"
)

risk_text = read_text(risk_file)

risk_terms = [
    "AMBIGUOUS_LANGUAGE",
    "CLOSE_TOP_PREDICTIONS",
    "LOW_CONFIDENCE",
    "SEMANTIC_SAFETY_GUARD",
]

if exists(risk_file) and contains_all(
    risk_text,
    risk_terms
):

    add_result(
        "Refined risk and ambiguity detection",
        "PASS",
        "src/08b_risk_detection.py",
        "Refined ambiguity, confidence and semantic-safety controls are present."
    )

elif exists(risk_file):

    add_result(
        "Refined risk and ambiguity detection",
        "PARTIAL",
        "src/08b_risk_detection.py",
        "Risk module exists but not all expected safeguards were detected.",
        "Review the final Step 8B implementation."
    )

else:

    add_result(
        "Refined risk and ambiguity detection",
        "MISSING",
        "src/08b_risk_detection.py",
        "Risk detection module not found."
    )


# ============================================================
# 15. GENERATIVE AI
# ============================================================

genai_file = os.path.join(
    SRC_DIR,
    "07_generative_ai.py"
)

genai_text = read_text(
    genai_file
)

genai_terms = [
    "process_customer_query",
    "generated_response",
    "confidence",
]

if exists(genai_file) and contains_all(
    genai_text,
    genai_terms
):

    add_result(
        "Generative AI response",
        "PASS",
        "src/07_generative_ai.py",
        "Generative/support-response pipeline is present."
    )

elif exists(genai_file):

    add_result(
        "Generative AI response",
        "PARTIAL",
        "src/07_generative_ai.py",
        "GenAI module exists but complete response evidence was not automatically confirmed.",
        "Manually verify response generation."
    )

else:

    add_result(
        "Generative AI response",
        "MISSING",
        "src/07_generative_ai.py",
        "Generative AI module not found."
    )


# ============================================================
# 16. SAFETY / HALLUCINATION CONTROL
# ============================================================

app_text = read_text(APP_FILE)

safety_terms = [
    "build_safe_response",
    "safety_check",
    "HUMAN_ESCALATION",
    "VERIFY_WITH_CUSTOMER",
]

if exists(APP_FILE) and contains_all(
    app_text,
    safety_terms
):

    add_result(
        "AI safety / controlled response",
        "PASS",
        "app.py",
        "Safe-response and risk-controlled handling are implemented in the application."
    )

elif exists(APP_FILE):

    add_result(
        "AI safety / controlled response",
        "PARTIAL",
        "app.py",
        "Application exists but complete safety evidence was not detected.",
        "Manually test low-confidence and semantically risky queries."
    )

else:

    add_result(
        "AI safety / controlled response",
        "MISSING",
        "app.py",
        "Final Streamlit application was not found."
    )


# ============================================================
# 17. CONVERSATION INTELLIGENCE / SUPPORT SUMMARY
# ============================================================

conversation_terms = [
    "Support Summary",
    "Customer Issue",
    "Detected Intent",
    "Priority",
    "Final Action",
]

if exists(APP_FILE) and contains_all(
    app_text,
    conversation_terms
):

    add_result(
        "Conversation intelligence / support summary",
        "PASS",
        "app.py",
        "Support summary includes issue, intent, priority and action."
    )

elif exists(APP_FILE):

    add_result(
        "Conversation intelligence / support summary",
        "PARTIAL",
        "app.py",
        "Application exists but complete support-summary evidence was not detected.",
        "Verify the support summary manually."
    )

else:

    add_result(
        "Conversation intelligence / support summary",
        "MISSING",
        "app.py",
        "Application missing."
    )


# ============================================================
# 18. STREAMLIT APPLICATION
# ============================================================

streamlit_terms = [
    "st.set_page_config",
    "st.text_area",
    "st.button",
    "st.tabs",
    "Model Performance Analytics",
]

if exists(APP_FILE) and contains_all(
    app_text,
    streamlit_terms
):

    add_result(
        "Streamlit application",
        "PASS",
        "app.py",
        "Streamlit interface, query input, interaction and analytics tabs detected."
    )

elif exists(APP_FILE):

    add_result(
        "Streamlit application",
        "PARTIAL",
        "app.py",
        "Streamlit application exists but complete UI evidence was not detected.",
        "Run and manually verify the application."
    )

else:

    add_result(
        "Streamlit application",
        "MISSING",
        "app.py",
        "Final Streamlit application missing."
    )


# ============================================================
# 19. STREAMLIT REQUIRED OUTPUTS
# ============================================================

ui_terms = [
    "Predicted Intent",
    "Confidence",
    "Priority",
    "Risk & Escalation Decision",
    "AI Customer Support Response",
    "Support Summary",
    "Model Performance Analytics",
]

if exists(APP_FILE) and contains_all(
    app_text,
    ui_terms
):

    add_result(
        "Required Streamlit outputs",
        "PASS",
        "app.py",
        "Required prediction, confidence, priority, response, escalation, summary and analytics sections detected."
    )

elif exists(APP_FILE):

    add_result(
        "Required Streamlit outputs",
        "PARTIAL",
        "app.py",
        "Some required UI sections were not automatically detected.",
        "Review the Streamlit interface manually."
    )

else:

    add_result(
        "Required Streamlit outputs",
        "MISSING",
        "app.py",
        "Application missing."
    )


# ============================================================
# 20. ML VS TRANSFORMER COMPARISON
# ============================================================

comparison_terms = [
    "TF-IDF + Logistic Regression",
    "DistilBERT",
    "87.79%",
    "88.67%",
]

if exists(APP_FILE) and contains_all(
    app_text,
    comparison_terms
):

    add_result(
        "ML vs Transformer comparison",
        "PASS",
        "app.py",
        "Final application contains the ML baseline vs DistilBERT comparison."
    )

else:

    comparison_files = find_files(
        RESULTS_DIR
    )

    found_comparison = False

    for path in comparison_files:

        text = read_text(path)

        if contains_all(
            text,
            [
                "87.79",
                "88.67",
            ]
        ):

            found_comparison = True
            break

    if found_comparison:

        add_result(
            "ML vs Transformer comparison",
            "PASS",
            "results/",
            "Comparison metrics were found in result artifacts."
        )

    else:

        add_result(
            "ML vs Transformer comparison",
            "MISSING",
            "app.py / results/",
            "Final ML vs Transformer comparison was not detected.",
            "Preserve the final comparison metrics."
        )


# ============================================================
# 21. ERROR ANALYSIS
# ============================================================

error_file = os.path.join(
    SRC_DIR,
    "08_error_analysis.py"
)

error_results = find_files(
    os.path.join(
        RESULTS_DIR,
        "error_analysis"
    )
)

if exists(error_file) and error_results:

    add_result(
        "Error analysis",
        "PASS",
        (
            "src/08_error_analysis.py; "
            "results/error_analysis"
        ),
        "Error-analysis script and output artifacts are present."
    )

elif exists(error_file):

    add_result(
        "Error analysis",
        "PARTIAL",
        "src/08_error_analysis.py",
        "Error-analysis code exists but result artifacts were not found.",
        "Preserve the final error-analysis outputs."
    )

else:

    add_result(
        "Error analysis",
        "MISSING",
        "src/08_error_analysis.py",
        "Error-analysis script missing."
    )


# ============================================================
# 22. DOCUMENTATION
# ============================================================

if exists(README_FILE):

    readme_text = read_text(
        README_FILE
    )

    documentation_terms = [
        "DistilBERT",
        "Streamlit",
        "dataset",
        "evaluation",
        "Project Workflow",
        "Machine Learning Baseline",
        "Deep Learning Model",
        "Prediction Confidence",
        "Escalation Strategy",
        "Generative AI",
        "AI Safety",
        "Project Structure",
        "Installation",
        "Running the Application",
        "Final Results",
        "Limitations",
    ]

    matched = sum(
        1
        for term in documentation_terms
        if term.lower() in readme_text.lower()
    )

    if matched >= 8:

        add_result(
            "Project documentation",
            "PASS",
            "README.md",
            "README contains substantial project documentation."
        )

    else:

        add_result(
            "Project documentation",
            "PARTIAL",
            "README.md",
            "README exists but appears incomplete.",
            "Add workflow, setup, model, evaluation and usage documentation."
        )

else:

    add_result(
        "Project documentation",
        "MISSING",
        "README.md",
        "README file not found.",
        "Create the final project README."
    )


# ============================================================
# 23. REQUIREMENTS FILE
# ============================================================

if exists(REQUIREMENTS_FILE):

    requirements_text = read_text(
        REQUIREMENTS_FILE
    )

    package_terms = [
        "streamlit",
        "torch",
        "transformers",
    ]

    matched = sum(
        1
        for term in package_terms
        if term.lower() in requirements_text.lower()
    )

    if matched >= 2:

        add_result(
            "Environment / requirements",
            "PASS",
            "requirements.txt",
            "Core runtime dependencies are represented."
        )

    else:

        add_result(
            "Environment / requirements",
            "PARTIAL",
            "requirements.txt",
            "requirements.txt exists but core dependencies were not all detected.",
            "Verify the final dependency list."
        )

else:

    add_result(
        "Environment / requirements",
        "MISSING",
        "requirements.txt",
        "requirements.txt missing.",
        "Create the final dependency file."
    )


# ============================================================
# 24. OFFICIAL TEST SET PROTECTION
# ============================================================

evaluation_code = read_text(
    evaluation_file
)

official_test_evidence = (
    "test_clean.csv" in evaluation_code
    or "test.csv" in evaluation_code
)

training_code = read_text(
    bert_file
)

training_uses_official_test = (
    "test_clean.csv" in training_code.lower()
    and (
        "train" in training_code.lower()
    )
)

if official_test_evidence and not training_uses_official_test:

    add_result(
        "Official test-set separation",
        "PASS",
        "03b_deep_learning_exp2.py and 04_evaluate_model.py",
        "Training/evaluation separation appears consistent with the final workflow."
    )

else:

    add_result(
        "Official test-set separation",
        "PARTIAL",
        "Training/evaluation scripts",
        "Automatic check could not conclusively verify test-set protection.",
        "Manually confirm official test.csv was reserved for final evaluation."
    )


# ============================================================
# 25. OPTIONAL DEPLOYMENT
# ============================================================

add_result(
    "Deployment",
    "N/A",
    "Project specification",
    "Deployment is optional according to the project specification."
)


# ============================================================
# 26. CALCULATE OVERALL STATUS
# ============================================================

status_counts = {
    "PASS": 0,
    "PARTIAL": 0,
    "MISSING": 0,
    "N/A": 0,
}

for item in results:

    status = item["status"]

    if status in status_counts:
        status_counts[status] += 1


mandatory_results = [
    item
    for item in results
    if item["status"] != "N/A"
]

mandatory_missing = sum(
    1
    for item in mandatory_results
    if item["status"] == "MISSING"
)

mandatory_partial = sum(
    1
    for item in mandatory_results
    if item["status"] == "PARTIAL"
)

mandatory_pass = sum(
    1
    for item in mandatory_results
    if item["status"] == "PASS"
)


if mandatory_missing > 0:

    overall_status = "NEEDS FIXES"

elif mandatory_partial > 0:

    overall_status = "NEEDS MANUAL VERIFICATION"

else:

    overall_status = "READY"


# ============================================================
# 27. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# 28. JSON REPORT
# ============================================================

audit_report = {
    "project": "AI-Powered Banking Customer Support",
    "audit_type": "Final Project Requirement Audit",
    "generated_at": datetime.now().isoformat(
        timespec="seconds"
    ),
    "project_path": BASE_DIR,
    "overall_status": overall_status,
    "summary": {
        "pass": status_counts["PASS"],
        "partial": status_counts["PARTIAL"],
        "missing": status_counts["MISSING"],
        "not_applicable": status_counts["N/A"],
    },
    "results": results,
}


with open(
    AUDIT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit_report,
        f,
        indent=4
    )


# ============================================================
# 29. TEXT REPORT
# ============================================================

with open(
    AUDIT_TEXT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "AI-POWERED BANKING CUSTOMER SUPPORT\n"
    )

    f.write(
        "FINAL PROJECT REQUIREMENT AUDIT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Overall Status: {overall_status}\n\n"
    )

    f.write(
        "Summary\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"PASS    : {status_counts['PASS']}\n"
    )

    f.write(
        f"PARTIAL : {status_counts['PARTIAL']}\n"
    )

    f.write(
        f"MISSING : {status_counts['MISSING']}\n"
    )

    f.write(
        f"N/A     : {status_counts['N/A']}\n\n"
    )

    for index, item in enumerate(
        results,
        start=1
    ):

        f.write(
            f"{index}. {item['requirement']}\n"
        )

        f.write(
            f"   Status   : {item['status']}\n"
        )

        f.write(
            f"   Evidence : {item['evidence']}\n"
        )

        f.write(
            f"   Details  : {item['details']}\n"
        )

        if item["recommended_action"]:

            f.write(
                f"   Action   : {item['recommended_action']}\n"
            )

        f.write("\n")


# ============================================================
# 30. DISPLAY FINAL AUDIT
# ============================================================

print()
print("=" * 70)
print("AI-POWERED BANKING CUSTOMER SUPPORT")
print("FINAL PROJECT REQUIREMENT AUDIT")
print("=" * 70)

print()
print(f"Project: {BASE_DIR}")

print()
print("-" * 70)

for index, item in enumerate(
    results,
    start=1
):

    print(
        f"{index:02d}. "
        f"{format_status(item['status']):10s} "
        f"{item['requirement']}"
    )

print()
print("=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(
    f"PASS    : {status_counts['PASS']}"
)

print(
    f"PARTIAL : {status_counts['PARTIAL']}"
)

print(
    f"MISSING : {status_counts['MISSING']}"
)

print(
    f"N/A     : {status_counts['N/A']}"
)

print()
print(
    f"OVERALL STATUS: {overall_status}"
)

print()
print("=" * 70)
print("REPORT FILES")
print("=" * 70)

print(
    AUDIT_OUTPUT
)

print(
    AUDIT_TEXT_OUTPUT
)

print()
print("=" * 70)

if overall_status == "READY":

    print(
        "🟢 All automatically verifiable mandatory requirements passed."
    )

elif overall_status == "NEEDS MANUAL VERIFICATION":

    print(
        "🟡 No mandatory requirement is missing, "
        "but some items require manual verification."
    )

else:

    print(
        "🔴 One or more mandatory requirements need attention."
    )

print("=" * 70)
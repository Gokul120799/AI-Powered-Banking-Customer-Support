# ============================================================
# STEP 6 - CONFIDENCE & ESCALATION
# ============================================================
#
# Purpose:
#   Convert the DistilBERT prediction into an operational
#   confidence/escalation decision.
#
# Workflow:
#
#   Customer Query
#        ↓
#   DistilBERT
#        ↓
#   Predicted Intent
#        ↓
#   Confidence
#        ↓
#   ┌───────────────┬────────────────┬──────────────────┐
#   │ HIGH          │ MEDIUM         │ LOW              │
#   │ >= 80%        │ 60% - <80%     │ < 60%            │
#   │               │                │                  │
#   │ Automated     │ Verification   │ Human Escalation │
#   └───────────────┴────────────────┴──────────────────┘
#
# IMPORTANT:
#   - No training is performed.
#   - No model tuning is performed.
#   - Official test.csv is NOT used.
#   - Uses the already selected Experiment 2 model.
#   - Uses the official 77-class label mapping.
#
# NOTE:
#   The 80% / 60% thresholds are the INITIAL operational
#   thresholds used for this project stage. They will be
#   documented and can be calibrated later if required.
#
# ============================================================

import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints_step3_exp2",
    "best_model"
)

LABEL_MAPPING_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "confidence"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. INITIAL CONFIDENCE THRESHOLDS
# ============================================================

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.60

MAX_LENGTH = 64
TOP_K = 3


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 75)
print("STEP 6 - CONFIDENCE & ESCALATION")
print("=" * 75)

print(f"Device:              {DEVICE}")
print(f"Transformer model:   {MODEL_DIR}")
print(f"Label mapping:       {LABEL_MAPPING_FILE}")
print(f"Results directory:   {RESULTS_DIR}")

print("\nInitial thresholds:")
print(f"High confidence:     >= {HIGH_THRESHOLD:.0%}")
print(
    f"Medium confidence:   "
    f"{MEDIUM_THRESHOLD:.0%} - < {HIGH_THRESHOLD:.0%}"
)
print(
    f"Low confidence:      "
    f"< {MEDIUM_THRESHOLD:.0%}"
)

print("=" * 75)


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_DIR):

    raise FileNotFoundError(
        f"\n[ERROR] Model not found:\n{MODEL_DIR}"
    )


if not os.path.exists(LABEL_MAPPING_FILE):

    raise FileNotFoundError(
        f"\n[ERROR] Label mapping not found:\n"
        f"{LABEL_MAPPING_FILE}"
    )


print("\n[OK] Required files are available.")


# ============================================================
# 6. LOAD LABEL MAPPING
# ============================================================

print("\nLoading official label mapping...")

with open(
    LABEL_MAPPING_FILE,
    "r",
    encoding="utf-8"
) as f:

    mapping_data = json.load(f)


if "id2label" not in mapping_data:

    raise KeyError(
        "label_mapping.json does not contain 'id2label'."
    )


id2label = {
    int(k): v
    for k, v in mapping_data["id2label"].items()
}

NUM_CLASSES = len(id2label)

print(
    f"[OK] Number of classes: {NUM_CLASSES}"
)

if NUM_CLASSES != 77:

    raise ValueError(
        f"Expected 77 classes, found {NUM_CLASSES}."
    )


# ============================================================
# 7. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("[OK] Tokenizer loaded.")


# ============================================================
# 8. LOAD MODEL
# ============================================================

print("\nLoading final DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

print("[OK] Final Transformer model loaded.")

print(
    f"[OK] Model classes: "
    f"{model.config.num_labels}"
)


# ============================================================
# 9. MODEL / LABEL VALIDATION
# ============================================================

if model.config.num_labels != NUM_CLASSES:

    raise ValueError(
        "Model and label mapping class counts do not match."
    )


print(
    "[OK] Model and label mapping are consistent."
)


# ============================================================
# 10. CONFIDENCE DECISION FUNCTION
# ============================================================

def confidence_level(confidence):
    """
    Convert probability into confidence level.

    HIGH:
        confidence >= 0.80

    MEDIUM:
        0.60 <= confidence < 0.80

    LOW:
        confidence < 0.60
    """

    confidence = float(confidence)

    if confidence >= HIGH_THRESHOLD:

        return "HIGH"

    elif confidence >= MEDIUM_THRESHOLD:

        return "MEDIUM"

    return "LOW"


# ============================================================
# 11. ESCALATION DECISION
# ============================================================

def escalation_decision(level):
    """
    Convert confidence level into operational action.
    """

    if level == "HIGH":

        return {
            "action": "AUTOMATED_RESPONSE",
            "escalate": False,
            "verification_required": False,
            "description":
                "High-confidence prediction. "
                "Eligible for automated handling."
        }

    elif level == "MEDIUM":

        return {
            "action": "VERIFY_WITH_CUSTOMER",
            "escalate": False,
            "verification_required": True,
            "description":
                "Medium-confidence prediction. "
                "Verify the intent before automated handling."
        }

    else:

        return {
            "action": "HUMAN_ESCALATION",
            "escalate": True,
            "verification_required": False,
            "description":
                "Low-confidence prediction. "
                "Escalate to human support."
        }


# ============================================================
# 12. PREDICT FUNCTION
# ============================================================

def predict_with_escalation(
    text,
    top_k=TOP_K
):
    """
    Predict intent and determine confidence/escalation action.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not isinstance(text, str):

        raise TypeError(
            "Customer query must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Customer query cannot be empty."
        )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**encoded)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

    # --------------------------------------------------------
    # Top-K predictions
    # --------------------------------------------------------

    k = min(
        int(top_k),
        NUM_CLASSES
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=k,
        dim=-1
    )

    top_probabilities = (
        top_probabilities[0]
        .cpu()
        .tolist()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # Main prediction
    # --------------------------------------------------------

    predicted_id = int(
        top_indices[0]
    )

    predicted_category = id2label[
        predicted_id
    ]

    confidence = float(
        top_probabilities[0]
    )

    # --------------------------------------------------------
    # Confidence level
    # --------------------------------------------------------

    level = confidence_level(
        confidence
    )

    # --------------------------------------------------------
    # Escalation action
    # --------------------------------------------------------

    decision = escalation_decision(
        level
    )

    # --------------------------------------------------------
    # Top predictions
    # --------------------------------------------------------

    top_predictions = []

    for rank, (
        class_id,
        probability
    ) in enumerate(
        zip(
            top_indices,
            top_probabilities
        ),
        start=1
    ):

        top_predictions.append(
            {
                "rank": int(rank),
                "class_id": int(class_id),
                "category": id2label[
                    int(class_id)
                ],
                "confidence": round(
                    float(probability),
                    4
                ),
                "confidence_percent": round(
                    float(probability) * 100,
                    2
                )
            }
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "query": text,

        "predicted_class_id":
            predicted_id,

        "predicted_category":
            predicted_category,

        "confidence":
            round(
                confidence,
                4
            ),

        "confidence_percent":
            round(
                confidence * 100,
                2
            ),

        "confidence_level":
            level,

        "action":
            decision["action"],

        "escalate":
            decision["escalate"],

        "verification_required":
            decision[
                "verification_required"
            ],

        "decision_description":
            decision["description"],

        "top_predictions":
            top_predictions
    }

    return result


# ============================================================
# 13. DISPLAY RESULT
# ============================================================

def display_result(result):

    print("\n" + "=" * 75)
    print("CUSTOMER SUPPORT DECISION")
    print("=" * 75)

    print(
        f"Customer Query:\n"
        f"{result['query']}"
    )

    print("\n" + "-" * 75)

    print(
        f"Predicted Category : "
        f"{result['predicted_category']}"
    )

    print(
        f"Class ID           : "
        f"{result['predicted_class_id']}"
    )

    print(
        f"Confidence         : "
        f"{result['confidence_percent']:.2f}%"
    )

    print(
        f"Confidence Level   : "
        f"{result['confidence_level']}"
    )

    print(
        f"Action             : "
        f"{result['action']}"
    )

    print(
        f"Escalate           : "
        f"{result['escalate']}"
    )

    print(
        f"Verification       : "
        f"{result['verification_required']}"
    )

    print(
        f"\nDecision:\n"
        f"{result['decision_description']}"
    )

    print("\n" + "-" * 75)
    print("TOP-3 PREDICTIONS")
    print("-" * 75)

    for item in result["top_predictions"]:

        print(
            f"{item['rank']}. "
            f"{item['category']} "
            f"--> "
            f"{item['confidence_percent']:.2f}%"
        )

    print("=" * 75)


# ============================================================
# 14. SAVE INDIVIDUAL RESULT
# ============================================================

def save_result(result):

    output_file = os.path.join(
        RESULTS_DIR,
        "latest_inference_decision.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\n[OK] Result saved to:\n"
        f"{output_file}"
    )


# ============================================================
# 15. INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print(
        "\nConfidence and escalation system is ready."
    )

    print(
        "\nType a banking customer query."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        try:

            user_query = input(
                "\nCustomer Query: "
            ).strip()

            if user_query.lower() == "exit":

                print(
                    "\nConfidence/escalation session ended."
                )

                break

            if not user_query:

                print(
                    "[WARNING] Please enter a query."
                )

                continue

            result = predict_with_escalation(
                user_query
            )

            display_result(
                result
            )

            save_result(
                result
            )

        except KeyboardInterrupt:

            print(
                "\n\nSession interrupted."
            )

            break

        except Exception as e:

            print(
                f"\n[ERROR] "
                f"{type(e).__name__}: {e}"
            )
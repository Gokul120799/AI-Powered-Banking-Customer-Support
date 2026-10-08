# ============================================================
# STEP 5 - BANKING CUSTOMER SUPPORT INFERENCE
# ============================================================
#
# Purpose:
#   Load the final trained DistilBERT model and predict the
#   banking intent/category for a new customer query.
#
# IMPORTANT:
#   - No training happens in this file.
#   - No model tuning happens in this file.
#   - Official test data is NOT used here.
#   - Uses the official 77-class label mapping.
#
# Input:
#   A customer banking query
#
# Output:
#   - Predicted category
#   - Confidence
#   - Top-3 predictions
#   - Suggested priority
#
# ============================================================

import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. CONFIGURATION
# ============================================================

MAX_LENGTH = 64
TOP_K = 3


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("=" * 75)
print("STEP 5 - BANKING CUSTOMER SUPPORT INFERENCE")
print("=" * 75)

print(f"Device:              {DEVICE}")
print(f"Transformer model:   {MODEL_DIR}")
print(f"Label mapping:       {LABEL_MAPPING_FILE}")
print("=" * 75)


if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"\n[ERROR] Final model not found:\n{MODEL_DIR}\n"
    )


if not os.path.exists(LABEL_MAPPING_FILE):
    raise FileNotFoundError(
        f"\n[ERROR] Label mapping not found:\n{LABEL_MAPPING_FILE}\n"
    )


print("\n[OK] Required files are available.")


# ============================================================
# 5. LOAD OFFICIAL LABEL MAPPING
# ============================================================

print("\nLoading official label mapping...")

with open(LABEL_MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)


# ------------------------------------------------------------
# The preprocessing script stores:
#   label2id
#   id2label
# ------------------------------------------------------------

if "id2label" not in mapping_data:
    raise KeyError(
        "label_mapping.json does not contain 'id2label'."
    )


id2label_raw = mapping_data["id2label"]

# JSON converts dictionary keys to strings.
id2label = {
    int(k): v
    for k, v in id2label_raw.items()
}

NUM_CLASSES = len(id2label)

print(f"[OK] Number of classes: {NUM_CLASSES}")

if NUM_CLASSES != 77:
    raise ValueError(
        f"Expected 77 classes, but found {NUM_CLASSES}."
    )


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("[OK] Tokenizer loaded.")


# ============================================================
# 7. LOAD FINAL DISTILBERT MODEL
# ============================================================

print("\nLoading final DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

print("[OK] Final Transformer model loaded.")
print(f"[OK] Model classes: {model.config.num_labels}")


# ============================================================
# 8. VERIFY MODEL / LABEL MAPPING
# ============================================================

if model.config.num_labels != NUM_CLASSES:
    raise ValueError(
        f"Model has {model.config.num_labels} classes, "
        f"but label mapping has {NUM_CLASSES} classes."
    )

print("[OK] Model and label mapping are consistent.")


# ============================================================
# 9. INFERENCE FUNCTION
# ============================================================

def predict_intent(text, top_k=TOP_K):
    """
    Predict the banking intent/category for a customer query.

    Parameters
    ----------
    text : str
        Customer's banking query.

    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict
        Prediction information.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(text, str):
        raise TypeError("Customer query must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Customer query cannot be empty.")

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
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**encoded)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

    # --------------------------------------------------------
    # Top-k predictions
    # --------------------------------------------------------

    k = min(top_k, NUM_CLASSES)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=k,
        dim=-1
    )

    top_probabilities = top_probabilities[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    # --------------------------------------------------------
    # Main prediction
    # --------------------------------------------------------

    predicted_id = top_indices[0]

    predicted_category = id2label[predicted_id]

    confidence = top_probabilities[0]

    # --------------------------------------------------------
    # Priority suggestion
    #
    # This is an initial inference-stage priority helper.
    # Formal confidence/escalation rules will be handled in
    # Step 6.
    # --------------------------------------------------------

    if confidence >= 0.80:
        priority = "Normal"
    elif confidence >= 0.60:
        priority = "Review"
    else:
        priority = "High Attention"

    # --------------------------------------------------------
    # Build top-k result
    # --------------------------------------------------------

    top_predictions = []

    for rank, (class_id, probability) in enumerate(
        zip(top_indices, top_probabilities),
        start=1
    ):

        top_predictions.append(
            {
                "rank": rank,
                "class_id": int(class_id),
                "category": id2label[class_id],
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
        "predicted_class_id": int(predicted_id),
        "predicted_category": predicted_category,
        "confidence": round(
            float(confidence),
            4
        ),
        "confidence_percent": round(
            float(confidence) * 100,
            2
        ),
        "priority": priority,
        "top_predictions": top_predictions
    }

    return result


# ============================================================
# 10. DISPLAY FUNCTION
# ============================================================

def display_prediction(result):
    """
    Display prediction results in a readable format.
    """

    print("\n" + "=" * 75)
    print("CUSTOMER QUERY")
    print("=" * 75)

    print(result["query"])

    print("\n" + "-" * 75)
    print("PREDICTION")
    print("-" * 75)

    print(
        f"Predicted Category : "
        f"{result['predicted_category']}"
    )

    print(
        f"Class ID           : "
        f"{result['predicted_class_id']}"
    )

    print(
        f"Confidence          : "
        f"{result['confidence_percent']:.2f}%"
    )

    print(
        f"Priority            : "
        f"{result['priority']}"
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
# 11. INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\nModel is ready for inference.")

    print("\nType a banking customer query.")
    print("Type 'exit' to stop.")

    while True:

        try:

            user_query = input(
                "\nCustomer Query: "
            ).strip()

            if user_query.lower() == "exit":
                print("\nInference session ended.")
                break

            if not user_query:
                print("[WARNING] Please enter a query.")
                continue

            result = predict_intent(
                user_query
            )

            display_prediction(result)

        except KeyboardInterrupt:

            print(
                "\n\nInference session interrupted."
            )
            break

        except Exception as e:

            print(
                f"\n[ERROR] {type(e).__name__}: {e}"
            )
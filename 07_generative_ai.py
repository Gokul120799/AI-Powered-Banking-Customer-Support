# ============================================================
# STEP 7 - GENERATIVE AI CUSTOMER RESPONSE
# ============================================================
#
# Purpose:
#   Generate a professional banking-support response using:
#
#       Customer Query
#            ↓
#       DistilBERT Intent
#            ↓
#       Confidence
#            ↓
#       Escalation Decision
#            ↓
#       Safe Response Generation
#
# IMPORTANT:
#   - No model training.
#   - No official test data used.
#   - Uses the existing Experiment 2 model.
#   - Uses the official 77-class label mapping.
#   - Low-confidence cases are NOT treated as reliable intents.
#   - High-confidence cases can receive a category-aware
#     response, but the system avoids inventing account facts.
#
# ============================================================

import os
import json
import re
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
    "generative_ai"
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
# 3. CONFIGURATION
# ============================================================

MAX_LENGTH = 64
TOP_K = 3

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.60


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 75)
print("STEP 7 - GENERATIVE AI CUSTOMER RESPONSE")
print("=" * 75)

print(f"Device:              {DEVICE}")
print(f"Transformer model:   {MODEL_DIR}")
print(f"Label mapping:       {LABEL_MAPPING_FILE}")
print(f"Results directory:   {RESULTS_DIR}")

print("=" * 75)


# ============================================================
# 5. CHECK FILES
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
# 6. LOAD OFFICIAL LABEL MAPPING
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
# 8. LOAD DISTILBERT
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


if model.config.num_labels != NUM_CLASSES:

    raise ValueError(
        "Model and label mapping class counts do not match."
    )


print(
    "[OK] Model and label mapping are consistent."
)


# ============================================================
# 9. INTENT PREDICTION
# ============================================================

def predict_intent(text):
    """
    Run DistilBERT inference and return:
        predicted category
        confidence
        confidence level
        top-3 predictions
    """

    if not isinstance(text, str):

        raise TypeError(
            "Customer query must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Customer query cannot be empty."
        )

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

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    k = min(
        TOP_K,
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

    predicted_id = int(
        top_indices[0]
    )

    confidence = float(
        top_probabilities[0]
    )

    predicted_category = id2label[
        predicted_id
    ]

    if confidence >= HIGH_THRESHOLD:

        confidence_level = "HIGH"

    elif confidence >= MEDIUM_THRESHOLD:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

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

    return {
        "predicted_class_id": predicted_id,
        "predicted_category": predicted_category,
        "confidence": round(
            confidence,
            4
        ),
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "confidence_level": confidence_level,
        "top_predictions": top_predictions
    }


# ============================================================
# 10. ESCALATION DECISION
# ============================================================

def get_escalation_decision(
    confidence_level
):
    """
    Convert confidence level into an operational action.
    """

    if confidence_level == "HIGH":

        return {
            "action": "AUTOMATED_RESPONSE",
            "escalate": False,
            "verification_required": False
        }

    if confidence_level == "MEDIUM":

        return {
            "action": "VERIFY_WITH_CUSTOMER",
            "escalate": False,
            "verification_required": True
        }

    return {
        "action": "HUMAN_ESCALATION",
        "escalate": True,
        "verification_required": False
    }


# ============================================================
# 11. CATEGORY DESCRIPTION
# ============================================================
#
# These descriptions are deliberately conservative.
# They do NOT claim that a particular account transaction
# actually occurred.
#
# ============================================================

CATEGORY_DESCRIPTIONS = {

    "direct_debit_payment_not_recognised":
        "a direct debit payment that the customer does not recognise",

    "lost_or_stolen_phone":
        "a lost or stolen phone",

    "lost_or_stolen_card":
        "a lost or stolen card",

    "card_swallowed":
        "a card retained or swallowed by a cash machine",

    "failed_transfer":
        "a transfer that has failed",

    "Refund_not_showing_up":
        "a refund that has not appeared",

    "unable_to_verify_identity":
        "difficulty completing identity verification",

    "terminate_account":
        "a request or issue related to closing an account",

    "compromised_card":
        "a potentially compromised card",

    "card_delivery_estimate":
        "a question about card delivery timing",

    "supported_cards_and_currencies":
        "questions about supported cards or currencies",

    "card_payment_fee_charged":
        "a fee charged for a card payment"
}


# ============================================================
# 12. NORMALIZE CATEGORY NAME
# ============================================================

def category_description(
    category
):
    """
    Return a safe human-readable description.
    """

    if category in CATEGORY_DESCRIPTIONS:

        return CATEGORY_DESCRIPTIONS[
            category
        ]

    # Convert unknown label into readable text.
    readable = category.replace(
        "_",
        " "
    )

    return readable


# ============================================================
# 13. SAFE RESPONSE GENERATOR
# ============================================================

def generate_response(
    query,
    prediction
):
    """
    Generate a safe customer-support response.

    The response uses:
        - original query
        - predicted category
        - confidence
        - confidence level
        - escalation decision
        - top predictions

    It deliberately does not invent:
        - account balances
        - transaction IDs
        - dates
        - refunds
        - successful/failed transactions
        - personal account information
    """

    category = prediction[
        "predicted_category"
    ]

    confidence = prediction[
        "confidence_percent"
    ]

    confidence_level = prediction[
        "confidence_level"
    ]

    decision = get_escalation_decision(
        confidence_level
    )

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if confidence_level == "LOW":

        response = (
            "I’m sorry you’re experiencing this issue. "
            "I’m not confident enough to determine the "
            "exact banking intent from your message. "
            "To avoid giving you incorrect guidance, "
            "I recommend connecting you with a human "
            "support representative. "
            "If possible, please provide a little more "
            "detail about what happened."
        )

        return {
            "response": response,
            "response_type": "ESCALATION",
            "safe": True
        }

    # --------------------------------------------------------
    # MEDIUM CONFIDENCE
    # --------------------------------------------------------

    if confidence_level == "MEDIUM":

        description = category_description(
            category
        )

        response = (
            f"I understand that your query may be related "
            f"to {description}. "
            f"Before I provide specific guidance, "
            f"could you please confirm that this is the "
            f"issue you are experiencing? "
            f"This will help us avoid giving you "
            f"incorrect information."
        )

        return {
            "response": response,
            "response_type": "VERIFICATION",
            "safe": True
        }

    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    description = category_description(
        category
    )

    response = (
        f"I understand that your query is related to "
        f"{description}. "
        f"I can help you with the next steps. "
    )

    # --------------------------------------------------------
    # Safe category-specific guidance
    # --------------------------------------------------------

    if category == "lost_or_stolen_phone":

        response += (
            "If your phone has been lost or stolen, "
            "please secure access to your banking account "
            "and contact your bank's official support "
            "channel as soon as possible."
        )

    elif category == "lost_or_stolen_card":

        response += (
            "If your card has been lost or stolen, "
            "please use your bank's official card-control "
            "or support channel to secure the card."
        )

    elif category == "card_swallowed":

        response += (
            "If a cash machine has retained your card, "
            "please contact your bank through its official "
            "support channel for assistance."
        )

    elif category == "compromised_card":

        response += (
            "If you believe your card may be compromised, "
            "please secure the card through your bank's "
            "official card-control or support channel "
            "and report any transaction you do not recognise."
        )

    elif category == "failed_transfer":

        response += (
            "Please check the transfer details and your "
            "banking application's transaction status. "
            "If the transfer remains unresolved, contact "
            "your bank's official support channel."
        )

    elif category == "Refund_not_showing_up":

        response += (
            "Please check the transaction or refund status "
            "in your banking application. If the refund "
            "is still not visible, contact the relevant "
            "merchant or your bank's official support "
            "channel."
        )

    elif category == "unable_to_verify_identity":

        response += (
            "Please check the information and verification "
            "steps requested by your banking application. "
            "If verification continues to fail, contact "
            "your bank's official support channel."
        )

    elif category == "terminate_account":

        response += (
            "If you want to close your account, please "
            "follow your bank's official account-closure "
            "process or contact official support."
        )

    elif category == "direct_debit_payment_not_recognised":

        response += (
            "If you do not recognise a direct debit "
            "payment, please review the transaction details "
            "and contact your bank through its official "
            "support channel if the payment is unfamiliar."
        )

    elif category == "card_delivery_estimate":

        response += (
            "Please check your bank's official card-delivery "
            "tracking or delivery information for the latest "
            "status."
        )

    elif category == "supported_cards_and_currencies":

        response += (
            "Please check your bank's official information "
            "for the currently supported cards and currencies."
        )

    elif category == "card_payment_fee_charged":

        response += (
            "Please review the transaction details and "
            "applicable fee information. If the charge "
            "is unclear, contact your bank's official "
            "support channel."
        )

    else:

        response += (
            "Please check the relevant information in your "
            "banking application. If the issue remains "
            "unresolved, contact your bank's official "
            "support channel."
        )

    return {
        "response": response,
        "response_type": "AUTOMATED",
        "safe": True
    }


# ============================================================
# 14. BUILD COMPLETE SUPPORT RESULT
# ============================================================

def process_customer_query(
    query
):
    """
    Complete Step 7 pipeline.
    """

    prediction = predict_intent(
        query
    )

    decision = get_escalation_decision(
        prediction["confidence_level"]
    )

    generated = generate_response(
        query,
        prediction
    )

    result = {

        "query":
            query,

        "predicted_category":
            prediction[
                "predicted_category"
            ],

        "predicted_class_id":
            prediction[
                "predicted_class_id"
            ],

        "confidence":
            prediction[
                "confidence"
            ],

        "confidence_percent":
            prediction[
                "confidence_percent"
            ],

        "confidence_level":
            prediction[
                "confidence_level"
            ],

        "action":
            decision[
                "action"
            ],

        "escalate":
            decision[
                "escalate"
            ],

        "verification_required":
            decision[
                "verification_required"
            ],

        "top_predictions":
            prediction[
                "top_predictions"
            ],

        "generated_response":
            generated[
                "response"
            ],

        "response_type":
            generated[
                "response_type"
            ],

        "safety_check":
            generated[
                "safe"
            ]
    }

    return result


# ============================================================
# 15. DISPLAY RESULT
# ============================================================

def display_result(
    result
):

    print("\n" + "=" * 75)
    print("GENERATIVE AI CUSTOMER SUPPORT RESULT")
    print("=" * 75)

    print(
        f"\nCustomer Query:\n"
        f"{result['query']}"
    )

    print("\n" + "-" * 75)

    print(
        f"Predicted Category : "
        f"{result['predicted_category']}"
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

    print("\n" + "-" * 75)

    print("TOP-3 PREDICTIONS")
    print("-" * 75)

    for item in result[
        "top_predictions"
    ]:

        print(
            f"{item['rank']}. "
            f"{item['category']} "
            f"--> "
            f"{item['confidence_percent']:.2f}%"
        )

    print("\n" + "-" * 75)

    print("GENERATED CUSTOMER RESPONSE")
    print("-" * 75)

    print(
        result[
            "generated_response"
        ]
    )

    print("\n" + "-" * 75)

    print(
        f"Response Type : "
        f"{result['response_type']}"
    )

    print(
        f"Safety Check  : "
        f"{result['safety_check']}"
    )

    print("=" * 75)


# ============================================================
# 16. SAVE RESULT
# ============================================================

def save_result(
    result
):

    output_file = os.path.join(
        RESULTS_DIR,
        "latest_support_response.json"
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
# 17. INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print(
        "\nGenerative AI support system is ready."
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
                    "\nGenerative AI session ended."
                )

                break

            if not user_query:

                print(
                    "[WARNING] Please enter a query."
                )

                continue

            result = process_customer_query(
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
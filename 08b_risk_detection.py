"""
STEP 8B — REFINED RISK & AMBIGUITY DETECTION

Purpose:
- Add a safety layer on top of the existing DistilBERT prediction.
- DO NOT retrain or modify the model.
- Flag queries that should not receive an automatic response.
- Can be imported later by Streamlit.

This is a safety/decision layer, not a replacement for the classifier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = PROJECT_ROOT / "results" / "evaluation" / "transformer_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "risk_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Operational thresholds
# ------------------------------------------------------------
# These are safety thresholds, not claims of model accuracy.
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60

# If top-1 and top-2 are too close, the prediction is ambiguous.
TOP2_MARGIN_THRESHOLD = 0.15

# Very short queries often lack enough context.
SHORT_QUERY_WORDS = 3

# Very long queries may contain multiple issues or details.
LONG_QUERY_WORDS = 80


# ------------------------------------------------------------
# Useful ambiguity / multi-issue language
# ------------------------------------------------------------
# These terms are useful only as supporting risk signals.
# They should NOT automatically block a high-confidence prediction.
AMBIGUOUS_TERMS = {
    "issue",
    "problem",
    "something wrong",
    "what happened",
    "not sure",
    "confused",
}

# Stronger indicators of two separate requests.
MULTI_ISSUE_MARKERS = {
    " also ",
    " as well as ",
    " another issue",
    " another problem",
    " plus ",
    ";",
}


def detect_card_security_risk(query: str) -> bool:
    """Detect lost, stolen, or potentially compromised cards."""
    q = " ".join(str(query).lower().split())

    security_phrases = (
        "lost card",
        "lost my card",
        "lost my debit card",
        "lost my atm card",
        "atm card is missed",
        "atm card is missing",
        "stolen card",
        "stolen my card",
        "someone may be using it",
        "someone is using my card",
        "card compromised",
        "unauthorized card transaction",
        "unauthorised card transaction",
        "suspicious transaction",
    )

    return any(phrase in q for phrase in security_phrases)


# ------------------------------------------------------------
# Known semantic safety guard
# ------------------------------------------------------------
def semantic_safety_guard(query: str, predicted_category: str) -> Optional[str]:
    """
    Safety guard for a known high-risk semantic distinction.

    'debit card' is not automatically the same thing as
    an unrecognized 'direct debit' payment.

    This does not change the model prediction. It only asks for
    verification when the wording and predicted intent may conflict.
    """
    q = query.lower().strip()
    category = predicted_category.lower().strip()

    if (
        "debit card" in q
        and category == "direct_debit_payment_not_recognised"
        and not any(
            phrase in q
            for phrase in [
                "direct debit",
                "don't recognize",
                "do not recognize",
                "not recognise",
                "do not recognise",
                "unrecognized",
                "unrecognised",
            ]
        )
    ):
        return (
            "Possible debit-card vs direct-debit semantic mismatch. "
            "Verify the customer's intended issue before answering."
        )

    return None


# ------------------------------------------------------------
# Risk assessment
# ------------------------------------------------------------
def assess_query_risk(
    query: str,
    top1_category: str,
    top1_confidence: float,
    top2_category: Optional[str] = None,
    top2_confidence: Optional[float] = None,
    top3_category: Optional[str] = None,
    top3_confidence: Optional[float] = None,
) -> Dict:
    """
    Assess operational risk using several signals.

    Important:
    - Does not change model probabilities.
    - Does not retrain the model.
    - Does not claim that a prediction is definitely correct/incorrect.
    """

    query = str(query).strip()
    top1_confidence = float(top1_confidence)

    if top2_confidence is None:
        top2_confidence = 0.0

    top2_confidence = float(top2_confidence)
    margin = top1_confidence - top2_confidence

    words = re.findall(r"\b[\w']+\b", query.lower())
    word_count = len(words)

    q_lower = query.lower()

    flags: List[str] = []
    reasons: List[str] = []

    # 1. Confidence
    if top1_confidence < MEDIUM_CONFIDENCE:
        flags.append("LOW_CONFIDENCE")
        reasons.append("Prediction confidence is below 60%.")
    elif top1_confidence < HIGH_CONFIDENCE:
        flags.append("MEDIUM_CONFIDENCE")
        reasons.append("Prediction confidence is between 60% and 80%.")

    # 2. Top-1 / Top-2 ambiguity
    if top2_category and margin < TOP2_MARGIN_THRESHOLD:
        flags.append("CLOSE_TOP_PREDICTIONS")
        reasons.append(
            f"Top-1/top-2 confidence margin is only {margin:.2%}."
        )

    # 3. Very short query
    if word_count <= SHORT_QUERY_WORDS:
        flags.append("SHORT_QUERY")
        reasons.append(
            f"Query contains only {word_count} word(s), so context may be insufficient."
        )

    # 4. Very long query
    if word_count >= LONG_QUERY_WORDS:
        flags.append("LONG_COMPLEX_QUERY")
        reasons.append(
            f"Query contains {word_count} words and may contain multiple details."
        )

    # 5. Ambiguous language
    # Only treat generic wording as a risk signal when the prediction
    # is not already very strong. This prevents normal queries such as
    # "can you help me activate my card?" from being over-flagged.
    matched_ambiguous = [
        term for term in AMBIGUOUS_TERMS if term in q_lower
    ]
    if matched_ambiguous and top1_confidence < HIGH_CONFIDENCE:
        flags.append("AMBIGUOUS_LANGUAGE")
        reasons.append(
            "Ambiguous wording detected: " + ", ".join(sorted(matched_ambiguous))
        )

    # 6. Multiple-issue indicators
    # Do NOT treat every occurrence of "and" as multiple issues.
    # For example, "I got my card and want to activate it" is one intent.
    matched_multi = [
        marker.strip()
        for marker in MULTI_ISSUE_MARKERS
        if marker in f" {q_lower} "
    ]

    # A semicolon or explicit "another issue/problem" is strong evidence.
    explicit_multi = (
        ";" in q_lower
        or "another issue" in q_lower
        or "another problem" in q_lower
        or " as well as " in q_lower
    )

    # "also" / "plus" are treated as multi-issue only when the query
    # contains multiple banking problem/action signals.
    action_terms = [
        "lost", "stolen", "charged", "payment", "transfer", "card",
        "refund", "cash", "withdraw", "top up", "deposit", "fee",
        "activate", "blocked", "declined", "pending", "not working",
        "not received", "missing", "unknown", "recognise", "recognize",
    ]
    action_signal_count = sum(
        1 for term in action_terms if term in q_lower
    )

    contextual_multi = (
        bool(matched_multi)
        and action_signal_count >= 2
        and (
            " also " in f" {q_lower} "
            or " plus " in f" {q_lower} "
            or " as well as " in f" {q_lower} "
        )
    )

    if explicit_multi or contextual_multi:
        flags.append("POSSIBLE_MULTIPLE_ISSUES")
        reasons.append(
            "The query may contain more than one distinct issue or request."
        )

    
    # 7. Card security takes precedence over model confidence.
    card_security_risk = detect_card_security_risk(query)

    if card_security_risk:
        flags.append("POSSIBLE_CARD_SECURITY_RISK")
        reasons.append(
            "Possible lost, stolen, or compromised card. "
            "Immediate official-channel security handling is required."
        )


    # 7. Known semantic safety guard
    semantic_reason = semantic_safety_guard(query, top1_category)
    if semantic_reason:
        flags.append("SEMANTIC_SAFETY_GUARD")
        reasons.append(semantic_reason)

    # --------------------------------------------------------
    # Final decision
    
    # Security must take precedence over classifier confidence.
    if "POSSIBLE_CARD_SECURITY_RISK" in flags:
        risk_level = "HIGH"
        decision = "HUMAN_ESCALATION"
        escalate = True
        verification_required = False

    elif "LOW_CONFIDENCE" in flags:
        risk_level = "HIGH"
        decision = "HUMAN_ESCALATION"
        escalate = True
        verification_required = False

    elif any(
        flag in flags
        for flag in [
            "SEMANTIC_SAFETY_GUARD",
            "CLOSE_TOP_PREDICTIONS",
            "POSSIBLE_MULTIPLE_ISSUES",
        ]
    ):
        risk_level = "MEDIUM"
        decision = "VERIFY_WITH_CUSTOMER"
        escalate = False
        verification_required = True

    elif "SHORT_QUERY" in flags and top1_confidence < HIGH_CONFIDENCE:
        risk_level = "MEDIUM"
        decision = "VERIFY_WITH_CUSTOMER"
        escalate = False
        verification_required = True

    elif top1_confidence >= HIGH_CONFIDENCE:
        risk_level = "LOW"
        decision = "AUTOMATED_RESPONSE"
        escalate = False
        verification_required = False

    else:
        risk_level = "MEDIUM"
        decision = "VERIFY_WITH_CUSTOMER"
        escalate = False
        verification_required = True

    
    return {
        "query": query,
        "risk_level": risk_level,
        "decision": decision,
        "escalate": escalate,
        "verification_required": verification_required,
        "risk_flags": flags,
        "reasons": reasons,
        "top1_category": top1_category,
        "top1_confidence": top1_confidence,
        "top2_category": top2_category,
        "top2_confidence": top2_confidence,
        "margin": margin,
        "word_count": word_count,
    }


# ------------------------------------------------------------
# Analyze existing official-test predictions
# ------------------------------------------------------------
def analyze_saved_predictions() -> None:
    if not PREDICTIONS_FILE.exists():
        print(f"Prediction file not found: {PREDICTIONS_FILE}")
        print("Run src\\04_evaluate_model.py first.")
        return

    import pandas as pd

    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "text",
        "predicted_category",
        "confidence",
        "top_1_category",
        "top_2_category",
        "top_3_category",
        "top_1_confidence",
        "top_2_confidence",
        "top_3_confidence",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print("Missing required prediction columns:", missing)
        return

    results = []

    for _, row in df.iterrows():
        result = assess_query_risk(
            query=row["text"],
            top1_category=row["predicted_category"],
            top1_confidence=row["confidence"],
            top2_category=row["top_2_category"],
            top2_confidence=row["top_2_confidence"],
            top3_category=row["top_3_category"],
            top3_confidence=row["top_3_confidence"],
        )

        result["true_category"] = row.get("true_category", "")
        result["correct"] = bool(row.get("correct", False))
        results.append(result)

    risk_df = pd.DataFrame(results)

    risk_df.to_csv(
        OUTPUT_DIR / "risk_assessment_test_predictions.csv",
        index=False,
    )

    summary = {
        "prediction_rows_analyzed": int(len(risk_df)),
        "low_risk_automated": int(
            (risk_df["decision"] == "AUTOMATED_RESPONSE").sum()
        ),
        "medium_risk_verification": int(
            (risk_df["decision"] == "VERIFY_WITH_CUSTOMER").sum()
        ),
        "high_risk_escalation": int(
            (risk_df["decision"] == "HUMAN_ESCALATION").sum()
        ),
        "risk_flag_counts": {
            flag: int(
                risk_df["risk_flags"].apply(
                    lambda x, f=flag: f in x
                ).sum()
            )
            for flag in sorted(
                {
                    flag
                    for flags in risk_df["risk_flags"]
                    for flag in flags
                }
            )
        },
        "note": (
            "This is a safety-layer analysis of existing predictions. "
            "No model retraining or parameter tuning was performed."
        ),
    }

    with open(
        OUTPUT_DIR / "risk_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    print("=" * 70)
    print("STEP 8B — REFINED RISK & AMBIGUITY ANALYSIS")
    print("=" * 70)
    print("No model training/retraining will be performed.")
    print(f"Prediction rows analyzed: {len(risk_df):,}")
    print()
    print("DECISION DISTRIBUTION")
    print(
        f"  AUTOMATED_RESPONSE : {summary['low_risk_automated']:,}"
    )
    print(
        f"  VERIFY_WITH_CUSTOMER: {summary['medium_risk_verification']:,}"
    )
    print(
        f"  HUMAN_ESCALATION    : {summary['high_risk_escalation']:,}"
    )
    print()
    print("RISK FLAG COUNTS")
    for flag, count in summary["risk_flag_counts"].items():
        print(f"  {flag:28s}: {count:,}")

    # Show high-confidence predictions that our safety layer flags.
    high_conf_risky = risk_df[
        (risk_df["top1_confidence"] >= HIGH_CONFIDENCE)
        & (risk_df["decision"] != "AUTOMATED_RESPONSE")
    ].sort_values("top1_confidence", ascending=False)

    high_conf_risky.head(50).to_csv(
        OUTPUT_DIR / "high_confidence_risky_queries.csv",
        index=False,
    )

    print()
    print("HIGH-CONFIDENCE QUERIES FLAGGED FOR SAFETY REVIEW")
    if high_conf_risky.empty:
        print("  None")
    else:
        for _, row in high_conf_risky.head(10).iterrows():
            print(
                f"  {row['top1_confidence']:.2%} | "
                f"{row['decision']:22s} | "
                f"{row['query']}"
            )

    print()
    print("=" * 70)
    print("STEP 8B COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("Model was not retrained or modified.")


# ------------------------------------------------------------
# Example live-query tests
# ------------------------------------------------------------
def run_examples() -> None:
    examples = [
        {
            "query": "account locked",
            "top1": "terminate_account",
            "c1": 0.1405,
            "top2": "compromised_card",
            "c2": 0.0618,
            "top3": "topping_up_by_card",
            "c3": 0.0518,
        },
        {
            "query": "i am not getting messages",
            "top1": "unable_to_verify_identity",
            "c1": 0.1516,
            "top2": "lost_or_stolen_phone",
            "c2": 0.1402,
            "top3": "card_delivery_estimate",
            "c3": 0.0898,
        },
        {
            "query": "issue with debit card",
            "top1": "direct_debit_payment_not_recognised",
            "c1": 0.9176,
            "top2": "supported_cards_and_currencies",
            "c2": 0.0067,
            "top3": "card_payment_fee_charged",
            "c3": 0.0067,
        },
        {
            "query": "I don't recognize a direct debit payment from my account",
            "top1": "direct_debit_payment_not_recognised",
            "c1": 0.90,
            "top2": "card_payment_not_recognised",
            "c2": 0.03,
            "top3": "cash_withdrawal_not_recognised",
            "c3": 0.02,
        },
    ]

    print()
    print("=" * 70)
    print("LIVE SAFETY-LAYER EXAMPLES")
    print("=" * 70)

    for item in examples:
        result = assess_query_risk(
            query=item["query"],
            top1_category=item["top1"],
            top1_confidence=item["c1"],
            top2_category=item["top2"],
            top2_confidence=item["c2"],
            top3_category=item["top3"],
            top3_confidence=item["c3"],
        )

        print()
        print("Query:", result["query"])
        print("Prediction:", result["predicted_category"])
        print(f"Confidence: {result['top1_confidence']:.2%}")
        print(f"Top-2 margin: {result['top2_margin']:.2%}")
        print("Risk:", result["risk_level"])
        print("Decision:", result["decision"])
        print("Flags:", ", ".join(result["risk_flags"]) or "None")


if __name__ == "__main__":
    analyze_saved_predictions()
    run_examples()

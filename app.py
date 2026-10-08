# ============================================================
# AI-POWERED BANKING CUSTOMER SUPPORT
# FINAL STREAMLIT APPLICATION - STEP 8C
# ============================================================
# Uses:
#   - Fine-tuned DistilBERT
#   - Official 77-class label mapping
#   - Confidence analysis
#   - Refined risk & ambiguity detection
#   - Confidence-aware escalation
#   - Safe customer-support response
#   - Final official evaluation analytics
#
# IMPORTANT:
# No model training or test-set evaluation is performed here.
# ============================================================

import os
import sys
import json
import html
import inspect
import importlib
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Banking Customer Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #eef4ff 0%,
            #f7f4ff 45%,
            #eefcf8 100%
        );
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1d4ed8 45%,
            #7c3aed 100%
        );
        border-radius: 26px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 15px 40px rgba(37, 99, 235, 0.25);
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 850;
        letter-spacing: -0.8px;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.65;
        opacity: 0.92;
        max-width: 950px;
    }

    .section-title {
        color: #172554;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .info-card {
        background: white;
        border-radius: 19px;
        padding: 1.25rem;
        min-height: 135px;
        border: 1px solid #dbeafe;
        box-shadow: 0 7px 24px rgba(15, 23, 42, 0.08);
    }

    .card-icon {
        font-size: 1.65rem;
        margin-bottom: 0.35rem;
    }

    .card-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .card-value {
        color: #172554;
        font-size: 1.18rem;
        font-weight: 850;
        margin-top: 0.4rem;
        word-break: break-word;
    }

    .query-card {
        background: linear-gradient(135deg, #ffffff, #f7fbff);
        border-radius: 19px;
        padding: 1.3rem 1.5rem;
        border: 1px solid #bfdbfe;
        box-shadow: 0 6px 22px rgba(30, 64, 175, 0.08);
        font-size: 1.05rem;
        line-height: 1.65;
    }

    .response-card {
        background: white;
        border-radius: 20px;
        border-left: 7px solid #2563eb;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.09);
        color: #1e293b;
        font-size: 1.04rem;
        line-height: 1.8;
    }

    .risk-card {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 7px solid #f97316;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        line-height: 1.7;
    }

    .safe-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 7px solid #22c55e;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        line-height: 1.7;
    }

    .badge-high, .badge-medium, .badge-low,
    .priority-normal, .priority-review, .priority-high {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-weight: 800;
    }

    .badge-high, .priority-normal {
        background: #dcfce7;
        color: #166534;
    }

    .badge-medium, .priority-review {
        background: #fef3c7;
        color: #92400e;
    }

    .badge-low, .priority-high {
        background: #fee2e2;
        color: #991b1b;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 45%,
            #312e81 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton > button {
        min-height: 3.2rem;
        border-radius: 15px;
        border: none;
        font-weight: 800;
        font-size: 1rem;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.28);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9);
        color: white;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD AI PIPELINES
# ============================================================

@st.cache_resource
def load_ai_pipeline():
    return importlib.import_module("07_generative_ai")


@st.cache_resource
def load_risk_pipeline():
    return importlib.import_module("08b_risk_detection")


try:
    ai = load_ai_pipeline()
except Exception as e:
    st.error("Unable to load the AI response pipeline.")
    st.exception(e)
    st.stop()


try:
    risk = load_risk_pipeline()
except Exception as e:
    risk = None


# ============================================================
# 5. HELPERS
# ============================================================

def get_priority(confidence_level):
    if confidence_level == "HIGH":
        return "Normal"
    if confidence_level == "MEDIUM":
        return "Review"
    return "High Attention"


def get_priority_badge(priority):
    if priority == "Normal":
        return '<span class="priority-normal">🟢 Normal</span>'
    if priority == "Review":
        return '<span class="priority-review">🟡 Review</span>'
    return '<span class="priority-high">🔴 High Attention</span>'


def get_confidence_badge(level):
    if level == "HIGH":
        return '<span class="badge-high">🟢 HIGH</span>'
    if level == "MEDIUM":
        return '<span class="badge-medium">🟡 MEDIUM</span>'
    return '<span class="badge-low">🔴 LOW</span>'


def load_final_metrics():
    path = os.path.join(EVALUATION_DIR, "final_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def confidence_from_result(result):
    """Return top-1 confidence as 0..1."""
    if "confidence" in result:
        try:
            return float(result["confidence"])
        except Exception:
            pass

    if "confidence_percent" in result:
        try:
            return float(result["confidence_percent"]) / 100.0
        except Exception:
            pass

    return 0.0


def top2_confidence_from_result(result):
    top = result.get("top_predictions", [])
    if len(top) >= 2:
        try:
            return float(top[1].get("confidence", 0.0))
        except Exception:
            pass
    return 0.0


def fallback_refined_risk(query, predicted_category, top1, top2):
    """
    Local safety fallback matching the refined Step 8B policy.
    It is only used if the risk module does not expose a callable
    public assessment function.
    """
    q = query.lower().strip()
    flags = []
    reasons = []

    # Confidence bands
    if top1 < 0.60:
        flags.append("LOW_CONFIDENCE")
        reasons.append("Prediction confidence is below 60%.")
        decision = "HUMAN_ESCALATION"
        risk_level = "HIGH"
    elif top1 < 0.80:
        flags.append("MEDIUM_CONFIDENCE")
        reasons.append("Prediction confidence is between 60% and 80%.")
        decision = "VERIFY_WITH_CUSTOMER"
        risk_level = "MEDIUM"
    else:
        decision = "AUTOMATED_RESPONSE"
        risk_level = "LOW"

    # Close competing predictions
    margin = max(top1 - top2, 0.0)
    if top1 >= 0.80 and margin < 0.10:
        flags.append("CLOSE_TOP_PREDICTIONS")
        reasons.append("The top two intent probabilities are relatively close.")
        decision = "VERIFY_WITH_CUSTOMER"
        risk_level = "MEDIUM"

    # Generic ambiguity only when confidence is below 80%
    ambiguity_terms = [
        "issue", "problem", "something wrong", "not working",
        "what happened", "why", "help me", "help with"
    ]
    if top1 < 0.80 and any(term in q for term in ambiguity_terms):
        flags.append("AMBIGUOUS_LANGUAGE")
        reasons.append("The query uses broad or ambiguous wording.")
        if decision != "HUMAN_ESCALATION":
            decision = "VERIFY_WITH_CUSTOMER"
            risk_level = "MEDIUM"

    # Refined multiple-issue detection
    multi_markers = [
        "also", "as well as", "another issue", "another problem",
        "plus", ";"
    ]
    banking_signals = [
        "transfer", "payment", "card", "cash", "withdraw",
        "refund", "top up", "deposit", "account", "fee",
        "exchange", "verification"
    ]
    signal_count = sum(1 for term in banking_signals if term in q)

    if any(marker in q for marker in multi_markers) and signal_count >= 2:
        flags.append("POSSIBLE_MULTIPLE_ISSUES")
        reasons.append("The query may contain more than one banking issue.")
        decision = "VERIFY_WITH_CUSTOMER"
        risk_level = "MEDIUM"

    # Short-query protection
    if len(q.split()) <= 3 and top1 < 0.80:
        flags.append("SHORT_QUERY")
        reasons.append("The query is very short and may lack context.")
        if decision != "HUMAN_ESCALATION":
            decision = "VERIFY_WITH_CUSTOMER"
            risk_level = "MEDIUM"

    # Refined semantic safety guard
    direct_debit_explicit = (
        "direct debit" in q
        and (
            "don't recognize" in q
            or "do not recognize" in q
            or "not recognize" in q
            or "unrecognized" in q
            or "unauthorized" in q
            or "unknown" in q
        )
    )

    if (
        "debit card" in q
        and predicted_category == "direct_debit_payment_not_recognised"
        and not direct_debit_explicit
    ):
        flags.append("SEMANTIC_SAFETY_GUARD")
        reasons.append(
            "The wording refers to a debit card but does not explicitly "
            "indicate an unrecognized direct debit."
        )
        decision = "VERIFY_WITH_CUSTOMER"
        risk_level = "MEDIUM"

    escalate = decision == "HUMAN_ESCALATION"
    verification = decision == "VERIFY_WITH_CUSTOMER"

    return {
        "risk_level": risk_level,
        "decision": decision,
        "escalate": escalate,
        "verification_required": verification,
        "risk_flags": flags,
        "risk_reasons": reasons,
        "top2_margin": margin,
    }


def assess_risk(query, result):
    """
    Use the project's 08b risk module when it exposes a public
    assessment function. A signature adapter keeps this app robust
    across the exact saved Step 8B implementation.
    """
    top1 = confidence_from_result(result)
    top2 = top2_confidence_from_result(result)
    category = result.get("predicted_category", "")

    if risk is not None:
        candidate_names = [
            "assess_risk",
            "analyze_risk",
            "evaluate_risk",
            "assess_query",
            "analyze_query",
            "process_query",
        ]

        for name in candidate_names:
            fn = getattr(risk, name, None)
            if not callable(fn):
                continue

            try:
                sig = inspect.signature(fn)
                params = sig.parameters

                kwargs = {}

                mapping = {
                    "query": query,
                    "text": query,
                    "customer_query": query,
                    "predicted_category": category,
                    "category": category,
                    "confidence": top1,
                    "top1_confidence": top1,
                    "top2_confidence": top2,
                    "top1": top1,
                    "top2": top2,
                    "top_predictions": result.get("top_predictions", []),
                }

                required_missing = False
                positional = []

                for p in params.values():
                    if p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    if p.name in mapping:
                        kwargs[p.name] = mapping[p.name]
                    elif p.default is inspect.Parameter.empty:
                        required_missing = True
                        break

                if not required_missing:
                    output = fn(**kwargs)
                    if isinstance(output, dict):
                        return output

                # Common positional fallbacks
                if len(params) == 4:
                    output = fn(query, category, top1, top2)
                    if isinstance(output, dict):
                        return output

                if len(params) == 3:
                    output = fn(query, category, top1)
                    if isinstance(output, dict):
                        return output

            except Exception:
                continue

    return fallback_refined_risk(query, category, top1, top2)


def normalize_risk_result(risk_result, result):
    """Ensure UI always receives the fields used below."""
    output = dict(risk_result or {})

    decision = output.get("decision", "")
    if decision not in {
        "AUTOMATED_RESPONSE",
        "VERIFY_WITH_CUSTOMER",
        "HUMAN_ESCALATION",
    }:
        decision = "HUMAN_ESCALATION" if confidence_from_result(result) < 0.60 else (
            "VERIFY_WITH_CUSTOMER"
            if confidence_from_result(result) < 0.80
            else "AUTOMATED_RESPONSE"
        )

    output["decision"] = decision
    output["escalate"] = bool(
        output.get("escalate", decision == "HUMAN_ESCALATION")
    )
    output["verification_required"] = bool(
        output.get(
            "verification_required",
            decision == "VERIFY_WITH_CUSTOMER",
        )
    )
    output["risk_level"] = output.get(
        "risk_level",
        "HIGH"
        if decision == "HUMAN_ESCALATION"
        else "MEDIUM"
        if decision == "VERIFY_WITH_CUSTOMER"
        else "LOW",
    )
    output["risk_flags"] = list(output.get("risk_flags", []))
    output["risk_reasons"] = list(output.get("risk_reasons", []))
    output["top2_margin"] = float(output.get("top2_margin", 0.0))
    return output


def build_safe_response(result, risk_result):
    """
    The risk layer is authoritative for whether the generated
    response may be treated as automated.
    """
    decision = risk_result["decision"]

    if decision == "HUMAN_ESCALATION":
        return (
            "I’m sorry you’re experiencing this. I’m not confident enough "
            "to safely determine the exact banking issue from the information "
            "provided. I recommend routing this query to a human support "
            "specialist for review."
        )

    if decision == "VERIFY_WITH_CUSTOMER":
        return (
            "I want to make sure I understand your issue correctly before "
            "giving you specific next steps. Please confirm the exact "
            "transaction, payment, card issue, or banking service you are "
            "referring to."
        )

    return result.get(
        "generated_response",
        "I can help with your banking support query.",
    )


# ============================================================
# 6. HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🏦 AI-Powered Banking Customer Support
        </div>
        <div class="hero-subtitle">
            Intelligent 77-class banking intent classification using
            fine-tuned DistilBERT, confidence-aware risk detection,
            human escalation, and safe AI-assisted customer support.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:0.4rem 0 1rem 0;">
            <div style="font-size:3.2rem;">🏦</div>
            <div style="font-size:1.3rem;font-weight:850;">
                Banking AI
            </div>
            <div style="font-size:0.82rem;opacity:0.82;margin-top:0.3rem;">
                Customer Support Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 🤖 AI Model")
    st.write("**Model:** DistilBERT")
    st.write("**Classes:** 77")
    st.write("**Task:** Intent Classification")
    st.write("**Runtime:** CPU")

    st.divider()

    st.markdown("### 🚦 Confidence Policy")
    st.markdown("🟢 **HIGH** — ≥ 80%")
    st.markdown("🟡 **MEDIUM** — 60% to < 80%")
    st.markdown("🔴 **LOW** — < 60%")

    st.divider()

    st.markdown("### 🛡️ Risk Layer")
    st.markdown("🟢 **LOW RISK** — automated handling")
    st.markdown("🟡 **MEDIUM RISK** — verify customer")
    st.markdown("🔴 **HIGH RISK** — human escalation")

    st.divider()

    st.markdown("### 🔄 AI Workflow")
    st.markdown(
        """
        Customer Query  
        ↓  
        DistilBERT  
        ↓  
        Intent + Confidence  
        ↓  
        Risk & Ambiguity Detection  
        ↓  
        Escalation Decision  
        ↓  
        Safe AI Response
        """
    )


# ============================================================
# 8. MAIN TABS
# ============================================================

assistant_tab, analytics_tab = st.tabs(
    ["💬 Support Assistant", "📈 Model Analytics"]
)


# ============================================================
# 9. SUPPORT ASSISTANT
# ============================================================

with assistant_tab:
    st.markdown(
        '<div class="section-title">💬 Customer Query</div>',
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Describe the customer's banking issue",
        height=150,
        placeholder=(
            "Example: I don't recognize a direct debit payment "
            "from my account."
        ),
        label_visibility="collapsed",
    )

    analyze = st.button(
        "🔍  Analyze Customer Query",
        use_container_width=True,
    )

    if analyze:
        if not query.strip():
            st.warning("Please enter a banking customer query.")
            st.stop()

        with st.spinner("🧠 Analyzing customer query..."):
            try:
                result = ai.process_customer_query(query.strip())
                risk_result = normalize_risk_result(
                    assess_risk(query.strip(), result),
                    result,
                )
            except Exception as e:
                st.error("An error occurred while processing the query.")
                st.exception(e)
                st.stop()

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📋 Query Submitted</div>',
            unsafe_allow_html=True,
        )

        safe_query = html.escape(str(result.get("query", query.strip())))

        st.markdown(
            f"""
            <div class="query-card">
                <strong>Customer:</strong><br>
                {safe_query}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # FINAL OPERATIONAL DECISION
        # ----------------------------------------------------

        confidence_level = result.get("confidence_level", "LOW")
        priority = get_priority(confidence_level)

        decision = risk_result["decision"]

        if decision == "AUTOMATED_RESPONSE":
            final_action = "AUTOMATED_RESPONSE"
            final_escalate = False
            final_verify = False
        elif decision == "VERIFY_WITH_CUSTOMER":
            final_action = "VERIFY_WITH_CUSTOMER"
            final_escalate = False
            final_verify = True
        else:
            final_action = "HUMAN_ESCALATION"
            final_escalate = True
            final_verify = False

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 AI Classification</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-icon">🎯</div>
                    <div class="card-label">Predicted Intent</div>
                    <div class="card-value">
                        {html.escape(str(result["predicted_category"]))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-icon">📊</div>
                    <div class="card-label">Confidence</div>
                    <div class="card-value">
                        {float(result["confidence_percent"]):.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-icon">🚦</div>
                    <div class="card-label">Confidence Level</div>
                    <div class="card-value">
                        {get_confidence_badge(confidence_level)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-icon">⚡</div>
                    <div class="card-label">Priority</div>
                    <div class="card-value">
                        {get_priority_badge(priority)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # RISK & ESCALATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🛡️ Risk & Escalation Decision</div>',
            unsafe_allow_html=True,
        )

        if decision == "AUTOMATED_RESPONSE":
            st.success(
                "🟢 LOW RISK — controlled automated handling is allowed."
            )
        elif decision == "VERIFY_WITH_CUSTOMER":
            st.warning(
                "🟡 MEDIUM RISK — customer intent should be verified "
                "before automated handling."
            )
        else:
            st.error(
                "🔴 HIGH RISK — human support escalation is recommended."
            )

        d1, d2, d3 = st.columns(3)

        with d1:
            st.write(f"**Final Action:** {final_action}")

        with d2:
            st.write(
                f"**Escalation:** "
                f"{'Required' if final_escalate else 'Not Required'}"
            )

        with d3:
            st.write(
                f"**Verification:** "
                f"{'Required' if final_verify else 'Not Required'}"
            )

        flags = risk_result.get("risk_flags", [])
        reasons = risk_result.get("risk_reasons", [])

        if flags:
            flag_text = ", ".join(flags)
            reason_text = (
                "<br>".join(
                    f"• {html.escape(str(x))}" for x in reasons
                )
                if reasons
                else "No additional explanation was returned."
            )

            st.markdown(
                f"""
                <div class="risk-card">
                    <strong>Risk flags:</strong> {html.escape(flag_text)}
                    <br><br>
                    <strong>Why this was flagged:</strong><br>
                    {reason_text}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="safe-card">
                    <strong>Risk layer:</strong>
                    No additional ambiguity or semantic-safety flags
                    were detected for this query.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Top-3 Intent Predictions</div>',
            unsafe_allow_html=True,
        )

        for item in result.get("top_predictions", [])[:3]:
            left, right = st.columns([5, 1])

            with left:
                st.write(
                    f"**{item['rank']}. {item['category']}**"
                )
                st.progress(
                    min(float(item["confidence"]), 1.0)
                )

            with right:
                st.markdown(
                    f"**{float(item['confidence_percent']):.2f}%**"
                )

        # ----------------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🤖 AI Customer Support Response</div>',
            unsafe_allow_html=True,
        )

        if final_action == "AUTOMATED_RESPONSE":
            st.success("Automated response")
        elif final_action == "VERIFY_WITH_CUSTOMER":
            st.warning("Verification response")
        else:
            st.error("Escalation response")

        final_response = build_safe_response(result, risk_result)
        safe_response = html.escape(str(final_response))

        st.markdown(
            f"""
            <div class="response-card">
                {safe_response}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🛡️ AI Safety Check</div>',
            unsafe_allow_html=True,
        )

        if final_action == "AUTOMATED_RESPONSE" and result.get(
            "safety_check", False
        ):
            st.success(
                "✅ Safety check passed — no customer-specific account "
                "facts were invented."
            )
        elif final_action != "AUTOMATED_RESPONSE":
            st.info(
                "ℹ️ The generated response has been replaced by a "
                "conservative verification/escalation response because "
                "the risk layer did not approve direct automation."
            )
        else:
            st.error("⚠️ Safety check failed.")

        # ----------------------------------------------------
        # SUPPORT SUMMARY
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📝 Support Summary</div>',
            unsafe_allow_html=True,
        )

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.write(f"**Customer Issue:** {result['query']}")
            st.write(
                f"**Detected Intent:** {result['predicted_category']}"
            )
            st.write(f"**Priority:** {priority}")

        with summary_col2:
            st.write(
                f"**Confidence:** {float(result['confidence_percent']):.2f}%"
            )
            st.write(f"**Final Action:** {final_action}")
            st.write(
                f"**Risk Level:** {risk_result['risk_level']}"
            )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander("🔧 View Technical Prediction Details"):
            st.json(
                {
                    "model_result": result,
                    "risk_result": risk_result,
                    "final_action": final_action,
                    "final_escalate": final_escalate,
                    "final_verification": final_verify,
                }
            )


# ============================================================
# 10. MODEL ANALYTICS
# ============================================================

with analytics_tab:
    st.markdown(
        '<div class="section-title">📈 Model Performance Analytics</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "These metrics come from the completed official final test "
        "evaluation. The official test set was not used for model training."
    )

    metrics = load_final_metrics()

    # Completed official final evaluation values
    transformer_accuracy = 88.67
    transformer_f1 = 88.16
    ml_accuracy = 87.79
    ml_f1 = 87.87
    accuracy_improvement = 0.88
    f1_improvement = 0.28

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric("DistilBERT Accuracy", "88.67%")

    with a2:
        st.metric("DistilBERT Weighted F1", "88.16%")

    with a3:
        st.metric("ML Baseline Accuracy", "87.79%")

    with a4:
        st.metric("Accuracy Improvement", "+0.88 pp")

    st.markdown(
        '<div class="section-title">⚖️ ML Baseline vs Transformer</div>',
        unsafe_allow_html=True,
    )

    comparison_data = {
        "Metric": ["Accuracy", "Weighted F1"],
        "TF-IDF + Logistic Regression": ["87.79%", "87.87%"],
        "DistilBERT": ["88.67%", "88.16%"],
        "Improvement": ["+0.88 pp", "+0.28 pp"],
    }

    st.table(comparison_data)

    st.markdown(
        '<div class="section-title">🎯 Confidence Distribution</div>',
        unsafe_allow_html=True,
    )

    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)

    with confidence_col1:
        st.metric("🟢 High Confidence", "57.03%")
        st.progress(0.5703)

    with confidence_col2:
        st.metric("🟡 Medium Confidence", "22.28%")
        st.progress(0.2228)

    with confidence_col3:
        st.metric("🔴 Low Confidence", "20.69%")
        st.progress(0.2069)

    st.markdown(
        '<div class="section-title">📊 Confidence Statistics</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("Mean Confidence", "75.32%")

    with s2:
        st.metric("Median Confidence", "83.27%")

    with s3:
        st.metric("Minimum", "10.86%")

    with s4:
        st.metric("Maximum", "95.86%")

    st.markdown(
        '<div class="section-title">🛡️ Refined Risk Policy</div>',
        unsafe_allow_html=True,
    )

    policy_col1, policy_col2, policy_col3 = st.columns(3)

    with policy_col1:
        st.success(
            """
            **LOW RISK**

            High-confidence predictions can proceed with controlled
            automated handling when no semantic or ambiguity flags are
            triggered.
            """
        )

    with policy_col2:
        st.warning(
            """
            **MEDIUM RISK**

            Verify the customer's intent when confidence is moderate,
            competing intents are close, or a semantic safety rule fires.
            """
        )

    with policy_col3:
        st.error(
            """
            **HIGH RISK**

            Low-confidence queries should be routed to human support.
            """
        )

    st.markdown(
        '<div class="section-title">🧪 Error-Analysis Findings</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The completed official-test error analysis found that the model
        achieved 88.67% accuracy, with 2,730 correct and 349 incorrect
        predictions across 3,079 official test rows.

        The weakest observed class was **virtual_card_not_working**,
        where the model confused examples with related virtual-card
        intents. Other notable confusion areas included
        **why_verify_identity vs verify_my_identity** and
        **get_disposable_virtual_card vs related disposable-card intents**.

        The confidence analysis also showed that high-confidence
        predictions are generally much more reliable than low-confidence
        predictions. However, the application therefore uses the refined
        risk layer as an additional safety check rather than treating
        confidence alone as proof of correctness.
        """
    )

    st.markdown(
        '<div class="section-title">🔐 Semantic Safety Example</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The refined risk layer specifically protects against a known
        semantic mismatch such as **"issue with debit card"** being
        classified with high confidence as
        **direct_debit_payment_not_recognised**.

        In that situation, the application changes the operational
        decision to **VERIFY_WITH_CUSTOMER** instead of blindly sending
        an automated intent-specific response.
        """
    )

    st.markdown(
        '<div class="section-title">ℹ️ Evaluation Note</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The final DistilBERT model achieved 88.67% accuracy and 88.16%
        weighted F1 on the official test set. The TF-IDF + Logistic
        Regression baseline achieved 87.79% accuracy and 87.87% weighted
        F1.

        The 80% / 60% confidence bands are the operational thresholds
        used in this application. The refined Step 8B risk layer adds
        ambiguity, close-prediction, multiple-issue, short-query and
        semantic-safety checks on top of the confidence bands.
        """
    )


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🏦 AI-Powered Banking Customer Support
        &nbsp; • &nbsp;
        DistilBERT 77-Class Intent Classification
        &nbsp; • &nbsp;
        Confidence & Risk-Aware AI
        &nbsp; • &nbsp;
        Safe Customer Assistance
    </div>
    """,
    unsafe_allow_html=True,
)

# AI-Powered Banking Customer Support

## 1. Project Overview

This project develops an AI-powered banking customer support system that automatically understands customer queries, classifies their intent, estimates prediction confidence, determines the appropriate escalation level, and generates a controlled support response.

The system combines:

- Natural Language Processing (NLP)
- Exploratory Data Analysis (EDA)
- TF-IDF + Logistic Regression
- Fine-tuned DistilBERT
- Prediction confidence analysis
- Risk and ambiguity detection
- Confidence-based escalation
- Controlled Generative AI responses
- Conversation intelligence
- Streamlit application
- Model analytics and error analysis

---

## 2. Business Objective

The main objective is to reduce manual effort in banking customer support by automatically understanding customer queries and routing them appropriately.

The system provides:

1. Automatic intent classification
2. Prediction confidence
3. Priority assessment
4. Risk and ambiguity detection
5. Automated response for sufficiently reliable predictions
6. Customer verification for uncertain cases
7. Human escalation for low-confidence or risky cases
8. Controlled AI-generated support responses
9. Support conversation summaries
10. Model performance analytics

---

## 3. Dataset

The project uses the official banking customer-support dataset.

### Dataset

| Dataset | Records | Purpose |
|---|---:|---|
| train.csv | 10,003 | Model development |
| test.csv | 3,080 | Final evaluation |

The dataset contains **77 banking customer-support intent categories**.

The official test dataset is kept separate from model training and is used only for final evaluation.

---

## 4. Project Workflow

```text
Customer Query
      |
      v
Data Preprocessing
      |
      v
Exploratory Data Analysis
      |
      v
TF-IDF + Logistic Regression
      |
      v
Fine-tuned DistilBERT
      |
      v
Intent + Confidence
      |
      v
Risk & Ambiguity Detection
      |
      v
Escalation Decision
      |
      v
Controlled AI Response
      |
      v
Streamlit Application
# ============================================================
# STEP 1: DATA PREPROCESSING
# Project: AI-Powered Banking Customer Support
# ============================================================

import os
import json
import re
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "raw",
    "test.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "reports"
)

TRAIN_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "train_clean.csv"
)

TEST_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "test_clean.csv"
)

LABEL_MAPPING_OUTPUT = os.path.join(
    PROCESSED_DIR,
    "label_mapping.json"
)

STATISTICS_OUTPUT = os.path.join(
    RESULTS_DIR,
    "dataset_statistics.json"
)


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 3. START
# ============================================================

print("=" * 70)
print("STEP 1: DATA PREPROCESSING")
print("AI-Powered Banking Customer Support")
print("=" * 70)


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

print("\n[1/10] Checking dataset files...")

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_PATH}\n\n"
        "Please make sure train.csv is inside:\n"
        "dataset/raw/"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"\nTest file not found:\n{TEST_PATH}\n\n"
        "Please make sure test.csv is inside:\n"
        "dataset/raw/"
    )

print(f"Training file found: {TRAIN_PATH}")
print(f"Test file found:     {TEST_PATH}")


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n[2/10] Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training dataset shape: {train_df.shape}")
print(f"Test dataset shape:     {test_df.shape}")


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[3/10] Checking required columns...")

required_columns = {"text", "category"}

train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

missing_train_columns = required_columns - train_columns
missing_test_columns = required_columns - test_columns

if missing_train_columns:
    raise ValueError(
        f"Training dataset is missing columns: "
        f"{missing_train_columns}"
    )

if missing_test_columns:
    raise ValueError(
        f"Test dataset is missing columns: "
        f"{missing_test_columns}"
    )

print("Required columns found:")
print("  - text")
print("  - category")


# ============================================================
# 7. BASIC DATA QUALITY CHECK
# ============================================================

print("\n[4/10] Performing data quality checks...")

print("\nTraining missing values:")
print(train_df[["text", "category"]].isnull().sum())

print("\nTest missing values:")
print(test_df[["text", "category"]].isnull().sum())

print("\nTraining duplicate rows:", train_df.duplicated().sum())
print("Test duplicate rows:", test_df.duplicated().sum())

print("\nTraining duplicate texts:", train_df["text"].duplicated().sum())
print("Test duplicate texts:", test_df["text"].duplicated().sum())


# ============================================================
# 8. REMOVE INVALID ROWS
# ============================================================

print("\n[5/10] Removing invalid rows...")

original_train_rows = len(train_df)
original_test_rows = len(test_df)

# Remove rows where text/category is missing
train_df = train_df.dropna(
    subset=["text", "category"]
).copy()

test_df = test_df.dropna(
    subset=["text", "category"]
).copy()

# Convert to string
train_df["text"] = train_df["text"].astype(str)
train_df["category"] = train_df["category"].astype(str)

test_df["text"] = test_df["text"].astype(str)
test_df["category"] = test_df["category"].astype(str)

print(
    f"Training rows removed because of missing values: "
    f"{original_train_rows - len(train_df)}"
)

print(
    f"Test rows removed because of missing values: "
    f"{original_test_rows - len(test_df)}"
)


# ============================================================
# 9. TEXT CLEANING FUNCTION
# ============================================================

print("\n[6/10] Cleaning text data...")


def clean_text(text):
    """
    Basic text cleaning.

    Operations:
    1. Convert to lowercase
    2. Remove leading/trailing spaces
    3. Replace multiple whitespace characters with one space
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove leading/trailing spaces
    text = text.strip()

    # Replace multiple spaces/newlines/tabs with one space
    text = re.sub(r"\s+", " ", text)

    return text


train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)


# ============================================================
# 10. REMOVE EMPTY TEXT AFTER CLEANING
# ============================================================

before_empty_removal_train = len(train_df)
before_empty_removal_test = len(test_df)

train_df = train_df[
    train_df["text"].str.len() > 0
].copy()

test_df = test_df[
    test_df["text"].str.len() > 0
].copy()

print(
    "Empty training texts removed:",
    before_empty_removal_train - len(train_df)
)

print(
    "Empty test texts removed:",
    before_empty_removal_test - len(test_df)
)


# ============================================================
# 11. REMOVE EXACT DUPLICATES
# ============================================================

print("\n[7/10] Removing exact duplicate records...")

train_before_duplicates = len(train_df)
test_before_duplicates = len(test_df)

train_df = train_df.drop_duplicates(
    subset=["text", "category"]
).copy()

test_df = test_df.drop_duplicates(
    subset=["text", "category"]
).copy()

print(
    "Training duplicate records removed:",
    train_before_duplicates - len(train_df)
)

print(
    "Test duplicate records removed:",
    test_before_duplicates - len(test_df)
)


# ============================================================
# 12. ADD TEXT LENGTH
# ============================================================

print("\nCalculating text statistics...")

train_df["text_length"] = train_df["text"].str.len()
test_df["text_length"] = test_df["text"].str.len()


# ============================================================
# 13. CHECK CATEGORY CONSISTENCY
# ============================================================

print("\n[8/10] Checking category consistency...")

train_categories = set(train_df["category"].unique())
test_categories = set(test_df["category"].unique())

categories_only_in_test = test_categories - train_categories

if categories_only_in_test:
    raise ValueError(
        "\nERROR: Test dataset contains categories that are "
        "not present in the training dataset:\n"
        f"{sorted(categories_only_in_test)}"
    )

print(f"Number of training categories: {len(train_categories)}")
print(f"Number of test categories:     {len(test_categories)}")

print("Category consistency check: PASSED")


# ============================================================
# 14. CREATE LABEL MAPPING
# ============================================================

print("\n[9/10] Creating label mapping...")

# Sort categories so mapping is reproducible
categories = sorted(train_categories)

label2id = {
    category: index
    for index, category in enumerate(categories)
}

id2label = {
    str(index): category
    for index, category in enumerate(categories)
}

# Add numeric labels
train_df["label"] = train_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)


# Check that no labels are missing
if train_df["label"].isnull().any():
    raise ValueError(
        "Training dataset contains categories without label IDs."
    )

if test_df["label"].isnull().any():
    raise ValueError(
        "Test dataset contains categories without label IDs."
    )

# Convert labels to integers
train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)


# ============================================================
# 15. SAVE LABEL MAPPING
# ============================================================

label_mapping = {
    "num_classes": len(categories),
    "label2id": label2id,
    "id2label": id2label
}

with open(
    LABEL_MAPPING_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        label_mapping,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 16. DATASET STATISTICS
# ============================================================

print("\nGenerating dataset statistics...")

train_category_counts = (
    train_df["category"]
    .value_counts()
    .sort_index()
)

test_category_counts = (
    test_df["category"]
    .value_counts()
    .sort_index()
)

statistics = {
    "training": {
        "rows": int(len(train_df)),
        "columns": int(len(train_df.columns)),
        "num_categories": int(train_df["category"].nunique()),
        "text_length": {
            "minimum": int(train_df["text_length"].min()),
            "maximum": int(train_df["text_length"].max()),
            "mean": float(train_df["text_length"].mean()),
            "median": float(train_df["text_length"].median())
        },
        "category_distribution": {
            str(k): int(v)
            for k, v in train_category_counts.items()
        }
    },

    "test": {
        "rows": int(len(test_df)),
        "columns": int(len(test_df.columns)),
        "num_categories": int(test_df["category"].nunique()),
        "text_length": {
            "minimum": int(test_df["text_length"].min()),
            "maximum": int(test_df["text_length"].max()),
            "mean": float(test_df["text_length"].mean()),
            "median": float(test_df["text_length"].median())
        },
        "category_distribution": {
            str(k): int(v)
            for k, v in test_category_counts.items()
        }
    }
}

with open(
    STATISTICS_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        statistics,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 17. SAVE PROCESSED DATASETS
# ============================================================

print("\nSaving processed datasets...")

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    TEST_OUTPUT,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFINAL DATASET INFORMATION")
print("-" * 70)

print(f"Training rows : {len(train_df)}")
print(f"Test rows     : {len(test_df)}")
print(f"Categories    : {len(categories)}")

print("\nTraining columns:")
print(list(train_df.columns))

print("\nTest columns:")
print(list(test_df.columns))

print("\nFiles created:")

print(f"1. {TRAIN_OUTPUT}")
print(f"2. {TEST_OUTPUT}")
print(f"3. {LABEL_MAPPING_OUTPUT}")
print(f"4. {STATISTICS_OUTPUT}")

print("\nFirst 5 training records:")
print(train_df.head())

print("\nCategory examples:")
for category in categories[:10]:
    print(
        f"  {label2id[category]:>2} -> {category}"
    )

print("\n" + "=" * 70)
print("READY FOR STEP 2: EDA / DATA UNDERSTANDING")
print("=" * 70)
# ============================================================
# EXPERIMENT 2 - HIGH ACCURACY DISTILBERT
# AI-Powered Banking Customer Support
# ============================================================
#
# GOAL:
#   Improve Transformer validation accuracy toward >90%
#
# IMPORTANT:
#   - Official test.csv is NOT used here.
#   - Uses only processed train data.
#   - Full DistilBERT fine-tuning.
#   - Checkpoints are saved separately from Experiment 1.
#   - Training can resume after interruption.
#
# ============================================================

import os
import json
import time
import random
import warnings

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "train_clean.csv"
)

LABEL_MAPPING_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)

# IMPORTANT:
# Separate directory from the old Experiment 1 checkpoint.
CHECKPOINT_ROOT = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints_step3_exp2"
)

RESULTS_ROOT = os.path.join(
    BASE_DIR,
    "results",
    "step3_exp2"
)

BEST_MODEL_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "best_model"
)

LATEST_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "latest_checkpoint"
)

HISTORY_FILE = os.path.join(
    RESULTS_ROOT,
    "training_history.json"
)

METRICS_FILE = os.path.join(
    RESULTS_ROOT,
    "validation_metrics.json"
)

REPORT_FILE = os.path.join(
    RESULTS_ROOT,
    "classification_report.txt"
)

CONFIG_FILE = os.path.join(
    RESULTS_ROOT,
    "experiment_config.json"
)


# ============================================================
# HIGH-ACCURACY SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 64

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

CHECKPOINT_EVERY_STEPS = 100

PATIENCE = 2

NUM_WORKERS = 0


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)


# ============================================================
# 3. SAFETY CHECK
# ============================================================

old_checkpoint_path = os.path.abspath(
    os.path.join(BASE_DIR, "dataset", "checkpoints")
)

new_checkpoint_path = os.path.abspath(
    CHECKPOINT_ROOT
)

if new_checkpoint_path == old_checkpoint_path:
    raise RuntimeError(
        "SAFETY ERROR: Experiment 2 is pointing to the old checkpoint directory."
    )

if os.path.commonpath(
    [new_checkpoint_path, old_checkpoint_path]
) == old_checkpoint_path:

    raise RuntimeError(
        "SAFETY ERROR: Experiment 2 checkpoint directory is inside "
        "the old checkpoint directory."
    )


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("EXPERIMENT 2 - HIGH ACCURACY DISTILBERT")
print("=" * 70)

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max length: {MAX_LENGTH}")
print(f"Checkpoint interval: {CHECKPOINT_EVERY_STEPS} steps")

print("=" * 70)


# ============================================================
# 6. VERIFY INPUT FILES
# ============================================================

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(LABEL_MAPPING_FILE):

    raise FileNotFoundError(
        f"Label mapping not found:\n{LABEL_MAPPING_FILE}"
    )


# ============================================================
# 7. LOAD DATA
# ============================================================

print("\nLoading processed training data...")

df = pd.read_csv(TRAIN_FILE)

print(f"Training data shape: {df.shape}")

required_columns = {
    "text",
    "category",
    "label"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(
    subset=["text", "category", "label"]
).copy()

df["text"] = df["text"].astype(str)

df["label"] = df["label"].astype(int)


# ============================================================
# 8. LOAD OFFICIAL LABEL MAPPING
# ============================================================

with open(
    LABEL_MAPPING_FILE,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)


# label_mapping was generated by Step 1.
# We support either:
#
# {
#     "label2id": {...},
#     "id2label": {...}
# }
#
# or direct mapping structures.

if "id2label" in label_mapping:

    id2label_raw = label_mapping["id2label"]

    id2label = {
        int(k): v
        for k, v in id2label_raw.items()
    }

elif "label2id" in label_mapping:

    label2id_raw = label_mapping["label2id"]

    id2label = {
        int(v): k
        for k, v in label2id_raw.items()
    }

else:

    raise ValueError(
        "Could not find id2label or label2id in label_mapping.json"
    )


label2id = {
    name: idx
    for idx, name in id2label.items()
}

NUM_LABELS = len(id2label)

print(f"Number of classes: {NUM_LABELS}")


# ============================================================
# 9. VERIFY LABEL CONSISTENCY
# ============================================================

data_labels = set(df["label"].unique())

mapping_labels = set(id2label.keys())

if data_labels != mapping_labels:

    print("\nWARNING: Label mismatch detected.")

    print(
        "Labels in data but not mapping:",
        data_labels - mapping_labels
    )

    print(
        "Labels in mapping but not data:",
        mapping_labels - data_labels
    )

    raise ValueError(
        "Training labels and official label mapping are inconsistent."
    )


# ============================================================
# 10. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating stratified train/validation split...")

train_df, val_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# ============================================================
# 11. DATASET CLASS
# ============================================================

class BankingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.texts = dataframe["text"].tolist()

        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, idx):

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
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )

        return item


# ============================================================
# 12. TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 13. CREATE DATASETS
# ============================================================

train_dataset = BankingDataset(
    train_df,
    tokenizer,
    MAX_LENGTH
)

val_dataset = BankingDataset(
    val_df,
    tokenizer,
    MAX_LENGTH
)


# ============================================================
# 14. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# ============================================================
# 15. LOAD MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

model.to(device)


# ============================================================
# 16. FULL MODEL FINE-TUNING
# ============================================================

print("\nFull DistilBERT fine-tuning enabled.")

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 18. TOTAL TRAINING STEPS
# ============================================================

total_steps = (
    len(train_loader) * EPOCHS
)

warmup_steps = int(
    total_steps * WARMUP_RATIO
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Total training steps: {total_steps}")
print(f"Warmup steps:        {warmup_steps}")


# ============================================================
# 19. RESUME VARIABLES
# ============================================================

start_epoch = 1
global_step = 0

best_val_f1 = -1.0
best_val_accuracy = -1.0

epochs_without_improvement = 0

training_history = []


# ============================================================
# 20. RESUME FROM CHECKPOINT
# ============================================================

if os.path.exists(
    LATEST_CHECKPOINT_DIR
):

    checkpoint_model_file = os.path.join(
        LATEST_CHECKPOINT_DIR,
        "pytorch_model.bin"
    )

    checkpoint_state_file = os.path.join(
        LATEST_CHECKPOINT_DIR,
        "training_state.pt"
    )

    if os.path.exists(checkpoint_state_file):

        print("\nExisting Experiment 2 checkpoint found.")

        print(
            "Checkpoint:",
            LATEST_CHECKPOINT_DIR
        )

        try:

            model = DistilBertForSequenceClassification.from_pretrained(
                LATEST_CHECKPOINT_DIR
            )

            model.to(device)

            # Recreate optimizer after loading model.
            optimizer = AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            state = torch.load(
                checkpoint_state_file,
                map_location=device
            )

            if "optimizer_state_dict" in state:

                optimizer.load_state_dict(
                    state["optimizer_state_dict"]
                )

            if "scheduler_state_dict" in state:

                scheduler.load_state_dict(
                    state["scheduler_state_dict"]
                )

            global_step = state.get(
                "global_step",
                0
            )

            start_epoch = state.get(
                "epoch",
                1
            )

            best_val_f1 = state.get(
                "best_val_f1",
                -1.0
            )

            best_val_accuracy = state.get(
                "best_val_accuracy",
                -1.0
            )

            epochs_without_improvement = state.get(
                "epochs_without_improvement",
                0
            )

            print(
                f"Resuming from epoch {start_epoch}"
            )

            print(
                f"Global step: {global_step}"
            )

            print(
                f"Best validation F1: {best_val_f1:.4f}"
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f}"
            )

        except Exception as e:

            print(
                "\nWARNING: Checkpoint could not be resumed."
            )

            print(
                "Reason:",
                str(e)
            )

            print(
                "Starting Experiment 2 from scratch."
            )

            start_epoch = 1
            global_step = 0
            best_val_f1 = -1.0
            best_val_accuracy = -1.0
            epochs_without_improvement = 0


# ============================================================
# 21. SAVE CHECKPOINT FUNCTION
# ============================================================

def save_latest_checkpoint(
    epoch,
    global_step,
    best_val_f1,
    best_val_accuracy,
    epochs_without_improvement
):

    # Save model/tokenizer
    model.save_pretrained(
        LATEST_CHECKPOINT_DIR
    )

    tokenizer.save_pretrained(
        LATEST_CHECKPOINT_DIR
    )

    state = {

        "epoch": epoch,

        "global_step": global_step,

        "best_val_f1": best_val_f1,

        "best_val_accuracy": best_val_accuracy,

        "epochs_without_improvement":
            epochs_without_improvement,

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
    }

    torch.save(
        state,
        os.path.join(
            LATEST_CHECKPOINT_DIR,
            "training_state.pt"
        )
    )

    print(
        f"\n[CHECKPOINT SAVED]"
        f" Epoch: {epoch}"
        f" | Step: {global_step}"
    )


# ============================================================
# 22. VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    avg_loss = total_loss / len(val_loader)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "weighted_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions
    }


# ============================================================
# 23. TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING EXPERIMENT 2 TRAINING")
print("=" * 70)

experiment_start = time.time()


for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    print("\n")
    print("=" * 70)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 70)

    model.train()

    epoch_start = time.time()

    total_train_loss = 0.0

    correct_train = 0

    total_train = 0


    for batch_index, batch in enumerate(
        train_loader,
        start=1
    ):

        optimizer.zero_grad()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        scheduler.step()

        total_train_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct_train += (
            predictions == batch["labels"]
        ).sum().item()

        total_train += (
            batch["labels"].size(0)
        )

        global_step += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % 25 == 0
            or batch_index == len(train_loader)
        ):

            running_loss = (
                total_train_loss / batch_index
            )

            running_accuracy = (
                correct_train / total_train
            )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Step {global_step} | "
                f"Loss {running_loss:.4f} | "
                f"Accuracy {running_accuracy:.4f}"
            )


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            global_step % CHECKPOINT_EVERY_STEPS == 0
        ):

            save_latest_checkpoint(
                epoch=epoch,
                global_step=global_step,
                best_val_f1=best_val_f1,
                best_val_accuracy=best_val_accuracy,
                epochs_without_improvement=
                    epochs_without_improvement
            )


    # ========================================================
    # END OF EPOCH
    # ========================================================

    train_loss = (
        total_train_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    validation = evaluate()

    val_loss = validation["loss"]

    val_accuracy = validation["accuracy"]

    val_precision = validation["precision"]

    val_recall = validation["recall"]

    val_f1 = validation["weighted_f1"]

    val_macro_f1 = validation["macro_f1"]


    epoch_time = (
        time.time() -
        epoch_start
    )


    print("\n")
    print("-" * 70)
    print(f"EPOCH {epoch} RESULTS")
    print("-" * 70)

    print(
        f"Train Loss:        {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:    {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss:   {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
        f" ({val_accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted Precision: {val_precision:.4f}"
    )

    print(
        f"Weighted Recall:    {val_recall:.4f}"
    )

    print(
        f"Weighted F1:        {val_f1:.4f}"
        f" ({val_f1 * 100:.2f}%)"
    )

    print(
        f"Macro F1:           {val_macro_f1:.4f}"
    )

    print(
        f"Epoch Time:         {epoch_time / 60:.2f} minutes"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    epoch_record = {

        "epoch": epoch,

        "global_step": global_step,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "validation_loss": val_loss,

        "validation_accuracy": val_accuracy,

        "validation_precision": val_precision,

        "validation_recall": val_recall,

        "validation_weighted_f1": val_f1,

        "validation_macro_f1": val_macro_f1,

        "epoch_time_minutes":
            epoch_time / 60
    }

    training_history.append(
        epoch_record
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            training_history,
            f,
            indent=4
        )


    # ========================================================
    # BEST MODEL
    # ========================================================

    improved = False


    # Primary criterion = weighted F1
    if val_f1 > best_val_f1:

        improved = True


    if improved:

        best_val_f1 = val_f1

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        print("\n*** NEW BEST MODEL ***")

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

        print(
            f"Best validation F1: "
            f"{best_val_f1 * 100:.2f}%"
        )

        model.save_pretrained(
            BEST_MODEL_DIR
        )

        tokenizer.save_pretrained(
            BEST_MODEL_DIR
        )

        print(
            "Best model saved to:"
        )

        print(
            BEST_MODEL_DIR
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement."
            f" Patience: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )


    # ========================================================
    # END-OF-EPOCH CHECKPOINT
    # ========================================================

    save_latest_checkpoint(
        epoch=epoch + 1,
        global_step=global_step,
        best_val_f1=best_val_f1,
        best_val_accuracy=best_val_accuracy,
        epochs_without_improvement=
            epochs_without_improvement
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement >= PATIENCE
    ):

        print("\n")
        print("=" * 70)
        print("EARLY STOPPING")
        print("=" * 70)

        print(
            "Validation performance has stopped improving."
        )

        break


# ============================================================
# 24. TRAINING COMPLETE
# ============================================================

total_runtime = (
    time.time() -
    experiment_start
)

print("\n")
print("=" * 70)
print("EXPERIMENT 2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Total runtime: "
    f"{total_runtime / 60:.2f} minutes"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation weighted F1: "
    f"{best_val_f1 * 100:.2f}%"
)


# ============================================================
# 25. LOAD BEST MODEL FOR FINAL VALIDATION REPORT
# ============================================================

if os.path.exists(
    BEST_MODEL_DIR
):

    print(
        "\nLoading best model for final validation report..."
    )

    best_model = (
        DistilBertForSequenceClassification
        .from_pretrained(
            BEST_MODEL_DIR
        )
    )

    best_model.to(device)

    model = best_model


# ============================================================
# 26. FINAL VALIDATION
# ============================================================

final_validation = evaluate()

final_accuracy = (
    final_validation["accuracy"]
)

final_precision = (
    final_validation["precision"]
)

final_recall = (
    final_validation["recall"]
)

final_weighted_f1 = (
    final_validation["weighted_f1"]
)

final_macro_f1 = (
    final_validation["macro_f1"]
)


# ============================================================
# 27. CLASSIFICATION REPORT
# ============================================================

target_names = [
    id2label[i]
    for i in range(NUM_LABELS)
]

report = classification_report(
    final_validation["labels"],
    final_validation["predictions"],
    labels=list(range(NUM_LABELS)),
    target_names=target_names,
    zero_division=0
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# 28. SAVE METRICS
# ============================================================

final_metrics = {

    "experiment": "Experiment 2 - High Accuracy DistilBERT",

    "model": MODEL_NAME,

    "num_classes": NUM_LABELS,

    "train_samples": len(train_df),

    "validation_samples": len(val_df),

    "epochs_configured": EPOCHS,

    "batch_size": BATCH_SIZE,

    "max_length": MAX_LENGTH,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "validation_accuracy":
        final_accuracy,

    "validation_precision":
        final_precision,

    "validation_recall":
        final_recall,

    "validation_weighted_f1":
        final_weighted_f1,

    "validation_macro_f1":
        final_macro_f1,

    "baseline_ml_accuracy": 0.8725,

    "experiment_1_accuracy": 0.7165,

    "target_accuracy": 0.90,

    "target_reached":
    bool(final_accuracy >= 0.90),

    "official_test_used": False,

    "best_model_path":
        BEST_MODEL_DIR,

    "latest_checkpoint_path":
        LATEST_CHECKPOINT_DIR
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 29. SAVE EXPERIMENT CONFIG
# ============================================================

experiment_config = {

    "experiment":
        "Experiment 2 - High Accuracy DistilBERT",

    "model":
        MODEL_NAME,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "max_length":
        MAX_LENGTH,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "warmup_ratio":
        WARMUP_RATIO,

    "validation_size":
        VALIDATION_SIZE,

    "random_state":
        RANDOM_STATE,

    "checkpoint_every_steps":
        CHECKPOINT_EVERY_STEPS,

    "early_stopping_patience":
        PATIENCE,

    "full_model_fine_tuning":
        True,

    "official_test_used":
        False,

    "ml_baseline_accuracy":
        0.8725,

    "experiment_1_accuracy":
        0.7165,

    "target_accuracy":
        0.90
}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        experiment_config,
        f,
        indent=4
    )


# ============================================================
# 30. FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EXPERIMENT 2 VALIDATION RESULT")
print("=" * 70)

print(
    f"Accuracy:          "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Precision:         "
    f"{final_precision * 100:.2f}%"
)

print(
    f"Recall:            "
    f"{final_recall * 100:.2f}%"
)

print(
    f"Weighted F1:       "
    f"{final_weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1:          "
    f"{final_macro_f1 * 100:.2f}%"
)

print("-" * 70)

if final_accuracy >= 0.90:

    print(
        "SUCCESS: Validation accuracy is above 90%."
    )

else:

    print(
        "TARGET NOT YET REACHED:"
    )

    print(
        "Validation accuracy is below 90%."
    )

    print(
        "We will analyze the results before changing anything."
    )


print("-" * 70)

print(
    "\nBest model:"
)

print(
    BEST_MODEL_DIR
)

print(
    "\nLatest checkpoint:"
)

print(
    LATEST_CHECKPOINT_DIR
)

print(
    "\nMetrics:"
)

print(
    METRICS_FILE
)

print(
    "\nClassification report:"
)

print(
    REPORT_FILE
)

print(
    "\nOfficial test.csv was NOT used."
)

print("=" * 70)
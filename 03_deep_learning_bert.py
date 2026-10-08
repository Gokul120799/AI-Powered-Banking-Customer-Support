# ============================================================
# STEP 3: CPU-OPTIMIZED DISTILBERT TRANSFORMER
# AI-Powered Banking Customer Support
#
# NEW / SEPARATE EXPERIMENT
#
# OLD CHECKPOINTS:
#   dataset/checkpoints/
#   ^ NEVER TOUCHED BY THIS SCRIPT
#
# NEW CHECKPOINTS:
#   dataset/checkpoints_step3_cpu/
#
# NEW RESULTS:
#   results/step3_cpu/
#
# Features:
#   - CPU optimized
#   - DistilBERT
#   - Stratified train/validation split
#   - Lower sequence length
#   - Frozen lower Transformer layers
#   - Periodic checkpoints
#   - Resume after interruption
#   - Separate best model
#   - Training history
#   - Official test set NOT used
# ============================================================

import os
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

# CPU optimization
MAX_LENGTH = 64
BATCH_SIZE = 16

# IMPORTANT:
# Start with ONE epoch.
# We will inspect runtime/performance before increasing it.
EPOCHS = 1

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01

VALIDATION_SIZE = 0.20

# Save every 100 batches
CHECKPOINT_EVERY_STEPS = 100

# DistilBERT has 6 Transformer layers.
# Freeze first 4.
# Train final 2 + classifier.
FREEZE_LAYERS = 4

# Windows-safe
NUM_WORKERS = 0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ------------------------------------------------------------
# EXISTING SUCCESSFUL DATA
# ------------------------------------------------------------

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "train_clean.csv"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)


# ------------------------------------------------------------
# NEW STEP 3 CHECKPOINT DIRECTORY
#
# IMPORTANT:
# This is intentionally NOT:
# dataset/checkpoints/
# ------------------------------------------------------------

CHECKPOINT_ROOT = os.path.join(
    BASE_DIR,
    "dataset",
    "checkpoints_step3_cpu"
)

BEST_MODEL_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "best_model"
)

LATEST_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "latest_checkpoint"
)


# ------------------------------------------------------------
# NEW STEP 3 RESULTS DIRECTORY
# ------------------------------------------------------------

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "step3_cpu"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "validation_metrics.json"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

CONFIG_PATH = os.path.join(
    RESULTS_DIR,
    "training_config.json"
)


# Create ONLY the new directories
os.makedirs(
    CHECKPOINT_ROOT,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("=" * 75)
print("STEP 3: CPU-OPTIMIZED DISTILBERT")
print("=" * 75)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    print("GPU: Not available")
    print("CPU optimization is enabled.")

print("=" * 75)


# ============================================================
# 5. SHOW EXACT NEW STORAGE LOCATIONS
# ============================================================

print("\nNEW CHECKPOINT LOCATION:")
print(CHECKPOINT_ROOT)

print("\nNEW RESULTS LOCATION:")
print(RESULTS_DIR)

print("\nIMPORTANT:")
print(
    "This script does NOT use dataset/checkpoints/"
)

print(
    "Existing 5.3 GB checkpoints will remain untouched."
)


# ============================================================
# 6. VERIFY REQUIRED INPUT FILES
# ============================================================

print("\nChecking required files...")


if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_PATH}"
    )


if not os.path.exists(LABEL_MAPPING_PATH):

    raise FileNotFoundError(
        f"\nLabel mapping not found:\n"
        f"{LABEL_MAPPING_PATH}"
    )


print(
    f"[OK] {TRAIN_PATH}"
)

print(
    f"[OK] {LABEL_MAPPING_PATH}"
)


# ============================================================
# 7. LOAD OFFICIAL LABEL MAPPING
# ============================================================

print("\nLoading official label mapping...")


with open(
    LABEL_MAPPING_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)


label2id = label_mapping["label2id"]

id2label = {
    int(k): v
    for k, v in label_mapping["id2label"].items()
}


NUM_LABELS = len(label2id)


print(
    f"Number of classes: {NUM_LABELS}"
)


if NUM_LABELS != 77:

    raise ValueError(
        f"Expected 77 classes, found {NUM_LABELS}"
    )


# ============================================================
# 8. LOAD PROCESSED TRAIN DATA
# ============================================================

print("\nLoading processed training data...")


df = pd.read_csv(
    TRAIN_PATH
)


print(
    f"Dataset shape: {df.shape}"
)


required_columns = [
    "text",
    "category",
    "label"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# Remove only rows missing required fields
df = df.dropna(
    subset=[
        "text",
        "category",
        "label"
    ]
).copy()


df["text"] = df["text"].astype(str)

df["category"] = df["category"].astype(str)

df["label"] = df["label"].astype(int)


print(
    f"Usable samples: {len(df)}"
)


# ============================================================
# 9. VERIFY LABELS
# ============================================================

invalid_labels = sorted(
    set(df["label"].unique())
    -
    set(id2label.keys())
)


if invalid_labels:

    raise ValueError(
        "Invalid labels found: "
        f"{invalid_labels}"
    )


# Verify each category maps to one label
mapping_check = (
    df.groupby("category")["label"]
    .nunique()
)


if mapping_check.max() > 1:

    raise ValueError(
        "Category-to-label mapping is inconsistent."
    )


print(
    "[OK] Official 77-class mapping verified."
)


# ============================================================
# 10. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating stratified train/validation split..."
)


train_df, val_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=df["label"]
)


train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)


print(
    f"Training samples:   {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)


# ============================================================
# 11. TOKENIZER
# ============================================================

print(
    "\nLoading tokenizer..."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "[OK] Tokenizer loaded."
)


# ============================================================
# 12. DATASET
# ============================================================

class BankingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length
    ):

        self.texts = dataframe[
            "text"
        ].tolist()

        self.labels = dataframe[
            "label"
        ].tolist()

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        text = self.texts[index]

        label = self.labels[index]


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


        return {
            "input_ids":
                encoding[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }


# ============================================================
# 13. DATASETS
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
# 14. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


print("\nDataLoader information:")

print(
    f"Training batches:   {len(train_loader)}"
)

print(
    f"Validation batches: {len(val_loader)}"
)


# ============================================================
# 15. LOAD DISTILBERT
# ============================================================

print(
    "\nLoading DistilBERT..."
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)


# ============================================================
# 16. FREEZE LOWER LAYERS
# ============================================================

print(
    "\nApplying CPU optimization..."
)


# Freeze embeddings
for parameter in model.distilbert.embeddings.parameters():

    parameter.requires_grad = False


# Freeze first 4 Transformer layers
for layer_index in range(FREEZE_LAYERS):

    for parameter in (
        model
        .distilbert
        .transformer
        .layer[layer_index]
        .parameters()
    ):

        parameter.requires_grad = False


# Count parameters
total_parameters = 0

trainable_parameters = 0


for parameter in model.parameters():

    total_parameters += parameter.numel()

    if parameter.requires_grad:

        trainable_parameters += parameter.numel()


print(
    f"Total parameters:     "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Frozen parameters:    "
    f"{total_parameters - trainable_parameters:,}"
)


model.to(DEVICE)


# ============================================================
# 17. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 18. TRAIN FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    global_step,
    best_f1
):

    model.train()

    total_loss = 0.0

    predictions = []

    labels_all = []

    start_time = time.time()


    for batch_number, batch in enumerate(
        loader,
        start=1
    ):

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        loss = outputs.loss

        logits = outputs.logits


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        batch_predictions = torch.argmax(
            logits,
            dim=1
        )


        predictions.extend(
            batch_predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


        labels_all.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


        global_step += 1


        # ----------------------------------------------------
        # SAVE CHECKPOINT
        # ----------------------------------------------------

        if (
            global_step
            % CHECKPOINT_EVERY_STEPS
            == 0
        ):

            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                best_f1
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_number == 1
            or batch_number % 25 == 0
            or batch_number == len(loader)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            average_loss = (
                total_loss
                / batch_number
            )

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_number}/"
                f"{len(loader)} | "
                f"Step {global_step} | "
                f"Loss {average_loss:.4f} | "
                f"Time {elapsed/60:.1f} min"
            )


    epoch_loss = (
        total_loss
        / len(loader)
    )


    epoch_accuracy = accuracy_score(
        labels_all,
        predictions
    )


    return (
        epoch_loss,
        epoch_accuracy,
        global_step
    )


# ============================================================
# 19. VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    predictions = []

    labels_all = []

    losses = []


    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            losses.append(
                outputs.loss.item()
            )


            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1
            )


            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )


            labels_all.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )


    validation_loss = float(
        np.mean(losses)
    )


    accuracy = accuracy_score(
        labels_all,
        predictions
    )


    precision, recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels_all,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels_all,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    return {

        "loss":
            validation_loss,

        "accuracy":
            float(accuracy),

        "weighted_precision":
            float(precision),

        "weighted_recall":
            float(recall),

        "weighted_f1":
            float(weighted_f1),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "labels":
            labels_all,

        "predictions":
            predictions
    }


# ============================================================
# 20. CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    best_f1
):

    os.makedirs(
        LATEST_CHECKPOINT_DIR,
        exist_ok=True
    )


    # Save model
    model.save_pretrained(
        LATEST_CHECKPOINT_DIR
    )


    # Save tokenizer
    tokenizer.save_pretrained(
        LATEST_CHECKPOINT_DIR
    )


    # Save training state
    torch.save(
        {
            "epoch": epoch,

            "step": step,

            "best_f1": best_f1,

            "optimizer_state_dict":
                optimizer.state_dict(),

            "torch_random_state":
                torch.get_rng_state(),

            "numpy_random_state":
                np.random.get_state(),

            "python_random_state":
                random.getstate()
        },
        os.path.join(
            LATEST_CHECKPOINT_DIR,
            "training_state.pt"
        )
    )


    print(
        "\n[CHECKPOINT SAVED]"
    )

    print(
        f"Epoch: {epoch}"
    )

    print(
        f"Step: {step}"
    )

    print(
        f"Location: "
        f"{LATEST_CHECKPOINT_DIR}"
    )


# ============================================================
# 21. RESUME SUPPORT
# ============================================================

start_epoch = 1

global_step = 0

best_f1 = -1.0

history = []


STATE_PATH = os.path.join(
    LATEST_CHECKPOINT_DIR,
    "training_state.pt"
)


if (
    os.path.isdir(
        LATEST_CHECKPOINT_DIR
    )
    and os.path.exists(
        STATE_PATH
    )
):

    print(
        "\n" + "=" * 75
    )

    print(
        "PREVIOUS STEP 3 CPU CHECKPOINT FOUND"
    )

    print(
        "=" * 75
    )

    print(
        LATEST_CHECKPOINT_DIR
    )


    try:

        state = torch.load(
            STATE_PATH,
            map_location=DEVICE
        )


        # Load saved model
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                LATEST_CHECKPOINT_DIR
            )
        )


        model.to(DEVICE)


        # Recreate optimizer
        optimizer = torch.optim.AdamW(
            filter(
                lambda p:
                    p.requires_grad,
                model.parameters()
            ),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )


        optimizer.load_state_dict(
            state[
                "optimizer_state_dict"
            ]
        )


        previous_epoch = state.get(
            "epoch",
            0
        )

        global_step = state.get(
            "step",
            0
        )

        best_f1 = state.get(
            "best_f1",
            -1.0
        )


        start_epoch = (
            previous_epoch + 1
        )


        print(
            f"Previous epoch: "
            f"{previous_epoch}"
        )

        print(
            f"Previous step: "
            f"{global_step}"
        )

        print(
            f"Previous best F1: "
            f"{best_f1:.4f}"
        )

        print(
            f"Resume epoch: "
            f"{start_epoch}"
        )


    except Exception as error:

        print(
            "\n[WARNING]"
        )

        print(
            "Previous checkpoint exists "
            "but could not be loaded."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "The old checkpoint will NOT be deleted."
        )

        print(
            "A fresh Step 3 run will be started."
        )

else:

    print(
        "\nNo previous Step 3 CPU checkpoint."
    )

    print(
        "Starting a new experiment."
    )


# ============================================================
# 22. SAVE CONFIGURATION
# ============================================================

config = {

    "model_name":
        MODEL_NAME,

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "validation_size":
        VALIDATION_SIZE,

    "freeze_layers":
        FREEZE_LAYERS,

    "checkpoint_every_steps":
        CHECKPOINT_EVERY_STEPS,

    "num_workers":
        NUM_WORKERS,

    "device":
        str(DEVICE),

    "number_of_classes":
        NUM_LABELS,

    "training_samples":
        len(train_df),

    "validation_samples":
        len(val_df),

    "seed":
        SEED,

    "old_checkpoint_directory":
        "NOT USED",

    "new_checkpoint_directory":
        CHECKPOINT_ROOT
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# 23. TRAIN
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STARTING STEP 3 TRAINING"
)

print(
    "=" * 75
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Max length: {MAX_LENGTH}"
)

print(
    f"Frozen layers: {FREEZE_LAYERS}"
)

print(
    f"Checkpoint every: "
    f"{CHECKPOINT_EVERY_STEPS} batches"
)

print(
    f"New checkpoint folder:\n"
    f"{CHECKPOINT_ROOT}"
)

print(
    "=" * 75
)


training_start = time.time()


for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    print(
        "\n" + "=" * 75
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        "=" * 75
    )


    epoch_start = time.time()


    train_loss, train_accuracy, global_step = (
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            global_step=global_step,
            best_f1=best_f1
        )
    )


    print(
        "\nRunning validation..."
    )


    validation = evaluate(
        model=model,
        loader=val_loader,
        device=DEVICE
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    print(
        "\n" + "-" * 75
    )

    print(
        f"Epoch {epoch} Results"
    )

    print(
        "-" * 75
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation['loss']:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation['accuracy']:.4f}"
    )

    print(
        f"Weighted Precision: "
        f"{validation['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall: "
        f"{validation['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{validation['weighted_f1']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{validation['macro_f1']:.4f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} minutes"
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    epoch_history = {

        "epoch":
            epoch,

        "train_loss":
            float(train_loss),

        "train_accuracy":
            float(train_accuracy),

        "validation_loss":
            validation["loss"],

        "validation_accuracy":
            validation["accuracy"],

        "weighted_precision":
            validation["weighted_precision"],

        "weighted_recall":
            validation["weighted_recall"],

        "weighted_f1":
            validation["weighted_f1"],

        "macro_precision":
            validation["macro_precision"],

        "macro_recall":
            validation["macro_recall"],

        "macro_f1":
            validation["macro_f1"],

        "global_step":
            global_step,

        "epoch_time_minutes":
            epoch_time / 60
    }


    history.append(
        epoch_history
    )


    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    current_f1 = (
        validation["weighted_f1"]
    )


    if current_f1 > best_f1:

        best_f1 = current_f1


        print(
            "\n[NEW BEST MODEL]"
        )

        print(
            f"Best Weighted F1: "
            f"{best_f1:.4f}"
        )


        os.makedirs(
            BEST_MODEL_DIR,
            exist_ok=True
        )


        model.save_pretrained(
            BEST_MODEL_DIR
        )


        tokenizer.save_pretrained(
            BEST_MODEL_DIR
        )


        print(
            f"Saved to:\n"
            f"{BEST_MODEL_DIR}"
        )


    # --------------------------------------------------------
    # END-OF-EPOCH CHECKPOINT
    # --------------------------------------------------------

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
        best_f1=best_f1
    )


# ============================================================
# 24. TRAINING COMPLETE
# ============================================================

total_time = (
    time.time()
    - training_start
)


print(
    "\n" + "=" * 75
)

print(
    "STEP 3 TRAINING COMPLETE"
)

print(
    "=" * 75
)

print(
    f"Total time: "
    f"{total_time / 3600:.2f} hours"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)


# ============================================================
# 25. LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)


if not os.path.isdir(
    BEST_MODEL_DIR
):

    raise FileNotFoundError(
        "Best model directory does not exist."
    )


best_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BEST_MODEL_DIR
    )
)

best_model.to(DEVICE)


# ============================================================
# 26. FINAL VALIDATION
# ============================================================

print(
    "Running final validation..."
)


final_results = evaluate(
    model=best_model,
    loader=val_loader,
    device=DEVICE
)


# ============================================================
# 27. SAVE METRICS
# ============================================================

final_metrics = {

    "model":
        MODEL_NAME,

    "dataset":
        "dataset/processed/train_clean.csv",

    "training_samples":
        len(train_df),

    "validation_samples":
        len(val_df),

    "number_of_classes":
        NUM_LABELS,

    "accuracy":
        final_results["accuracy"],

    "weighted_precision":
        final_results[
            "weighted_precision"
        ],

    "weighted_recall":
        final_results[
            "weighted_recall"
        ],

    "weighted_f1":
        final_results[
            "weighted_f1"
        ],

    "macro_precision":
        final_results[
            "macro_precision"
        ],

    "macro_recall":
        final_results[
            "macro_recall"
        ],

    "macro_f1":
        final_results[
            "macro_f1"
        ],

    "device":
        str(DEVICE),

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "frozen_layers":
        FREEZE_LAYERS,

    "best_validation_f1":
        best_f1,

    "official_test_used":
        False
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 28. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    final_results["labels"],
    final_results["predictions"],
    labels=list(
        range(NUM_LABELS)
    ),
    target_names=[
        id2label[i]
        for i in range(NUM_LABELS)
    ],
    zero_division=0
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 3 FINAL SUMMARY"
)

print(
    "=" * 75
)

print(
    f"Transformer: "
    f"{MODEL_NAME}"
)

print(
    f"Classes: "
    f"{NUM_LABELS}"
)

print(
    f"Validation Accuracy: "
    f"{final_metrics['accuracy']:.4f}"
)

print(
    f"Weighted Precision: "
    f"{final_metrics['weighted_precision']:.4f}"
)

print(
    f"Weighted Recall: "
    f"{final_metrics['weighted_recall']:.4f}"
)

print(
    f"Weighted F1: "
    f"{final_metrics['weighted_f1']:.4f}"
)

print(
    f"Macro F1: "
    f"{final_metrics['macro_f1']:.4f}"
)


print("\nNEW FILE LOCATIONS:")

print(
    f"\nBest model:\n"
    f"{BEST_MODEL_DIR}"
)

print(
    f"\nLatest checkpoint:\n"
    f"{LATEST_CHECKPOINT_DIR}"
)

print(
    f"\nTraining history:\n"
    f"{HISTORY_PATH}"
)

print(
    f"\nValidation metrics:\n"
    f"{METRICS_PATH}"
)

print(
    f"\nClassification report:\n"
    f"{REPORT_PATH}"
)


print(
    "\nOfficial test.csv was NOT used."
)

print(
    "Official test evaluation will be performed "
    "in Step 4."
)

print(
    "\nOLD dataset/checkpoints/ was NOT used "
    "or modified."
)

print(
    "=" * 75
)
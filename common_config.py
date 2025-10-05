# -*- coding: utf-8 -*-
import os

DATA_ROOT = r"C:\Users\bbuser\Desktop\banana_dataset"

# CSV files
TRAIN_CSV = os.path.join(DATA_ROOT, "train", "_classes.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid", "_classes.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test", "_classes.csv")

# Processed images
PROCESSED_ROOT = os.path.join(DATA_ROOT, "processed")
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_ROOT, "train")
PROCESSED_VALID_DIR = os.path.join(PROCESSED_ROOT, "valid")
PROCESSED_TEST_DIR  = os.path.join(PROCESSED_ROOT, "test")

# Models
MODEL_DIR = os.path.join(DATA_ROOT, "models")
MODEL_KERAS = os.path.join(MODEL_DIR, "banana_quality_model.keras")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "banana_quality.weights.h5")  # fixed naming

# Database
SQLITE_DB = os.path.join(DATA_ROOT, "produce_quality.db")

# Classes
CLASS_COLS = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]

# Freshness mapping
FRESHNESS_MAP = {
    "freshripe": 9.5,
    "ripe": 8.5,
    "freshunripe": 6.5,
    "unripe": 5.0,
    "overripe": 4.0,
    "rotten": 1.5
}

# Training hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 5
LEARNING_RATE = 1e-4
SEED = 42

# -*- coding: utf-8 -*-
"""
Evaluate trained model on test dataset and insert results into SQLite database.
Compatible with MobileNetV2 dual-output training (class_output + freshness_output).
"""

import os, sqlite3, hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# ✅ Import constants from training
# ============================================================
from common_config import (
    DATA_ROOT,
    SQLITE_DB,
    MODEL_KERAS,
    TEST_CSV,
    CLASS_COLS,
    IMAGE_SIZE
)

# ============================================================
# 🔧 Register custom losses so model loads without retraining
# ============================================================
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def banana_loss(y_true, y_pred):
    """Dummy replacement for banana_loss used in training"""
    return categorical_crossentropy(y_true, y_pred)

@register_keras_serializable(package="Custom")
def weighted_categorical_crossentropy(y_true, y_pred):
    """Fallback if training used weighted loss"""
    return categorical_crossentropy(y_true, y_pred)

custom_objects = {
    "banana_loss": banana_loss,
    "weighted_categorical_crossentropy": weighted_categorical_crossentropy
}

# ============================================================
# 1. Database Schema Setup
# ============================================================
def ensure_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS Produce_Samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            item_name TEXT,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT
        );

        CREATE TABLE IF NOT EXISTS Quality_Results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            quality_class TEXT,
            confidence REAL,
            freshness_index REAL,
            FOREIGN KEY(sample_id) REFERENCES Produce_Samples(sample_id)
        );

        CREATE TABLE IF NOT EXISTS Shelf_Life_Metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            predicted_storage_days INTEGER,
            optimal_temp_C REAL,
            mock_decay_rate REAL,
            FOREIGN KEY(sample_id) REFERENCES Produce_Samples(sample_id)
        );
    """)
    conn.commit()

# ============================================================
# 2. Image Loader
# ============================================================
def load_img(path, size):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# ============================================================
# 3. Path Builder
# ============================================================
def build_paths(df):
    df.columns = df.columns.str.strip().str.lower()
    if "image_path_processed" in df.columns:
        return df["image_path_processed"].values
    test_dir = os.path.join(DATA_ROOT, "test")
    return df["filename"].apply(lambda n: os.path.join(test_dir, n)).values

# ============================================================
# 4. File Hash
# ============================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ============================================================
# 5. Main Evaluation Logic
# ============================================================
def main():
    print("\n============================================================")
    print("🔍 Loading trained model...")
    print("============================================================")

    # ✅ Load safely with custom losses
    model = load_model(MODEL_KERAS, safe_mode=False, custom_objects=custom_objects)
    print(f"✅ Model loaded successfully from {MODEL_KERAS}")

    # ============================================================
    # Load CSV
    # ============================================================
    csv_path = TEST_CSV.replace(".csv", "_processed.csv")
    if not os.path.exists(csv_path):
        csv_path = TEST_CSV

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"✅ Loaded test CSV: {len(df)} samples")

    # ============================================================
    # Build images tensor
    # ============================================================
    paths = build_paths(df)
    X = tf.stack([load_img(p, IMAGE_SIZE) for p in paths])
    print(f"✅ Prepared {len(paths)} images for inference")

    # ============================================================
    # Predict
    # ============================================================
    print("\n============================================================")
    print("🧠 Running predictions on test set...")
    print("============================================================")

    outputs = model.predict(X, batch_size=32, verbose=1)

    if isinstance(outputs, list) and len(outputs) == 2:
        class_probs, fresh_vals = outputs
    else:
        raise ValueError("Model output does not match expected (class_output, freshness_output) format.")

    fresh_vals = fresh_vals.reshape(-1)
    y_true = df[CLASS_COLS].values.argmax(axis=1)
    y_pred = class_probs.argmax(axis=1)

    # ============================================================
    # Evaluation Metrics
    # ============================================================
    print("\n============================================================")
    print("📊 Test Metrics Summary")
    print("============================================================")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    print("\nDetailed classification report:\n")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=CLASS_COLS))

    # ============================================================
    # Database Insertion
    # ============================================================
    conn = sqlite3.connect(SQLITE_DB)
    ensure_schema(conn)
    cur = conn.cursor()

    for i, p in enumerate(paths):
        h = sha256(p)
        cur.execute(
            "INSERT OR IGNORE INTO Produce_Samples (file_hash, item_name, image_path) VALUES (?,?,?)",
            (h, "Banana", p)
        )

        cur.execute("SELECT sample_id FROM Produce_Samples WHERE file_hash=?", (h,))
        sid = cur.fetchone()[0]

        cls_idx = int(y_pred[i])
        cls_name = CLASS_COLS[cls_idx]
        conf = float(class_probs[i, cls_idx])
        fresh = float(fresh_vals[i])

        cur.execute(
            "INSERT INTO Quality_Results (sample_id, quality_class, confidence, freshness_index) VALUES (?,?,?,?)",
            (sid, cls_name, conf, fresh)
        )

        predicted_days = int(max(1, round(fresh / 10.0 * 7)))
        cur.execute(
            "INSERT INTO Shelf_Life_Metrics (sample_id, predicted_storage_days, optimal_temp_C, mock_decay_rate) VALUES (?,?,?,?)",
            (sid, predicted_days, 12.0, round(max(0.1, 1.1 - fresh / 10.0), 3))
        )

    conn.commit()
    conn.close()

    print("\n============================================================")
    print(f"✅ Inserted {len(paths)} predictions into SQLite DB → {SQLITE_DB}")
    print("============================================================")

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()

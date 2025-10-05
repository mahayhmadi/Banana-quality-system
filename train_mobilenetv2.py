# -*- coding: utf-8 -*-
# Optimized MobileNetV2 for Banana Ripeness Classification
# Specialized for single-fruit classification with speed optimizations
import os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from common_config import (
    DATA_ROOT, MODEL_DIR, MODEL_KERAS, MODEL_WEIGHTS, IMAGE_SIZE, BATCH_SIZE,
    EPOCHS_WARMUP, EPOCHS_FINETUNE, LEARNING_RATE, SEED, TRAIN_CSV, VALID_CSV, TEST_CSV, CLASS_COLS, FRESHNESS_MAP
)

os.makedirs(MODEL_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

def processed_or_raw(csv_path):
    cand = csv_path.replace(".csv", "_processed.csv")
    return cand if os.path.exists(cand) else csv_path

def load_csv(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip().str.lower(); return df

train_df = load_csv(processed_or_raw(TRAIN_CSV))
valid_df = load_csv(processed_or_raw(VALID_CSV))
test_df  = load_csv(processed_or_raw(TEST_CSV))

IMG_COL = "image_path_processed" if "image_path_processed" in train_df.columns else "filename"
NUM_CLASSES = len(CLASS_COLS); IMG_SIZE = IMAGE_SIZE

def computed_freshness(row):
    val = 0.0
    for c in CLASS_COLS: val += float(row.get(c, 0.0)) * float(FRESHNESS_MAP.get(c, 5.0))
    return float(np.clip(val, 0.0, 10.0))

for df in (train_df, valid_df, test_df):
    if "freshness_index" not in df.columns:
        df["freshness_index"] = df.apply(computed_freshness, axis=1)

def build_paths(df, split):
    if IMG_COL == "filename":
        return df[IMG_COL].apply(lambda n: os.path.join(DATA_ROOT, split, n)).values
    return df[IMG_COL].values

# Banana-optimized preprocessing (emphasizes color channels)
def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method='bilinear', antialias=True)
    img = tf.cast(img, tf.float32)
    # Use ImageNet preprocessing for transfer learning
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Banana-specific augmentation (color is KEY for ripeness!)
@tf.function
def augment_image(img):
    # Geometric - bananas can appear at any angle
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    
    # Random zoom/crop (different banana sizes and perspectives)
    crop_size = tf.random.uniform([], 0.85, 1.0)
    img = tf.image.central_crop(img, crop_size)
    img = tf.image.resize(img, IMG_SIZE)
    
    # Random rotation using transpose operations
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    img = tf.image.resize(img, IMG_SIZE)  # Ensure consistent size
    
    # CRITICAL: Color augmentation for banana ripeness
    # Brightness (lighting conditions vary)
    img = tf.image.random_brightness(img, 0.25)
    
    # Contrast (helps distinguish spots/bruises)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    
    # Saturation (green->yellow->brown transition)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    
    # Hue (critical for ripeness stages)
    img = tf.image.random_hue(img, 0.05)  # Keep small to preserve color accuracy
    
    # Slight noise (camera sensor variation)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.015)
    add_noise = tf.cast(tf.random.uniform([]) > 0.8, tf.float32)
    img = img + noise * add_noise
    
    # Random shadow simulation (uneven lighting)
    shadow_strength = tf.random.uniform([], 0.6, 1.0)
    apply_shadow = tf.cast(tf.random.uniform([]) > 0.7, tf.float32)
    img = img * (shadow_strength * apply_shadow + (1.0 - apply_shadow))
    
    # Random jpeg quality simulation (compression artifacts)
    if tf.random.uniform([]) > 0.8:
        quality = tf.random.uniform([], 75, 100, dtype=tf.int32)
        img = tf.image.adjust_jpeg_quality(img, quality)
    
    return img

def make_ds(df, split, training=False):
    x_paths = build_paths(df, split)
    y1 = df[CLASS_COLS].values.astype("float32")
    y2 = df["freshness_index"].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((x_paths, y1, y2))
    
    def _map(p, c, f):
        img = decode_image(p)
        if training:
            img = augment_image(img)
        return img, {"class_output": c, "freshness_output": tf.expand_dims(f, -1)}
    
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    
    if training: 
        ds = ds.shuffle(2048, seed=SEED)  # Reduced buffer for speed
        ds = ds.repeat()
    
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(valid_df) // BATCH_SIZE

train_ds = make_ds(train_df, "train", True)
valid_ds = make_ds(valid_df, "valid", False)

# Build simplified model for speed
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# Simpler architecture for speed
x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Classification head
class_logits = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

# Freshness regression head
fresh = layers.Dense(64, activation='relu')(x)
fresh = layers.Dropout(0.2)(fresh)
fresh = layers.Dense(1, activation="sigmoid")(fresh)
fresh = layers.Lambda(lambda z: z*10.0, name="freshness_output")(fresh)

model = models.Model(inputs=base.input, outputs=[class_logits, fresh])

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc", multi_label=True),
]

# Class weights for 6-stage banana classification
# These stages have natural progression: unripe→freshunripe→freshripe→ripe→overripe→rotten
class_weights_dict = None
class_weights_tensor = None

if len(train_df) > 0:
    class_dist = train_df[CLASS_COLS].values.sum(axis=0)
    total = class_dist.sum()
    class_weights_dict = {}
    weights_list = []
    
    print(f"\nComputing class weights for imbalanced stages:")
    for i, (col, weight) in enumerate(zip(CLASS_COLS, class_dist)):
        if weight > 0:
            # Moderate balancing - not too aggressive for sequential stages
            w = (total / (len(class_dist) * weight)) ** 0.6
            class_weights_dict[i] = w
            weights_list.append(w)
            print(f"  {col:.<15} count={int(weight):>4} → weight={w:.3f}")
        else:
            class_weights_dict[i] = 1.0
            weights_list.append(1.0)
            print(f"  {col:.<15} count={int(weight):>4} → weight=1.000 (no samples!)")
    
    # Convert to tensor for use in loss function
    class_weights_tensor = tf.constant(weights_list, dtype=tf.float32)
    print()

# Label smoothing for better generalization
def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / NUM_CLASSES
    y_true = y_true * confidence + smoothing_value
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Focal loss for hard examples (misclassified bananas)
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    fl = weight * ce
    return tf.reduce_sum(fl, axis=-1)

# Simplified loss for speed
def banana_loss(y_true, y_pred):
    # Base categorical crossentropy with label smoothing
    base_loss = label_smoothing_loss(y_true, y_pred, smoothing=0.1)
    
    # Apply class weights if available
    if class_weights_tensor is not None:
        weights = tf.reduce_sum(y_true * class_weights_tensor, axis=-1)
        base_loss = base_loss * weights
    
    return base_loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE*0.5),  # Slower start
    loss={
        "class_output": banana_loss,
        "freshness_output": "mse"
    },
    loss_weights={"class_output": 1.0, "freshness_output": 0.1},  # Focus on classification
    metrics={"class_output": METRICS, "freshness_output": ["mae"]}
)

# Callbacks
ckpt = ModelCheckpoint(MODEL_WEIGHTS, monitor="val_class_output_accuracy", mode="max",
                      save_best_only=True, save_weights_only=True, verbose=1)

es = EarlyStopping(monitor="val_class_output_accuracy", mode="max", patience=10,
                  restore_best_weights=True, verbose=1, min_delta=0.0005)

rlrop = ReduceLROnPlateau(monitor="val_class_output_accuracy", mode="max",
                          factor=0.25, patience=3, verbose=1, min_lr=1e-8)

# Cosine annealing with warmup
def lr_schedule(epoch, lr):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return LEARNING_RATE * 0.1 * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(1, EPOCHS_WARMUP - warmup_epochs)
        return LEARNING_RATE * 0.1 * (0.5 * (1 + np.cos(np.pi * progress)))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# Analyze class distribution and compute BALANCED weights
print(f"\n{'='*60}")
print(f"BANANA RIPENESS CLASSIFICATION - 6 STAGES")
print(f"{'='*60}")
print(f"Classes: {CLASS_COLS}")
print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_df)}")
print(f"  Validation samples: {len(valid_df)}")
print(f"  Test samples: {len(test_df)}")

# Show class distribution
print(f"\nClass Distribution (Training):")
class_counts = train_df[CLASS_COLS].sum().sort_values(ascending=False)
total_samples = len(train_df)
for cls, count in class_counts.items():
    pct = (count / total_samples) * 100
    bar = '█' * int(pct / 2)
    print(f"  {cls:.<15} {int(count):>5} ({pct:>5.1f}%) {bar}")

print(f"\nTraining Configuration:")
print(f"  Image size: {IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"{'='*60}\n")

# BALANCED class weights - not too aggressive
class_weights_dict = None
class_weights_tensor = None

if len(train_df) > 0:
    class_dist = train_df[CLASS_COLS].values.sum(axis=0)
    total = class_dist.sum()
    class_weights_dict = {}
    weights_list = []
    
    print(f"Computing BALANCED class weights:")
    for i, (col, weight) in enumerate(zip(CLASS_COLS, class_dist)):
        if weight > 0:
            # Moderate balancing - (total/count)^0.5 is good balance
            w = np.sqrt(total / (len(class_dist) * weight))
            # Cap maximum weight to avoid extreme values
            w = min(w, 3.0)
            class_weights_dict[i] = w
            weights_list.append(w)
            print(f"  {col:.<15} count={int(weight):>4} → weight={w:.3f}")
        else:
            class_weights_dict[i] = 1.0
            weights_list.append(1.0)
            print(f"  {col:.<15} count={int(weight):>4} → weight=1.000 (WARNING: no samples!)")
    
    # Convert to tensor
    class_weights_tensor = tf.constant(weights_list, dtype=tf.float32)
    print()

# Phase 1: Warmup with frozen base
print(f"\n{'='*60}")
print("PHASE 1: WARMUP TRAINING (Frozen Base)")
print(f"{'='*60}")

history1 = model.fit(
    train_ds, 
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds, 
    validation_steps=validation_steps,
    epochs=EPOCHS_WARMUP, 
    callbacks=[ckpt, es, rlrop], 
    verbose=1
)

# Phase 2: Fine-tune (unfreeze more layers for banana-specific features)
print(f"\n{'='*60}")
print("PHASE 2: FINE-TUNING (Unfrozen Layers)")
print(f"{'='*60}")

base.trainable = True
# Unfreeze last 60 layers for faster training
for layer in base.layers[:-60]: 
    layer.trainable = False

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_variables])
print(f"Trainable parameters: {trainable_count:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE*0.05),  # Very slow for fine-tuning
    loss={
        "class_output": banana_loss,
        "freshness_output": "mse"
    },
    loss_weights={"class_output": 1.0, "freshness_output": 0.1},
    metrics={"class_output": METRICS, "freshness_output": ["mae"]}
)

history2 = model.fit(
    train_ds, 
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds, 
    validation_steps=validation_steps,
    epochs=EPOCHS_FINETUNE,
    callbacks=[ckpt, es, rlrop], 
    verbose=1
)

# Load best weights and save
model.load_weights(MODEL_WEIGHTS)
model.save(MODEL_KERAS)
print(f"\n[✓] Saved best model to {MODEL_KERAS}")

# Comprehensive evaluation
print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")

test_ds = make_ds(test_df, "test", False)
test_steps = len(test_df) // BATCH_SIZE
test_results = model.evaluate(test_ds, steps=test_steps, verbose=1)

print(f"\n{'='*60}")
print("TEST SET RESULTS:")
print(f"{'='*60}")
for name, value in zip(model.metrics_names, test_results):
    if 'class_output' in name:
        percentage = value * 100
        status = "✓ EXCELLENT" if percentage >= 90 else "⚡ GOOD" if percentage >= 85 else "💡 NEEDS IMPROVEMENT"
        print(f"  {name:.<40} {percentage:>6.2f}% {status}")

val_results = model.evaluate(valid_ds, steps=validation_steps, verbose=0)
print(f"\n{'='*60}")
print("VALIDATION SET RESULTS:")
print(f"{'='*60}")
for name, value in zip(model.metrics_names, val_results):
    if 'class_output' in name:
        percentage = value * 100
        status = "✓ EXCELLENT" if percentage >= 90 else "⚡ GOOD" if percentage >= 85 else "💡 NEEDS IMPROVEMENT"
        print(f"  {name:.<40} {percentage:>6.2f}% {status}")

print(f"\n{'='*60}")
best_acc = max([h.history.get('val_class_output_accuracy', [0])[-1] for h in [history1, history2]])
print(f"🎯 Best Validation Accuracy: {best_acc*100:.2f}%")
print(f"{'='*60}")

if best_acc >= 0.90:
    print("🎉 OUTSTANDING! 90%+ ACCURACY ACHIEVED! 🎉")
    print("Your banana ripeness classifier is production-ready!")
elif best_acc >= 0.85:
    print("⚡ Very Good! Close to target.")
    print("Tips to reach 90%:")
    print("  • Increase EPOCHS_FINETUNE to 40-60")
    print("  • Add more training data (especially underrepresented classes)")
    print("  • Try larger image size (e.g., 299x299)")
elif best_acc >= 0.75:
    print("💡 Decent start. Room for improvement.")
    print("Recommendations:")
    print("  • Check data quality and label accuracy")
    print("  • Balance class distribution (add more samples to minority classes)")
    print("  • Increase EPOCHS_WARMUP and EPOCHS_FINETUNE")
    print("  • Consider using EfficientNetB0 instead of MobileNetV2")
else:
    print("⚠️  Lower than expected. Check these:")
    print("  • Data quality: Are images clear? Labels correct?")
    print("  • Class imbalance: Very uneven class distribution?")
    print("  • Dataset size: Need 200+ images per class minimum")
    print("  • Image preprocessing: Check if images load correctly")

print(f"{'='*60}")
print(f"\n6-Stage Banana Ripeness Order:")
print(f"  1. unripe (green)")
print(f"  2. freshunripe (green-yellow)")
print(f"  3. freshripe (yellow, firm)")
print(f"  4. ripe (yellow, soft)")
print(f"  5. overripe (yellow-brown spots)")
print(f"  6. rotten (brown/black)")
print(f"{'='*60}")
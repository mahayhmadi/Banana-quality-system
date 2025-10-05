# Banana Quality Assessment System (BananaScan)

## Project Overview
BananaScan is a deep learning system that automatically classifies banana ripeness and predicts a Freshness Index (0–10) from RGB images. It integrates image preprocessing, classification, and regression to assess quality, detect defects, and estimate shelf-life.  

**Goal:** Automate banana ripeness evaluation and freshness prediction using computer vision.  
**Tech Stack:** Python, TensorFlow/Keras, OpenCV, Pandas, Streamlit, SQLite  
**Model:** MobileNetV2 (pretrained on ImageNet) fine-tuned for 6 ripeness classes.  
**Outputs:** Ripeness class prediction + Freshness Index regression.  

---

## Repository Structure

banana_dataset/ # Dataset (train, valid, test)
cvs/ # CSV backup and splits
dataset_banana/ # Raw dataset
common_config.py # Global configuration file
preprocess_images.py # Image preprocessing script
train_mobilenetv2.py # Model training pipeline
evaluate_and_db.py # Evaluation and SQLite integration
streamlit_app.py # Interactive Streamlit dashboard
produce_quality.db # SQLite database
produce_quality_model.keras # Trained model
requirements.txt # Python dependencies



---

## Dataset and Preprocessing

**Dataset:**  
Contains 15,792 labeled images divided into train, validation, and test sets.  
Classes: `freshripe`, `freshunripe`, `overripe`, `ripe`, `rotten`, `unripe`.  
Images were captured under varying lighting conditions and backgrounds.

**Preprocessing Steps (preprocess_images.py):**
| Step | Description |
|------|--------------|
| Color Segmentation | HSV-based masking to isolate banana regions |
| Morphological Cleaning | Remove background and small artifacts |
| CLAHE | Enhance contrast in LAB color space |
| Gamma Correction | Adjust low-brightness images |
| Denoising | FastNlMeans algorithm to remove noise |
| Resize | Resize all images to 224×224 |
| Augmentation | Horizontal flips and brightness shifts |

The preprocessing pipeline ensures consistent lighting, reduces background bias, and preserves natural banana texture.

---

## Model Architecture

| Component | Details |
|------------|----------|
| Base Model | MobileNetV2 (frozen layers for warmup) |
| Classification Head | Dense layer with 6 softmax units |
| Regression Head | Dense layer (1 unit, scaled 0–10) |
| Loss Function | Weighted categorical crossentropy + MSE |
| Optimizer | Adam (LR = 1e-4) |
| Training Phases | Warmup (5 epochs) + Fine-tuning (5 epochs) |

---

## Model Performance Summary

| Metric | Result | Interpretation |
|---------|---------|----------------|
| Classification Accuracy | ~59% | Moderate performance for a 6-class classification problem |
| AUC (Area Under Curve) | 0.91 | Excellent separability between classes |
| Precision | ~0.69 | Indicates relatively clean predictions |
| Recall | ~0.50 | Suggests data imbalance or underrepresented classes |
| Freshness MAE | 1.65 / 10 | Low average prediction error |

**Summary:**  
The model demonstrates strong discriminative power (AUC > 0.9) despite complex visual variability and lighting conditions. It provides a robust baseline for future enhancement.

---

## Streamlit Dashboard

**Features (streamlit_app.py):**
- Upload or live camera input
- Class and confidence display
- Freshness Index output
- KPI cards for Accuracy, Precision, Recall, and F1-score
- SQLite logging and prediction history
- Embedded video and visual metrics

The dashboard provides real-time model inference and visual feedback for quality inspection.

---

## Database Schema (SQLite)
Stored in `produce_quality.db`

| Column | Description |
|--------|-------------|
| sample_id | Unique identifier |
| image_path | Input image path |
| quality_class | Predicted ripeness class |
| confidence | Softmax confidence score |
| freshness_index | Predicted numeric freshness index |
| created_at | Timestamp |

---

## Results Summary

The combination of preprocessing and MobileNetV2 architecture achieved:
- 59% classification accuracy
- 0.91 AUC score
- 1.65 mean absolute error in freshness prediction

These results indicate good generalization with potential improvement through further dataset balancing and advanced augmentation.

---

## Future Work
- Integrate YOLOv8-based banana detection for better region isolation  
- Expand dataset with balanced class distribution  
- Convert model for TensorFlow Lite deployment on mobile devices  
- Add time-series modeling for shelf-life forecasting  
- Extend to multi-fruit classification (mango, apple, orange)

---


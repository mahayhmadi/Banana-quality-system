# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import datetime

# ================================
# Paths
# ================================
DATA_ROOT = r"C:\Users\bbuser\Desktop\banana_dataset"

TRAIN_CSV = os.path.join(DATA_ROOT, "train", "_classes.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid", "_classes.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test", "_classes.csv")

PROCESSED_TRAIN_DIR = os.path.join(DATA_ROOT, "train_processed")
PROCESSED_VALID_DIR = os.path.join(DATA_ROOT, "valid_processed")
PROCESSED_TEST_DIR  = os.path.join(DATA_ROOT, "test_processed")

os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_VALID_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# ================================
# Preprocessing Parameters
# ================================
IMAGE_SIZE = (224, 224)

# نطاقات ألوان محسنة لجميع مراحل نضج الموز
COLOR_RANGES = {
    "green_very_light": ([20, 20, 30], [45, 255, 255]),    # أخضر فاتح جداً
    "green_light":      ([40, 30, 40], [75, 255, 255]),    # أخضر فاتح
    "green_yellow":     ([75, 25, 50], [95, 255, 255]),    # أخضر مصفر
    "yellow_light":     ([15, 50, 80], [35, 255, 255]),    # أصفر فاتح
    "yellow":           ([18, 80, 100], [32, 255, 255]),   # أصفر
    "yellow_dark":      ([10, 50, 60], [25, 200, 200]),    # أصفر داكن
    "brown_light":      ([8, 40, 40], [20, 180, 180]),     # بني فاتح
    "brown":            ([5, 30, 20], [15, 160, 160]),     # بني
    "brown_dark":       ([0, 20, 10], [10, 140, 140]),     # بني داكن
}

BACKGROUND_MODE = "smart"  # smart, black, white, blur

# ================================
# Helpers
# ================================
def auto_adjust_brightness(image):
    """تعديل السطوع تلقائياً بناءً على الصورة"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    
    if brightness < 80:
        # صورة داكنة - زيادة السطوع
        gamma = 1.3 + (80 - brightness) / 100
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    elif brightness > 200:
        # صورة ساطعة جداً - تقليل السطوع
        gamma = 0.8
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    return image

def smart_contrast_enhancement(image):
    """تحسين التباين بشكل ذكي"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE بمعاملات أخف
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def create_smart_mask(hsv_image):
    """إنشاء قناع ذكي لاكتشاف الموز"""
    masks = []
    for low, high in COLOR_RANGES.values():
        mask = cv2.inRange(hsv_image, np.array(low, np.uint8), np.array(high, np.uint8))
        masks.append(mask)
    
    # دمج جميع الأقنعة
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, m)
    
    # تنظيف خفيف فقط
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # إزالة النقاط الصغيرة
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # ملء الثقوب الصغيرة
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # تنعيم حواف القناع
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
    
    return combined_mask

def apply_mask_smoothly(image, mask):
    """تطبيق القناع بحواف ناعمة"""
    # إنشاء قناع ناعم مع حواف تدريجية
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 11)
    mask_float = mask_blur.astype(float) / 255.0
    
    # اختيار خلفية ذكية
    if BACKGROUND_MODE == "smart":
        # استخدام متوسط لون الحواف كخلفية
        background = cv2.blur(image, (51, 51))
    elif BACKGROUND_MODE == "white":
        background = np.ones_like(image) * 255
    elif BACKGROUND_MODE == "blur":
        background = cv2.GaussianBlur(image, (51, 51), 0)
    else:  # black
        background = np.zeros_like(image)
    
    # دمج بشكل ناعم
    result = (image * mask_float[..., None] + 
              background * (1 - mask_float[..., None])).astype(np.uint8)
    
    return result

def gentle_denoise(image):
    """إزالة تشويش خفيفة"""
    return cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6, 
                                           templateWindowSize=7, searchWindowSize=15)

def safe_augmentation(image):
    """Augmentation آمن ومحسن"""
    h, w = image.shape[:2]
    
    # Flip أفقي (50%)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # دوران خفيف (-10 إلى +10 درجات)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # تغيير حجم خفيف (95% إلى 105%)
    if random.random() > 0.5:
        scale = random.uniform(0.95, 1.05)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # قص أو توسيع للحجم الأصلي
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = cv2.copyMakeBorder(image, pad_h, h-new_h-pad_h, 
                                      pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
    
    # تعديل خفيف جداً في الألوان
    if random.random() > 0.6:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        hsv[:, :, 1] *= random.uniform(0.95, 1.05)  # تشبع
        hsv[:, :, 2] *= random.uniform(0.95, 1.05)  # سطوع
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

# ================================
# Preprocessing
# ================================
def preprocess_and_save(img_path, save_dir, augment=False):
    """معالجة وحفظ الصورة"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    original = img.copy()
    
    # 1. تحسين السطوع التلقائي
    img = auto_adjust_brightness(img)
    
    # 2. إنشاء قناع ذكي
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = create_smart_mask(hsv)
    
    # تحقق من وجود محتوى كافٍ
    if cv2.countNonZero(mask) < 100:
        # إذا فشل القناع، احفظ الصورة الأصلية مع تحسين بسيط
        img = cv2.resize(original, IMAGE_SIZE)
        img = smart_contrast_enhancement(img)
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        return save_path
    
    # 3. تطبيق القناع بشكل ناعم
    img = apply_mask_smoothly(img, mask)
    
    # 4. تحسين التباين بشكل خفيف
    img = smart_contrast_enhancement(img)
    
    # 5. إزالة التشويش بشكل خفيف
    img = gentle_denoise(img)
    
    # 6. تغيير الحجم
    img = cv2.resize(img, IMAGE_SIZE)
    
    # 7. Augmentation (فقط للتدريب)
    if augment:
        img = safe_augmentation(img)
    
    # 8. حفظ
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return save_path

# ================================
# Process CSV with Log
# ================================
def run_for_csv(csv_path, save_dir, augment=False):
    print(f"🔎 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    processed_paths = []
    failed_count = 0
    base_dir = os.path.dirname(csv_path)
    
    log_file = os.path.join(DATA_ROOT, f"preprocess_log_{os.path.basename(csv_path)}.txt")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Run started: {datetime.datetime.now()}\n")
        log.write(f"Augmentation: {augment}\n")
        log.write(f"Background mode: {BACKGROUND_MODE}\n")
        log.write(f"{'='*60}\n\n")
        
        for i, fname in enumerate(tqdm(df["filename"], desc=f"Processing {os.path.basename(csv_path)}")):
            img_path = os.path.join(base_dir, fname)
            
            if not os.path.exists(img_path):
                log.write(f"[ERROR] File not found: {fname}\n")
                processed_paths.append("")
                failed_count += 1
                continue
            
            out_path = preprocess_and_save(img_path, save_dir, augment=augment)
            
            if out_path:
                processed_paths.append(out_path)
            else:
                processed_paths.append("")
                failed_count += 1
                log.write(f"[FAILED] {fname}\n")
            
            # سجل التقدم كل 100 صورة
            if (i + 1) % 100 == 0:
                log.write(f"Progress: {i+1}/{len(df)} images processed\n")
        
        log.write(f"\n{'='*60}\n")
        log.write(f"Completed: {datetime.datetime.now()}\n")
        log.write(f"Total: {len(df)} | Success: {len(df)-failed_count} | Failed: {failed_count}\n")
        log.write(f"{'='*60}\n")
    
    df["processed_path"] = processed_paths
    new_csv = csv_path.replace("_classes.csv", "_classes_processed.csv")
    df.to_csv(new_csv, index=False)
    print(f"✅ Saved {new_csv}")
    print(f"📊 Success: {len(df)-failed_count}/{len(df)} | Failed: {failed_count}")

# ================================
# Main
# ================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍌 IMPROVED BANANA PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    print("📋 Processing Training Set...")
    run_for_csv(TRAIN_CSV, PROCESSED_TRAIN_DIR, augment=True)
    
    print("\n📋 Processing Validation Set...")
    run_for_csv(VALID_CSV, PROCESSED_VALID_DIR, augment=False)
    
    print("\n📋 Processing Test Set...")
    run_for_csv(TEST_CSV, PROCESSED_TEST_DIR, augment=False)
    
    print("\n" + "="*60)
    print("🎉 ALL PREPROCESSING COMPLETE!")
    print("="*60)
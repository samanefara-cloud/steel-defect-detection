"""
compare_gt_vs_pred.py
نمایش کنار هم باکس واقعی (Ground Truth از فایل لیبل) و باکس پیش‌بینی مدل
خروجی: یک عکس ترکیبی (سمت چپ GT، سمت راست Prediction) برای هر عکس تست
"""

import cv2                                  # برای رسم باکس و خواندن/نوشتن عکس
import os
from ultralytics import YOLO

# ---------- تنظیمات ----------
MODEL_PATH = "best.pt"
IMAGES_DIR = "test/images"                  # پوشه‌ی عکس‌های خام تست
LABELS_DIR = "test/labels"                  # پوشه‌ی فایل‌های لیبل (فرمت YOLO txt)
OUTPUT_DIR = "gt_vs_pred"
CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]  # ترتیب طبق data.yaml
CONF_THRESHOLD = 0.35
NUM_SAMPLES = 5                             # چند تا عکس نمونه بساز (بیشتر لازم نیست)

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)


def draw_gt_boxes(image, label_path, color=(0, 255, 0)):
    """رسم باکس‌های واقعی (سبز) از روی فایل لیبل YOLO"""
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        return image                                    # اگه عکس عیبی نداشته باشه، لیبل خالیه
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            cls_id = int(parts[0])
            # تبدیل مختصات نرمالایز‌شده (۰ تا ۱) به پیکسل واقعی
            xc, yc, bw, bh = [float(p) for p in parts[1:5]]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, CLASS_NAMES[cls_id], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


# ---------- پردازش چند عکس نمونه ----------
image_files = sorted(os.listdir(IMAGES_DIR))[:NUM_SAMPLES]

for filename in image_files:
    img_path = os.path.join(IMAGES_DIR, filename)
    label_path = os.path.join(LABELS_DIR, filename.rsplit(".", 1)[0] + ".txt")

    # --- تصویر سمت چپ: باکس واقعی ---
    gt_image = cv2.imread(img_path)
    gt_image = draw_gt_boxes(gt_image, label_path, color=(0, 255, 0))  # سبز = واقعی

    # --- تصویر سمت راست: باکس پیش‌بینی مدل ---
    result = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)[0]
    pred_image = result.plot()                          # ultralytics خودش با رنگ آبی/قرمز رسم می‌کنه

    # --- چسباندن دو عکس کنار هم ---
    gt_image = cv2.resize(gt_image, (pred_image.shape[1], pred_image.shape[0]))
    combined = cv2.hconcat([gt_image, pred_image])

    out_path = os.path.join(OUTPUT_DIR, f"compare_{filename}")
    cv2.imwrite(out_path, combined)
    print(f"ذخیره شد: {out_path}  (چپ=واقعی سبز, راست=پیش‌بینی مدل)")

print("✅ تمام مقایسه‌ها آماده شدن.")


"""
============================================================
خلاصه‌ی کد:

هدف کد:
    ساخت عکس‌های مقایسه‌ای (Ground Truth کنار Prediction) برای اثبات
    بصری دقت مدل به کارفرما/بازدیدکننده - قانع‌کننده‌تر از نشون‌دادن
    فقط یک عدد mAP.

پیش‌نیازها:
    - pip install ultralytics opencv-python
    - پوشه‌ی test/images و test/labels از دیتاست
    - وزن مدل best.pt
    - CLASS_NAMES باید دقیقاً با ترتیب کلاس‌ها در data.yaml یکی باشه

مراحل منطق کد:
    ۱. برای هر عکس نمونه، فایل لیبل واقعیش رو می‌خونه و باکس سبز رسم می‌کنه (GT)
    ۲. مدل رو روی همون عکس اجرا می‌کنه و باکس پیش‌بینی رو می‌گیره
    ۳. دو تصویر رو کنار هم (چپ=واقعی، راست=پیش‌بینی) می‌چسبونه
    ۴. ذخیره در پوشه‌ی gt_vs_pred/

این کد رو کامل به AI بسپار؛ تنها چیزی که باید خودت چک کنی اینه که
CLASS_NAMES با ترتیب واقعی کلاس‌ها در data.yaml مطابقت داشته باشه -
وگرنه اسم اشتباه زیر باکس‌ها نوشته میشه (خود جعبه درست میمونه، فقط برچسبش عوض میشه).
============================================================
"""

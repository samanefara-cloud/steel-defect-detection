"""
count_classes.py
شمارش تعداد باکس‌های هر کلاس در فایل‌های لیبل YOLO (برای تشخیص class imbalance)
"""

import os
from collections import Counter
import yaml

# ---------- تنظیمات (مسیر رو با توجه به دیتاست خودت عوض کن) ----------
DATA_YAML = "/content/NEU-DET/data.yaml"    # همون فایلی که توی args.yaml دیدیم
LABELS_DIR = "/content/NEU-DET/labels/train"  # پوشه‌ی لیبل‌های آموزش (فرمت txt استاندارد YOLO)

# ---------- خواندن اسم کلاس‌ها از data.yaml ----------
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)                 # خواندن فایل تنظیمات دیتاست
class_names = data_cfg["names"]                  # لیست اسم کلاس‌ها به ترتیب اندیس

# ---------- شمارش تعداد باکس هر کلاس ----------
counter = Counter()                              # شمارنده برای هر کلاس
total_images_with_class = Counter()              # تعداد عکس‌هایی که حداقل یک باکس از اون کلاس دارن

for filename in os.listdir(LABELS_DIR):
    if not filename.endswith(".txt"):
        continue                                  # فقط فایل‌های لیبل رو بخون
    seen_classes_in_file = set()
    with open(os.path.join(LABELS_DIR, filename)) as f:
        for line in f:
            class_id = int(line.split()[0])       # اولین عدد هر خط = شماره کلاس
            counter[class_id] += 1                # افزایش شمارنده‌ی باکس
            seen_classes_in_file.add(class_id)
    for c in seen_classes_in_file:
        total_images_with_class[c] += 1           # افزایش شمارنده‌ی تعداد عکس

# ---------- نمایش نتیجه ----------
print(f"{'کلاس':<20}{'تعداد باکس':<15}{'تعداد عکس'}")
for idx, name in enumerate(class_names):
    print(f"{name:<20}{counter[idx]:<15}{total_images_with_class[idx]}")


"""
============================================================
خلاصه‌ی کد:

هدف کد:
    شمارش تعداد نمونه (باکس) و تعداد عکس هر کلاس در دیتاست، برای
    تشخیص اینکه آیا کلاس‌های ضعیف (crazing, rolled-in_scale) واقعاً
    نمونه‌ی کمتری نسبت به بقیه دارن یا نه (class imbalance).

پیش‌نیازها:
    - pip install pyyaml
    - مسیر درست data.yaml و پوشه‌ی labels/train

مراحل منطق کد:
    ۱. خواندن اسم کلاس‌ها از data.yaml
    ۲. پیمایش تمام فایل‌های txt لیبل (هر خط = یک باکس با فرمت: class_id x y w h)
    ۳. شمارش تعداد کل باکس هر کلاس + تعداد عکس‌هایی که اون کلاس توشون هست
    ۴. چاپ جدول نهایی برای مقایسه

این بخش رو می‌تونی کامل به AI بسپاری (نیازی نیست خودت جزئیات پارس کردن
txt رو حفظ کنی)؛ چیزی که باید خودت بفهمی، فقط *تفسیر نتیجه‌ست*:
اگه crazing یا rolled-in_scale عدد خیلی کمتری داشتن، یعنی مشکل از
imbalance ـه و راه‌حل oversampling/augmentation ـه، نه صرفاً epoch بیشتر.
============================================================
"""

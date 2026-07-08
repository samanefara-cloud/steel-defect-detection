# تشخیص عیب سطحی فولاد با YOLOv8 | Steel Surface Defect Detection

تشخیص و لوکالایز کردن ۶ نوع عیب رایج در سطح فولاد (crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches) با استفاده از ترنسفر لرنینگ روی YOLOv8،
مناسب بازرسی کیفی خودکار در خط تولید.

## مسئله
بازرسی بصری دستی عیوب سطحی فولاد در خط تولید کند و وابسته به خطای انسانیه.
هدف این پروژه ساخت یک مدل تشخیص عیب (object detection) بود که بتونه محل دقیق
عیب روی قطعه رو با جعبه‌ی محاطی (bounding box) مشخص کنه — نه فقط بگه «معیوب هست یا نه».

## دیتاست
دیتاست [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
(دیتابیس استاندارد دانشگاه Northeastern برای عیوب سطحی فولاد)، آماده‌سازی و annotation
از طریق Roboflow. شامل ۶ کلاس عیب با در مجموع ~۱۸۰۰ عکس.

## رویکرد فنی
- **مدل:** YOLOv8n (کوچک‌ترین نسخه‌ی خانواده‌ی YOLOv8) با وزن‌های اولیه‌ی از‌پیش‌آموزش‌دیده روی COCO
- **روش:** ترنسفر لرنینگ با فریز کردن ۲۱ لایه‌ی اول (بک‌بون)، آموزش لایه‌های نهایی
- **زیرساخت:** آموزش روی Kaggle Notebook (GPU T4 رایگان)
- **دلیل انتخاب YOLOv8n:** نیاز پروژه به سرعت اجرا روی سیستم بدون GPU اختصاصی، مناسب استقرار در محیط صنعتی با منابع محدود

## نتایج

| متریک | مقدار |
|---|---|
| mAP50 | 0.730 |
| mAP50-95 | 0.380 |
| Precision | 0.709 |
| Recall | 0.672 |

### عملکرد به تفکیک کلاس (Recall)
| کلاس | Recall |
|---|---|
| patches | 0.89 |
| scratches | 0.89 |
| inclusion | 0.81 |
| pitted_surface | 0.79 |
| rolled-in_scale | 0.60 |
| crazing | 0.39 |

![Confusion Matrix](assets/confusion_matrix.png)

### نمونه‌ی پیش‌بینی روی عکس واقعی (سمت چپ = واقعیت، سمت راست = پیش‌بینی مدل)

**نمونه‌ی موفق (patches):**
![Sample Success Patches](assets/sample_success_patches.jpg)

**نمونه‌ی موفق (inclusion):**
![Sample Success Inclusion](assets/sample_success_inclusion.jpg)

**نمونه‌ی ناموفق (crazing) — نشان‌دهنده‌ی دقیق همان محدودیتی که در بخش زیر تحلیل شده:**
![Sample Failure Crazing](assets/sample_failure_crazing.jpg)

## درس‌های کلیدی و تحلیل خطا (این بخش رو حتماً بخون)

عملکرد مدل روی دو کلاس `crazing` و `rolled-in_scale` ضعیف‌تر از بقیه بود. برای پیدا کردن
علت، سه آزمایش کنترل‌شده (ablation) انجام شد:

| آزمایش | تغییر نسبت به baseline | نتیجه روی crazing |
|---|---|---|
| Baseline | فریز فعال، بدون augmentation هندسی | Recall = 0.39 |
| آزمایش ۱ | حذف فریز + augmentation (flipud, degrees) | Recall = 0.30 (بدتر) |
| آزمایش ۲ | فقط augmentation (فریز نگه داشته شد) | Recall = 0.185 (بدترین) |

**نتیجه:** با ایزوله کردن متغیرها، مشخص شد augmentation هندسی شدید (چرخش ۲۰ درجه)
باعث تار شدن ویژگی‌های نازک و کم‌کنتراست کلاس crazing میشه، نه اینکه حذف فریز مشکل باشه.
این یافته باعث شد baseline به‌عنوان بهترین نسخه‌ی نهایی انتخاب بشه، به‌جای اتلاف وقت
روی مسیر اشتباه.

**گام بعدی پیشنهادی (برای ادامه‌ی این پروژه):** تست augmentation ملایم‌تر (مثلاً
`degrees=5` بدون `flipud`)، یا استفاده از مدل بزرگ‌تر (`yolov8s`) با ظرفیت بیشتر
برای ویژگی‌های ظریف.

## نحوه‌ی اجرا
```bash
pip install -r requirements.txt

# آموزش مدل
python train_fast.py

# تولید عکس‌های نمونه‌ی پیش‌بینی
python predict_samples.py
```

## ساختار پروژه
```
.
├── train_fast.py              # اسکریپت آموزش (ترنسفر لرنینگ)
├── predict_samples.py         # اجرای مدل روی عکس نمونه + رسم باکس
├── count_classes.py           # تحلیل توازن کلاس‌ها در دیتاست
├── requirements.txt
├── assets/
│   ├── confusion_matrix.png
│   └── sample_prediction.png
└── README.md
```

## محدودیت‌های شناخته‌شده
- Recall کلاس `crazing` (۰.۳۹) برای استقرار واقعی صنعتی کافی نیست؛ نیاز به داده‌ی
  بیشتر یا augmentation هدفمند‌تر دارد.
- مدل روی دیتاست فولاد آموزش دیده؛ برای مواد دیگر (مثلاً ریخته‌گری) نیاز به
  آموزش مجدد دارد (پروژه‌ی دوم همین مجموعه).

## تکنولوژی‌ها
Python, PyTorch, Ultralytics YOLOv8, Roboflow, Kaggle Notebooks

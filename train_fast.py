"""
train_fast.py
اسکریپت آموزش سریع برای پروژه‌ی defect detection (فولاد / ریخته‌گری)
"""

import torch                                                    # کتابخانه‌ی اصلی یادگیری عمیق
import torch.nn as nn                                           # لایه‌ها و توابع لاس
from torch.utils.data import DataLoader, Subset                 # بارگذاری داده + گرفتن زیرمجموعه کوچک
from torchvision import datasets, transforms, models            # دیتاست آماده، تبدیل تصویر، مدل‌های از‌پیش‌آموزش‌دیده
from sklearn.metrics import classification_report, confusion_matrix  # ارزیابی نهایی
import numpy as np

# ---------- ۱. تنظیمات کلی (اینا رو با توجه به دیتاستت عوض کن) ----------
DATA_DIR = "data"          # باید دو زیرپوشه داشته باشه: data/train  و  data/val
                            # و هرکدوم زیرپوشه به تعداد کلاس (مثلا: defect / no_defect)
IMG_SIZE = 96               # عمدا کوچیک گرفتیم -> سرعت آموزش چند برابر میشه، دقت افت محسوسی نمی‌کنه
BATCH_SIZE = 32
SUBSET_FRACTION = 0.4       # فقط ۴۰٪ داده رو استفاده کن برای تست سریع -> بعدا اگه وقت/GPU بود کل داده رو بده
EPOCHS = 6                  # با فریز کردن بک‌بون، ۶ ایپاک کاملا کافیه
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"         # روی Kaggle GPU خودکار فعال میشه

# ---------- ۲. آماده‌سازی داده ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                    # یکسان‌سازی سایز تصاویر
    transforms.RandomHorizontalFlip(),                          # افزایش داده ساده (augmentation)
    transforms.ToTensor(),                                      # تبدیل به تنسور
    transforms.Normalize([0.485, 0.456, 0.406],                 # نرمال‌سازی مطابق ImageNet
                          [0.229, 0.224, 0.225]),
])

train_full = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)   # خواندن خودکار کلاس‌ها از نام پوشه
val_data   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)

# گرفتن زیرمجموعه‌ی تصادفی از داده‌ی آموزش برای سرعت بیشتر
n_samples = int(len(train_full) * SUBSET_FRACTION)
indices = np.random.choice(len(train_full), n_samples, replace=False)
train_data = Subset(train_full, indices)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

num_classes = len(train_full.classes)                            # تعداد کلاس‌ها رو خودکار می‌گیره

# ---------- ۳. ساخت مدل (ترنسفر لرنینگ) ----------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")             # مدل سبک و سریع، مناسب سیستم ضعیف
for param in model.features.parameters():
    param.requires_grad = False                                  # فریز کردن بک‌بون -> فقط لایه آخر آموزش می‌بینه

model.classifier[1] = nn.Linear(model.last_channel, num_classes) # جایگزینی لایه‌ی خروجی با تعداد کلاس‌های ما
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)  # فقط پارامترهای لایه آخر آپدیت میشن

# ---------- ۴. حلقه‌ی آموزش ----------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# ---------- ۵. ارزیابی نهایی (این بخش برای رزومه و ارائه به کارخانه خیلی مهمه) ----------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n--- گزارش دقت مدل ---")
print(classification_report(all_labels, all_preds, target_names=train_full.classes))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# ---------- ۶. ذخیره‌ی مدل ----------
torch.save(model.state_dict(), "defect_model.pth")
print("مدل ذخیره شد: defect_model.pth")


"""
============================================================
خلاصه‌ی کد (برای مرور سریع بدون نیاز به خوندن کل کد):

هدف کد:
    آموزش سریع یک مدل تشخیص عیب (defect / no-defect) با کمترین
    زمان و منابع ممکن، مناسب سیستم بدون GPU قوی یا اینترنت ضعیف.

پیش‌نیازها:
    - pip install torch torchvision scikit-learn
    - ساختار پوشه: data/train/<class_name>/*.jpg  و  data/val/<class_name>/*.jpg
    - بهتره روی Kaggle Notebook اجرا بشه نه Colab، چون Kaggle حتی
      اگه مرورگر رو ببندی به اجرا ادامه میده (مشکل قطعی اینترنت رو حل می‌کنه).

مراحل منطق کد:
    ۱. خواندن تصاویر با ImageFolder (خودش کلاس‌ها رو از اسم پوشه تشخیص میده)
    ۲. کوچیک کردن سایز تصویر و گرفتن فقط بخشی از داده -> سرعت بالا
    ۳. استفاده از مدل آماده‌ی mobilenet_v2 و فریز کردن بک‌بون
       (یعنی فقط لایه‌ی آخر آموزش می‌بینه، نه کل شبکه -> چند برابر سریع‌تر)
    ۴. آموزش فقط لایه‌ی طبقه‌بندی نهایی برای چند ایپاک کم
    ۵. ارزیابی با classification_report و confusion_matrix
       (این خروجی رو مستقیم می‌تونی توی رزومه/ارائه بذاری)
    ۶. ذخیره‌ی مدل نهایی

نکته‌ی مهم برای یادگیری واقعی (این بخش رو خودت باید بفهمی، به AI نسپار):
    - چرا بک‌بون رو فریز می‌کنیم؟ چون مدل از‌پیش روی میلیون‌ها عکس
      یاد گرفته لبه/بافت/شکل تشخیص بده؛ ما فقط لایه‌ی تصمیم‌گیری نهایی
      رو مخصوص دیتاست خودمون آموزش میدیم -> نیاز به داده و GPU کمتر.
    - توی defect detection، Recall کلاس "defect" از Accuracy مهم‌تره؛
      چون رد کردن یک قطعه‌ی معیوب (false negative) خیلی گرون‌تر از
      false positive تمومه. حتما توی گزارش نهایی این رو توضیح بده.

بخش‌هایی که می‌تونی به agentic AI بسپاری (فقط خلاصه‌شو بدون، وقت
زیاد روش نذار، توی کار واقعی هم همینا رو به AI میدی):
    - نوشتن boilerplate دیتالودر و transform
    - رسم نمودار loss/accuracy با matplotlib
    - ساخت یک اپ ساده‌ی Streamlit/Gradio برای دمو زنده به کارخانه
    - نوشتن README گیت‌هاب برای پروژه
============================================================
"""

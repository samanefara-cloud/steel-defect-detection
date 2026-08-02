import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="تشخیص عیوب صنعتی", layout="wide")
st.title("🏭 سیستم تشخیص عیوب صنعتی")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()
st.success("✅ مدل با موفقیت بارگذاری شد!")

uploaded_file = st.file_uploader("تصویر خود را انتخاب کنید", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    with st.spinner("🔄 در حال تشخیص..."):
        results = model(img_np)
    
    st.subheader("🔍 نتیجه تشخیص")
    result_img = results[0].plot()
    st.image(result_img, caption="نتیجه تشخیص", use_column_width=True)
    
    st.subheader("📋 جزئیات تشخیص")
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls)]
        conf = float(box.conf)
        st.write(f"🔹 **{cls}**: {conf*100:.1f}%")
import json

import os

import numpy as np

import streamlit as st

import tensorflow as tf

from PIL import Image

MODEL_PATH = "best_mobilenet.keras"

CLASS_PATH = "class_indices.json"

IMG_SIZE = 224

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="centered")

st.title("🌱 ระบบตรวจวินิจฉัยและจำแนกโรคใบพืช")

st.write("อัปโหลดภาพถ่ายใบพืช เพื่อจำแนกประเภทโรค")


@st.cache_resource

def load_resources():

    for path in (MODEL_PATH, CLASS_PATH):

        if not os.path.exists(path):

            raise FileNotFoundError(f"ไม่พบไฟล์: {path} (ต้องอยู่โฟลเดอร์เดียวกับ app.py)")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as f:

        class_dict = json.load(f)

    return model, class_dict


def split_name(raw):

    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""

    if "___" in raw:

        plant, disease = raw.split("___", 1)

        return plant.replace("_", " "), disease.replace("_", " ")

    return raw.replace("_", " "), ""


try:

    model, class_dict = load_resources()

except Exception as e:

    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

    st.stop()

uploaded_file = st.file_uploader("เลือกภาพใบพืช", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="ภาพที่อัปโหลด", width=400)

    if st.button("วิเคราะห์ภาพ"):

        with st.spinner("กำลังประมวลผล..."):

            # Same preprocessing as training: resize only, values stay 0-255

            # (the model's Rescaling layer converts to -1..1 internally)

            arr = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)

            arr = np.expand_dims(arr, axis=0)

            probs = model.predict(arr, verbose=0)[0]

        top = np.argsort(probs)[-3:][::-1]

        best = int(top[0])

        confidence = float(probs[best]) * 100

        plant, disease = split_name(class_dict.get(str(best), "Unknown"))

        st.success("ประมวลผลเสร็จสิ้น!")

        st.subheader(f"🍃 ชนิดพืช: {plant}")

        if disease.lower() == "healthy":
st.info("สถานะ: พืชสุขภาพดี")

        else:

            st.warning(f"ผลการจำแนก: {disease}")

        st.write(f"ค่าความมั่นใจ (Confidence): {confidence:.2f}%")

        if confidence < 60:

            st.caption("⚠️ ความมั่นใจต่ำ ภาพอาจไม่ชัดหรือไม่ใช่ใบพืชที่โมเดลรู้จัก")

        with st.expander("ดู 3 อันดับแรก"):

            for i in top:

                p, d = split_name(class_dict.get(str(int(i)), "Unknown"))

                st.write(f"{p} - {d or 'Healthy'}: {probs[i] * 100:.2f}%")

        st.caption("ผลการทำนายเป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรคโดยผู้เชี่ยวชาญ")
 

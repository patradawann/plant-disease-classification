import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import base64
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 ระบบตรวจวินิจฉัยและจำแนกโรคใบพืช")
st.write("อัปโหลดภาพถ่ายใบพืชเพื่อจำแนกประเภทโรค")


@st.cache_resource
def load_resources():
    model_path = "best_mobilenet.keras"
    json_path = "class_indices.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ไม่พบไฟล์คลาส: {json_path}")

    model = tf.keras.models.load_model(model_path)

    with open(json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)

    return model, class_dict


try:
    model, class_dict = load_resources()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "เลือกภาพใบพืช",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    # 1. อ่านไฟล์ภาพและนำไปประมวลผลสำหรับ Model
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. แปลงไฟล์เป็น Base64 เพื่อฝังรูปใน HTML โดยตรง (แก้ปัญหาการ Render รูปภาพบน Streamlit)
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"
    
    # แสดงรูปภาพด้วย HTML Data URI
    st.markdown(
        f'<div style="text-align: center;">'
        f'<img src="data:{mime_type};base64,{base64_image}" style="max-width: 100%; height: auto; border-radius: 8px;">'
        f'<p style="color: gray; font-size: 0.8em; margin-top: 5px;">ภาพใบพืชที่อัปโหลด</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button("วิเคราะห์ภาพ"):
        with st.spinner("กำลังประมวลผล..."):
            # 1. ปรับขนาดภาพ
            image_resized = image.resize((224, 224))

            # 2. แปลงภาพเป็น Array
            img_array = np.array(image_resized, dtype=np.float32)

            # 3. เพิ่มมิติ Batch
            img_array = np.expand_dims(img_array, axis=0)

            # 4. ทำนาย
            predictions = model.predict(img_array, verbose=0)[0]
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx]) * 100

            raw_class_name = class_dict.get(str(class_idx), "Unknown Class")

            # 5. แยกชื่อพืชและโรค
            if "___" in raw_class_name:
                plant_name, disease_name = raw_class_name.split("___", 1)
                plant_name = plant_name.replace("_", " ")
                disease_name = disease_name.replace("_", " ")
            else:
                plant_name = raw_class_name
                disease_name = ""

        # 6. แสดงผล
        st.success("ประมวลผลเสร็จสิ้น!")
        st.subheader(f"🍃 ชนิดพืช: {plant_name}")

        if disease_name.lower() == "healthy":
            st.info("สถานะ: พืชสุขภาพดี")
        else:
            st.warning(f"ผลการจำแนก: {disease_name}")

        st.write(f"ค่าความมั่นใจ (Confidence): {confidence:.2f}%")
        st.caption("ผลการทำนายเป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยโรคโดยผู้เชี่ยวชาญ")

import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model_and_labels():
    model = load_model("model/keras_model.h5")
    class_names = open("model/labels.txt", "r", encoding="UTF-8").readlines()
    return model, class_names

def load_and_predict(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    try:
        prediction = model.predict(data)
    except tf.errors.InvalidArgumentError:
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.keras.backend.set_session(sess)
            prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

def main():
    st.set_page_config(page_title="마스크 착용 여부 판단", page_icon="😷", layout="centered")

    st.markdown(
        """
        <h1 style='text-align: center; color: #2E3B55;'>🏥 마스크 착용 상태 확인</h1>
        <h4 style='text-align: center; color: #6C757D;'>얼굴 사진을 업로드하면 마스크를 착용하였는지 파악합니다.</h4>
        <hr style='border: 1px solid #DEE2E6;'>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "📸 이미지를 업로드하세요", type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG, WEBP 형식의 이미지를 업로드하세요."
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 업로드한 이미지", use_container_width=True)

        model, class_names = load_model_and_labels()

        with st.spinner("🕵️ 분석 중입니다... 잠시만 기다려 주세요!"):
            image = Image.open(uploaded_file)
            class_name, confidence_score = load_and_predict(image, model, class_names)

        class_name_masked = " ".join(class_name.split()[1:])
        styled_result = f"""
        <div style='text-align: center; padding: 15px; border-radius: 10px; 
                    background-color: #E3F2FD; color: #0D47A1; font-size: 22px;'>
            <b>{class_name_masked}습니다.<b>
        </div>
        """
        st.markdown(styled_result, unsafe_allow_html=True)

        st.success(f"📊 **분석 정확도: `{(confidence_score * 100):.2f}%`**", icon="✅")
    else:
        st.warning("📢 이미지를 업로드해 주세요.", icon="⚠️")

if __name__ == "__main__":
    main()

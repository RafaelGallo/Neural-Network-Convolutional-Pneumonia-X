import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# ✅ Este deve ser o primeiro comando relacionado ao Streamlit
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# 🎯 Título e descrição
st.title("🩺 Pneumonia X-Ray Classifier")
st.write("Este app usa uma CNN treinada para classificar imagens de raio-X como **NORMAL** ou **PNEUMONIA**.")

# 📂 Upload da imagem
uploaded_file = st.file_uploader("Faça o upload de uma imagem de raio-X (formato PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Converte para escala de cinza
    st.image(image, caption='Imagem carregada', use_column_width=True)

    # 🔄 Preprocessamento
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)  # (batch, height, width, channels)

    # 📥 Carrega o modelo (ajuste o caminho conforme necessário)
    model = load_model("models/CNN_Classification_1.h5")

    # 🔍 Predição
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

    # 🎯 Exibe resultado
    st.subheader("Resultado:")
    st.markdown(f"**Classe prevista**: :orange[{label}]")
    st.markdown(f"**Probabilidade**: `{prob:.2%}`")

    if label == "PNEUMONIA":
        st.warning("⚠️ Indício de pneumonia detectado. Consulte um médico especialista.")
    else:
        st.success("✅ Pulmões normais detectados.")

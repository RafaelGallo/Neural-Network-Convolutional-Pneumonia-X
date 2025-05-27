import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Caminho local do modelo treinado
MODEL_PATH = os.path.join("models", "CNN_Classification_1.h5")

# Carregar o modelo CNN
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Configurações da interface
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🧠 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image (JPG/PNG) and let the model predict if it's **Normal** or **Pneumonia**.")

# Upload da imagem
uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocessamento da imagem
    image = Image.open(uploaded_file).convert("L").resize((500, 500))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Converter para array, normalizar e adicionar canal RGB + batch dimension
    img_array = np.array(image) / 255.0
    img_rgb = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
    img_input = np.expand_dims(img_rgb, axis=0)

    # Predição
    prediction = model.predict(img_input)[0][0]
    label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### 🩺 Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

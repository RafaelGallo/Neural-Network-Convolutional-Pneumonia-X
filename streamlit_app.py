import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# 🔧 Definir configurações da página (deve ser a primeira linha)
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# 📂 Caminho do modelo (deve existir em /models)
MODEL_PATH = "models/CNN_Classification_1.h5"

@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Erro ao carregar o modelo. Verifique o caminho ou se há camadas personalizadas.")
        st.stop()

# Carregar modelo
model = load_cnn_model()

# 📄 Título
st.title("🫁 Pneumonia X-ray Classifier")
st.write("Classifica se uma radiografia é **Normal** ou possui **Pneumonia**.")

# 📤 Upload
uploaded_file = st.file_uploader("📤 Envie uma imagem de raio-x", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mostrar imagem
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # Pré-processar
    img_resized = image.resize((150, 150))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)  # (1, 150, 150, 3)
    img_array /= 255.0

    # Prever
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    prob = prediction if prediction > 0.5 else 1 - prediction

    # Mostrar resultado
    st.markdown(f"### 🧠 Previsão: **{label}**")
    st.progress(float(prob))
    st.write(f"Probabilidade: `{prob:.2%}`")

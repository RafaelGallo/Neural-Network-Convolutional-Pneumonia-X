import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# ✅ Precisa vir primeiro!
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

st.title("🔍 Pneumonia Detector com CNN")
st.markdown("Faça upload de uma imagem de raio-X de tórax para verificar indícios de pneumonia.")

# 📦 Carregar o modelo (com cache)
@st.cache_resource
def load_cnn_model():
    model = load_model("models/CNN_Classification_1.h5")  # ajuste se estiver em outro caminho
    return model

model = load_cnn_model()

# 📷 Upload da imagem
uploaded_file = st.file_uploader("📁 Escolha a imagem (formato PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Abrir e exibir imagem
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem carregada", use_container_width=True)

    # Pré-processamento
    image = image.resize((150, 150))  # mesmo tamanho usado no treino
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape final: (1, 150, 150, 1)

    # 🧠 Previsão
    prediction = model.predict(img_array)[0][0]

    # Resultado
    st.markdown("### 🧪 Resultado da Classificação:")
    if prediction >= 0.5:
        st.error(f"**Classe Prevista:** PNEUMONIA\n\n🔺 Probabilidade: {prediction*100:.2f}%\n\n⚠️ Indício de pneumonia detectado. Procure um especialista.")
    else:
        st.success(f"**Classe Prevista:** NORMAL\n\n✅ Probabilidade: {(1 - prediction)*100:.2f}%\n\n🟢 Nenhum indício de pneumonia detectado.")

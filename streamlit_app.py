import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------------
# ⚙️ CONFIGURAÇÃO INICIAL
# -------------------------
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Detecção de Pneumonia por Raio-X")
st.write("Este aplicativo usa uma CNN para classificar imagens de raio-X como **NORMAL** ou **PNEUMONIA**.")

# -------------------------
# 🔄 FUNÇÃO PARA CARREGAR MODELO
# -------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("models/CNN_Classification_1.h5")
        return model
    except Exception as e:
        st.error("❌ Erro ao carregar o modelo. Verifique o caminho ou se há camadas personalizadas.")
        st.stop()

# Carregar o modelo
model = load_cnn_model()

# -------------------------
# 📤 UPLOAD DE IMAGEM
# -------------------------
uploaded_file = st.file_uploader("Faça o upload de uma imagem de raio-X no formato PNG ou JPG", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Mostrar imagem
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # -------------------------
    # 🔄 PRÉ-PROCESSAMENTO
    # -------------------------
    img_resized = image.resize((150, 150))                # mesmo tamanho da CNN
    img_array = img_to_array(img_resized)                 # shape: (150,150,1)
    img_array = np.expand_dims(img_array, axis=0)         # shape: (1,150,150,1)
    img_array /= 255.0                                     # normalização

    # -------------------------
    # 🤖 PREVISÃO
    # -------------------------
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    prob = prediction if prediction > 0.5 else 1 - prediction

    # -------------------------
    # 📊 EXIBIR RESULTADO
    # -------------------------
    st.markdown(f"### 🧠 Previsão: **{label}**")
    st.progress(float(prob))
    st.write(f"Probabilidade: `{prob:.2%}`")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Configuração da página – deve ser a PRIMEIRA chamada!
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Função para carregar o modelo
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("models/CNN_Classification_1.h5")
    except Exception as e:
        st.error("❌ Erro ao carregar o modelo. Verifique o caminho ou se há camadas personalizadas.")
        st.stop()
    return model

# Função para preprocessar imagem
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("L")  # Converte para grayscale
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array / 255.0  # Normaliza
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão de batch
    img_array = np.repeat(img_array, 3, axis=-1)   # Converte para RGB com 3 canais
    return img_array

# Título
st.title("🩺 Pneumonia Detector - CNN")

# Instruções
st.markdown("Envie uma imagem de raio-X de tórax para verificar a probabilidade de pneumonia.")

# Upload de imagem
uploaded_file = st.file_uploader("Upload do raio-X", type=["png", "jpg", "jpeg"])

# Carrega o modelo
model = load_cnn_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", use_column_width=True)

    if st.button("🔍 Diagnosticar"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

        if prediction >= 0.5:
            st.error(f"⚠️ Resultado: **PNEUMONIA** com probabilidade de {prediction:.2%}")
        else:
            st.success(f"✅ Resultado: **NORMAL** com probabilidade de {(1 - prediction):.2%}")

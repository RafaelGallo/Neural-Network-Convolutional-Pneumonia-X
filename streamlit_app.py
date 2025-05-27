import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Configuração inicial
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Detecção de Pneumonia por Raio-X")
st.write("Este app usa uma rede neural para classificar raio-X como **NORMAL** ou **PNEUMONIA**.")

# Função para carregar o modelo
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("models/CNN_Classification_1.h5")
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        st.stop()

model = load_cnn_model()

# Upload da imagem
uploaded_file = st.file_uploader("📤 Envie um raio-X (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mostra imagem
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem enviada", use_container_width=True)

    # Redimensiona imagem para (20, 25)
    resized_img = image.resize((20, 25))
    img_array = img_to_array(resized_img) / 255.0  # Normaliza
    img_flatten = img_array.flatten()              # (500,)

    # Confirma shape esperado
    expected_shape = model.input_shape[1]
    if img_flatten.shape[0] != expected_shape:
        st.error(f"❌ Entrada incompatível: o modelo espera shape ({expected_shape},) e recebeu ({img_flatten.shape[0]},)")
        st.stop()

    # Previsão
    input_array = np.expand_dims(img_flatten, axis=0)  # (1, 500)
    prediction = model.predict(input_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    prob = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🔍 Previsão: **{label}**")
    st.progress(float(prob))
    st.write(f"Probabilidade: `{prob:.2%}`")

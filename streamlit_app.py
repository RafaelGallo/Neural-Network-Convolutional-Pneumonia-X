import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Configuração inicial
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Detecção de Pneumonia por Raio-X")
st.write("Este app usa uma CNN para classificar raio-X como **NORMAL** ou **PNEUMONIA**.")

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
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Imagem enviada", use_container_width=True)

    # Preprocessamento
    resized_img = image.resize((150, 150))
    img_array = img_to_array(resized_img)
    img_array = img_array / 255.0

    # Adiciona batch dimension e canal
    img_array = np.expand_dims(img_array, axis=0)       # (1, 150, 150)
    img_array = np.expand_dims(img_array, axis=-1)      # (1, 150, 150, 1)

    # Verifica o input shape do modelo
    expected_shape = model.input_shape
    if img_array.shape[1:] != expected_shape[1:]:
        st.error(f"❌ Shape incompatível: modelo espera {expected_shape[1:]}, mas recebeu {img_array.shape[1:]}")
        st.stop()

    # Predição
    prediction = model.predict(img_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    prob = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🔍 Previsão: **{label}**")
    st.progress(float(prob))
    st.write(f"Probabilidade: `{prob:.2%}`")

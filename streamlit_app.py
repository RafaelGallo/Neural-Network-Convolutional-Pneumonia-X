import streamlit as st
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# =========================
# 1. Configuração da página
# =========================
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🪜 Pneumonia Detection via Chest X-ray")

# =========================
# 2. Carregamento do Modelo
# =========================
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("models/CNN_Classification_1.h5")
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None

model = load_cnn_model()

# =========================
# 3. Upload da Imagem
# =========================
uploaded_file = st.file_uploader("Envie uma imagem de raio-X de tórax (formato PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Mostrar imagem original
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem enviada", use_container_width=True)

    # =========================
    # 4. Preprocessamento
    # =========================
    resized_img = image.resize((500, 500))  # modelo espera (500, 500, 1)
    img_array = img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (500, 500, 1)
    input_array = np.expand_dims(img_array, axis=0)  # (1, 500, 500, 1)

    # =========================
    # 5. Previsão
    # =========================
    prediction = model.predict(input_array)[0][0]
    label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
    prob = prediction if prediction >= 0.5 else 1 - prediction

    # =========================
    # 6. Resultado
    # =========================
    st.markdown("---")
    st.subheader("Resultado da Classificação:")
    st.write(f"**Classe Prevista:** `{label}`")
    st.write(f"**Probabilidade:** `{prob:.2%}`")
    if label == "PNEUMONIA":
        st.error("⚠️ Indício de Pneumonia detectado. Procure um especialista.")
    else:
        st.success("✅ Sem indícios de Pneumonia detectados.")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------------
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Detecção de Pneumonia por Raio-X")
st.write("Este app usa uma rede neural para classificar raio-X como **NORMAL** ou **PNEUMONIA**.")

# -------------------------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("models/CNN_Classification_1.h5")
        return model
    except Exception as e:
        st.error("❌ Erro ao carregar o modelo. Verifique o caminho do .h5 ou dependências.")
        st.stop()

model = load_cnn_model()

# -------------------------
uploaded_file = st.file_uploader("📤 Faça upload de um raio-X (JPG, PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Imagem enviada", use_column_width=True)

    # Redimensiona e normaliza
    img_resized = image.resize((150, 150))
    img_array = img_to_array(img_resized)          # shape: (150,150,1)
    img_array = img_array / 255.0                   # normalização
    img_flatten = img_array.flatten()               # shape: (150*150 = 22500,)

    # Ajustar para entrada: (1, N)
    input_array = np.expand_dims(img_flatten, axis=0)  # shape: (1, 22500)

    # Confirma forma esperada pelo modelo
    model_input_shape = model.input_shape[1]
    if input_array.shape[1] != model_input_shape:
        st.error(f"❌ Erro de compatibilidade: o modelo espera entrada com shape ({model_input_shape},), mas recebeu ({input_array.shape[1]},)")
        st.stop()

    # -------------------------
    prediction = model.predict(input_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    prob = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🔍 Previsão: **{label}**")
    st.progress(float(prob))
    st.write(f"Probabilidade: `{prob:.2%}`")

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ⚙️ Configuração inicial da página (deve ser a 1ª linha Streamlit)
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# 🧠 Carrega o modelo CNN treinado
@st.cache_resource
def load_cnn_model():
    model = load_model("models/CNN_Classification_1.h5")
    return model

model = load_cnn_model()

# 🎨 Estilo e título
st.title("🩺 Pneumonia X-Ray Classifier")
st.markdown("Upload a chest X-ray image to detect if the patient likely has **pneumonia** or is **normal**.")

# 📤 Upload de imagem
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

# 🖼️ Visualização + Previsão
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)

    # 🔍 Processamento da imagem
    image = Image.open(uploaded_file)
    expected_shape = model.input_shape  # (None, 224, 224, 1) ou (None, 224, 224, 3)

    if expected_shape[1:] == (224, 224, 3):
        image = image.convert("RGB")
        img_array = np.asarray(image.resize((224, 224))) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
    else:
        image = image.convert("L")
        img_array = np.asarray(image.resize((224, 224))) / 255.0
        img_array = img_array.reshape(1, 224, 224, 1)

    # 📊 Previsão
    prediction = model.predict(img_array)[0][0]

    # 🧾 Resultado
    st.subheader("🔍 Prediction Result:")
    if prediction >= 0.5:
        st.error(f"**{prediction:.2%} chance of Pneumonia**")
    else:
        st.success(f"**{1 - prediction:.2%} chance of Normal lungs**")

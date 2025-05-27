import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from reportlab.pdfgen import canvas
import datetime
import pandas as pd
from io import BytesIO

# Configuração da página
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Traduções
translations = {
    "pt": {
        "title": "Classificador de Pneumonia por Raio-X",
        "upload": "📤 Envie a imagem de raio-X",
        "language": "🌐 Idioma",
        "button": "🔍 Classificar",
        "result": "Resultado da Classificação:",
        "class": "Classe Prevista",
        "probability": "Probabilidade",
        "alert": "⚠️ Indício de Pneumonia detectado. Procure um especialista.",
        "normal": "✅ Sem indício de Pneumonia.",
        "download": "⬇️ Baixar Laudo em PDF",
        "history": "📚 Histórico de Classificações",
    },
    "en": {
        "title": "Pneumonia X-ray Classifier",
        "upload": "📤 Upload X-ray image",
        "language": "🌐 Language",
        "button": "🔍 Classify",
        "result": "Classification Result:",
        "class": "Predicted Class",
        "probability": "Probability",
        "alert": "⚠️ Indication of Pneumonia detected. Please consult a specialist.",
        "normal": "✅ No indication of Pneumonia.",
        "download": "⬇️ Download PDF Report",
        "history": "📚 Classification History",
    }
}

# Carregar modelo
@st.cache_resource
def load_cnn_model():
    return load_model("models/CNN_Classification_1.h5")

model = load_cnn_model()

# Definir idioma
lang = st.selectbox("🌐 Language / Idioma", options=["pt", "en"], index=0)
t = translations[lang]

st.title(t["title"])

uploaded_file = st.file_uploader(t["upload"], type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image_resized = image.resize((500, 500))
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=(0, -1))  # (1, 500, 500, 1)
    st.image(image_resized, caption="X-ray", use_column_width=True)

    if st.button(t["button"]):
        prediction = model.predict(img_array)[0][0]
        label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
        probability = prediction if prediction >= 0.5 else 1 - prediction

        st.markdown(f"### {t['result']}")
        st.write(f"**{t['class']}:** {label}")
        st.write(f"**{t['probability']}:** {probability * 100:.2f}%")
        
        if label == "PNEUMONIA":
            st.error(t["alert"])
        else:
            st.success(t["normal"])

        # Salvar histórico
        history = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "class": label,
            "probability": f"{probability * 100:.2f}%"
        }])

        history_file = "classification_history.csv"
        if os.path.exists(history_file):
            old = pd.read_csv(history_file)
            history = pd.concat([old, history], ignore_index=True)

        history.to_csv(history_file, index=False)

        # Gerar PDF
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 800, f"{t['result']}")
        c.setFont("Helvetica", 12)
        c.drawString(100, 770, f"{t['class']}: {label}")
        c.drawString(100, 750, f"{t['probability']}: {probability * 100:.2f}%")
        c.drawString(100, 730, f"Data/Hora: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        c.save()

        st.download_button(
            label=t["download"],
            data=buffer.getvalue(),
            file_name="laudo_pneumonia.pdf",
            mime="application/pdf"
        )

# Mostrar histórico
if os.path.exists("classification_history.csv"):
    st.subheader(t["history"])
    df_hist = pd.read_csv("classification_history.csv")
    st.dataframe(df_hist.tail(10), use_container_width=True)

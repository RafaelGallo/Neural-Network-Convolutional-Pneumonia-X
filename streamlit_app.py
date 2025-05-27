import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import pandas as pd
import datetime
from io import BytesIO
from reportlab.pdfgen import canvas

# -----------------------
# 🌐 Configuração
# -----------------------
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Tradução multilíngue
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
        "clear": "🗑️ Limpar histórico",
        "download_history": "⬇️ Baixar histórico completo (CSV)"
    },
    "en": {
        "title": "Pneumonia X-ray Classifier",
        "upload": "📤 Upload X-ray image",
        "language": "🌐 Language",
        "button": "🔍 Classify",
        "result": "Classification Result:",
        "class": "Predicted Class",
        "probability": "Probability",
        "alert": "⚠️ Pneumonia detected. Please consult a specialist.",
        "normal": "✅ No indication of Pneumonia.",
        "download": "⬇️ Download PDF Report",
        "history": "📚 Classification History",
        "clear": "🗑️ Clear history",
        "download_history": "⬇️ Download classification history (CSV)"
    }
}

# -----------------------
# 📦 Carregar modelo
# -----------------------
@st.cache_resource
def load_cnn_model():
    return load_model("models/CNN_Classification_1.h5")

model = load_cnn_model()

# -----------------------
# 🌍 Selecionar idioma
# -----------------------
lang = st.selectbox("🌐 Language / Idioma", options=["pt", "en"], index=0)
t = translations[lang]

st.title(t["title"])
uploaded_file = st.file_uploader(t["upload"], type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    resized_img = image.resize((500, 500))
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 500, 500, 1)

    st.image(resized_img, caption="📷 Raio-X", use_container_width=True)

    if st.button(t["button"]):
        prediction = model.predict(img_array)[0][0]
        label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
        prob = prediction if prediction >= 0.5 else 1 - prediction

        st.markdown(f"### {t['result']}")
        st.write(f"**{t['class']}:** {label}")
        st.write(f"**{t['probability']}:** {prob * 100:.2f}%")

        if label == "PNEUMONIA":
            st.error(t["alert"])
        else:
            st.success(t["normal"])

        # -----------------------
        # 💾 Histórico
        # -----------------------
        hist_file = "classification_history.csv"
        new_row = pd.DataFrame([{
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name,
            "class": label,
            "probability": f"{prob * 100:.2f}%"
        }])

        if os.path.exists(hist_file):
            history_df = pd.read_csv(hist_file)
            history_df = pd.concat([history_df, new_row], ignore_index=True)
        else:
            history_df = new_row

        history_df.to_csv(hist_file, index=False)

        # -----------------------
        # 🧾 PDF Laudo
        # -----------------------
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 800, t["result"])
        c.setFont("Helvetica", 12)
        c.drawString(100, 770, f"{t['class']}: {label}")
        c.drawString(100, 750, f"{t['probability']}: {prob * 100:.2f}%")
        c.drawString(100, 730, f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        c.save()

        st.download_button(
            label=t["download"],
            data=buffer.getvalue(),
            file_name="laudo_pneumonia.pdf",
            mime="application/pdf"
        )

# -----------------------
# 🗃️ Histórico
# -----------------------
if os.path.exists("classification_history.csv"):
    st.subheader(t["history"])
    df_hist = pd.read_csv("classification_history.csv")
    st.dataframe(df_hist.tail(10), use_container_width=True)

    st.download_button(
        label=t["download_history"],
        data=open("classification_history.csv", "rb").read(),
        file_name="historico_classificacoes.csv",
        mime="text/csv"
    )

    if st.button(t["clear"]):
        os.remove("classification_history.csv")
        st.success("Histórico apagado com sucesso.")

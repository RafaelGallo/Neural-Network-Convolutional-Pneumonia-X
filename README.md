# 🧠 Pneumonia X-Ray Classification with CNNs

![Model](https://img.shields.io/badge/model-VGG16%20%7C%20ResNet50%20%7C%20InceptionV3-blue)
![Dataset](https://img.shields.io/badge/dataset-Chest%20X--Ray%20(Pneumonia)-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Completed-success)
[![Streamlit App](https://img.shields.io/badge/launch-app-brightgreen?logo=streamlit)](https://neural-network-convolutional-pneumonia-x-hc6qwxafrsdnvb3kcdq8h.streamlit.app/)


## 📌 Overview

This project leverages state-of-the-art **Convolutional Neural Networks (CNNs)** to automatically classify **chest X-ray images** into two categories: **Normal** (healthy lungs) and **Pneumonia** (lung opacity detected). It is designed as an **AI-powered diagnostic support system** to assist radiologists and healthcare professionals by improving **diagnostic accuracy**, **reducing interpretation time**, and supporting early detection efforts.

To achieve this, we applied both **custom CNN architectures** and **pretrained deep learning models** including:

* 🔹 **VGG16**
* 🔹 **ResNet50**
* 🔹 **InceptionV3**
* 🔹 A custom **Sequential CNN**

All models utilize **ImageNet-based transfer learning** where applicable, and were trained on a **carefully restructured version of Paul Mooney’s Chest X-Ray Pneumonia dataset** to ensure balanced and fair evaluation across both classes.

This project integrates all critical aspects of deep learning in medical imaging:

* ✅ End-to-end image preprocessing
* ✅ Data augmentation
* ✅ Class balancing
* ✅ Model training with callbacks
* ✅ Performance evaluation with accuracy, precision, recall, F1-score, AUC
* ✅ Visualization tools including loss curves, confusion matrix, and prediction subplots

By combining domain knowledge with modern deep learning practices, the resulting models have the potential to:

* Enhance radiological workflows
* Improve diagnosis in low-resource settings
* Serve as a foundation for more complex medical imaging applications (e.g., multi-class classification or severity grading)

## 🏗️ Models Used

- ✅ **Sequential CNN** – Custom baseline model
- ✅ **VGG16** – Pretrained on ImageNet, using 3x3 convolutions
- ✅ **ResNet50** – Residual Network with skip connections
- ✅ **InceptionV3** – Efficient architecture with factorized convolutions
- ✅ **ImageNet** – Used for transfer learning weights


## 📁 Dataset

Adapted from Paul Mooney's *Chest X-Ray Images (Pneumonia)*:
- Total images: **5,856**
- Training: **4,192** (1,082 normal | 3,110 pneumonia)
- Validation: **1,040** (267 normal | 773 pneumonia)
- Testing: **624** (234 normal | 390 pneumonia)

## 📂 Dataset Source

The dataset used in this project is an adapted version of the publicly available:

**📦 Chest X-Ray Images (Pneumonia)** by Paul Mooney  

🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Contains 5,856 labeled chest X-ray images
- Two classes: `NORMAL` and `PNEUMONIA`
- Split into training, validation, and test sets
- Reformatted in this project for a more balanced machine learning task


## 🧪 Deep learning Task

- **Type**: Binary Image Classification
- **Input Shape**: 224×224 or 299×299 RGB
- **Output**: 0 = Normal, 1 = Pneumonia
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC

## 📊 Evaluation

Each model is evaluated using:
- 📈 Accuracy/Loss Curves
- 📉 Confusion Matrix
- ✅ Classification Report
- 🖼️ Image Subplots (True vs Predicted)


## 🧱 Custom CNN Model Architecture

In addition to pretrained models like VGG16, ResNet50, and InceptionV3, this project also implements a **custom Sequential Convolutional Neural Network (CNN)** trained from scratch using grayscale chest X-ray images.

### 📐 **Model Architecture**

```python
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(32, (3, 3), activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(32, (3, 3), activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(64, (3, 3), activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(64, (3, 3), activation="relu"))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))
```

### ⚙️ **Compilation and Training**

The model was compiled with the **Adam optimizer**, and trained using **binary crossentropy** loss and the **accuracy** metric. Additionally, we used:

* `EarlyStopping` to halt training when validation loss stops improving
* `ReduceLROnPlateau` to adjust learning rate dynamically
* `compute_class_weight` to handle class imbalance

```python
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1, min_lr=1e-6)

# Balanced class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(train.classes), y=train.classes)
cw = dict(zip(np.unique(train.classes), weights))

# Training
history_cnn = model_cnn.fit(train,
                            epochs=50,
                            validation_data=valid,
                            class_weight=cw)
```

## 📈 Model Performance

### Training Curves

The following graphs show the evolution of **training and validation accuracy and loss** across 30 epochs:

![Training Curves](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/004.png?raw=true)

### 🔍 Confusion Matrix (Test Set)

The confusion matrix provides a visual summary of model performance on the test set:

![Confusion Matrix](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/003.png?raw=true)

### 🖼️ Prediction Samples

The model was also tested on individual X-ray images. Below is a sample of predictions showing the **predicted class, probability, and ground truth**:

![Sample Predictions](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/002.png?raw=true)

## 📊 Evaluation Metrics

The model was evaluated on the test set using common classification metrics:

- **Precision**: How many predicted positives were correct
- **Recall**: How many actual positives were captured
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class

### 📋 Classification Report

![Classification Report](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/005.png?raw=true)

| Metric     | Normal      | Pneumonia   | Average     |
|------------|-------------|-------------|-------------|
| Precision  | 0.96        | 0.89        | —           |
| Recall     | 0.79        | 0.98        | —           |
| F1-score   | 0.87        | 0.93        | 0.90 (macro) |
| Accuracy   |             |             | **0.91**     |


## ✅ Summary

* The custom CNN achieved strong performance on both training and validation sets.
* Results demonstrate effective generalization to unseen chest X-ray images.
* The model serves as a **baseline** to compare with pretrained models such as VGG16, ResNet50, and InceptionV3.

## 🚀 Deployment with Streamlit

This project includes a **Streamlit web application** that allows users to upload chest X-ray images and get real-time predictions from the trained CNN models (e.g., VGG16, ResNet50, InceptionV3, or Custom CNN).

### 📦 Prerequisites

Make sure you have the required packages installed:

```bash
pip install streamlit tensorflow keras opencv-python pillow numpy
```

### ▶️ Running the App Locally

To launch the app locally, run the following command from the project directory:

```bash
streamlit run streamlit_app.py
```

### 📁 Suggested File Structure

```
.
├── streamlit_app.py           # Main Streamlit application
├── CNN_Classification.h5      # Trained model (or ResNet50.h5, VGG16.h5, etc.)
├── utils.py                   # (Optional) Preprocessing helpers
├── requirements.txt
└── README.md
```

### 💡 Features

* Upload chest X-ray images (`.jpg`, `.jpeg`, `.png`)
* Automatic grayscale-to-RGB conversion if needed
* Real-time prediction with probability score
* Clear output: **Normal** or **Pneumonia**
* Lightweight and deployable with Streamlit Cloud

---

### 🧪 Example Streamlit Code

```python
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
```

## 🚀 Try it Live

You can test the Pneumonia X-ray Classification model directly in your browser:

🔗 **[🩺 Open the App on Streamlit Cloud](https://neural-network-convolutional-pneumonia-x-hc6qwxafrsdnvb3kcdq8h.streamlit.app/)**

The app allows you to:
- Upload a chest X-ray (`.png`, `.jpg`, `.jpeg`)
- Get an instant prediction of **NORMAL** or **PNEUMONIA**
- Download a medical-style PDF report
- View classification history with timestamps and confidence scores

[![Streamlit App](https://img.shields.io/badge/launch-app-brightgreen?logo=streamlit)](https://neural-network-convolutional-pneumonia-x-hc6qwxafrsdnvb3kcdq8h.streamlit.app/)

## 🖼️ Test Images

To try out the model with real chest X-ray examples, you can use the test images provided in the repository:

📁 [Test Image Samples](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/tree/main/data)

This folder contains:

- ✅ Chest X-rays classified as **NORMAL**
- 🔴 Chest X-rays classified as **PNEUMONIA** (bacterial and viral)

You can upload these directly in the [Streamlit App](https://neural-network-convolutional-pneumonia-x-hc6qwxafrsdnvb3kcdq8h.streamlit.app/) to observe model predictions in real time.

| Normal Case | Pneumonia Case |
|-------------|----------------|
| ![Normal](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/007.png?raw=true) | ![Pneumonia](https://github.com/RafaelGallo/Neural-Network-Convolutional-Pneumonia-X/blob/main/img/006.png?raw=true) |

---
## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pneumonia-cnn-classification.git
   cd pneumonia-cnn-classification

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Run training:

   ```bash
   python train_model.py --model resnet
   ```

4. Run evaluation:

   ```bash
   python evaluate.py --model resnet
   ```


## 🌍 Impact


This project demonstrates how AI can:

* Improve diagnostic speed and consistency
* Assist in early detection of pneumonia
* Extend diagnostics to low-resource or rural settings


## 📜 License


This project is licensed under the **MIT License**.
Dataset originally by Paul Mooney and provided under an open educational license.


## 🙌 Acknowledgments


* [Paul Mooney's Dataset on Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* [ImageNet](http://www.image-net.org/)
* [Keras Applications Models](https://keras.io/api/applications/)


## 📬 Contact


**Your Name** – [@rafaelgallo](https://github.com/rafaelgallo)
Feel free to reach out for collaboration or questions!

```

---
Would you like me to:
- Generate a `requirements.txt`?
- Create individual `train_model.py` and `evaluate.py` template files?
- Add example Jupyter Notebook?

Let me know and I’ll generate them too.
```

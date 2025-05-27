# 🧠 Pneumonia X-Ray Classification with CNNs

![Model](https://img.shields.io/badge/model-VGG16%20%7C%20ResNet50%20%7C%20InceptionV3-blue)
![Dataset](https://img.shields.io/badge/dataset-Chest%20X--Ray%20(Pneumonia)-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Completed-success)

## 📌 Overview

This project leverages **Convolutional Neural Networks (CNNs)** to automatically classify chest X-ray images as either **Normal** or **Pneumonia**. It aims to assist radiologists with an AI-driven diagnostic tool using powerful deep learning models — trained and evaluated on a balanced version of the well-known **Chest X-Ray Images (Pneumonia)** dataset.

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

# 🧠 Pneumonia X-Ray Classification with CNNs

![Model](https://img.shields.io/badge/model-VGG16%20%7C%20ResNet50%20%7C%20InceptionV3-blue)
![Dataset](https://img.shields.io/badge/dataset-Chest%20X--Ray%20(Pneumonia)-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Completed-success)

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

# 🌿 CropCare AI
**Deep Learning Powered Plant Disease Detection System**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-green)

CropCare AI is a web application that helps farmers identify crop diseases by uploading leaf images. It uses a fine-tuned **EfficientNetB4** model (transfer learning) to classify diseases across **9 crop types: Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato**.

## 📊 Results
- **Test Accuracy:** 91%
- **Disease Classes:** 26
- **Inference latency:** less than 2 seconds per prediction via Flask REST API

---

## 📌 What This Project Shows
- End-to-end deep learning workflow for image classification
- Transfer learning with EfficientNetB4
- Deployment of a trained model in a Flask web application
- Real-world agriculture use case with portfolio-ready presentation

---

## 📸 Project Demo
*(Screenshots of the working application)*

| **Home Page** |
<img width="1880" height="882" alt="image" src="https://github.com/user-attachments/assets/75b4118b-240f-405d-b07d-22e9f4fe6f6a" />

| **Result Page** |
<img width="1869" height="830" alt="image" src="https://github.com/user-attachments/assets/a40b323b-6af2-4890-9c4f-683a0c661fe1" />

---

## 📊 Model Performance
- **Final Test Accuracy:** 91% across 4,913 test images and 26 disease classes
- **Architecture:** EfficientNetB4 transfer learning with a custom classification head
- **Strongest classes:** Grape Leaf Blight, Corn Healthy, and Grape Black Rot (all at or above 99% F1-score)
- **Known limitations:** Tomato Early Blight had lower recall and was sometimes confused with visually similar late blight classes. Tomato-healthy and Cherry-healthy leaves were also occasionally confused.
- Full training run and classification report are available in `crop-care-model-training.ipynb`

## 🧠 Model Architecture
Built using **TensorFlow** and **Keras** with transfer learning for stronger performance on a limited dataset.

- **Model Type:** EfficientNetB4 pretrained on ImageNet with a custom classification head
- **Approach:** Transfer learning with added Dense and Dropout layers for 26-class classification
- **Dataset:** Curated subset of the PlantVillage dataset with 22,812 training images, 4,880 validation images, and 4,913 test images
- **Classes Supported:** 26 disease and healthy classes across 9 crop types

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, NumPy
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **Image Processing:** Pillow (PIL)

---

## ⚙️ Installation & Usage
> **Important:** The trained model file (`cropcare_model.keras`) is not included because of GitHub file size limits.

To run this project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mukta0702/CropCare-AI.git
   cd CropCare-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the model**
   - Train the model yourself using the provided notebook and dataset workflow
   - Or place a pre-trained `.keras` model inside the `backend/` folder
   - Rename the model file to `cropcare_model.keras`

4. **Run the app**
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:5000`

---

## 👤 Author
**Mukta**
- **GitHub:** [@Mukta0702](https://github.com/Mukta0702)
- **LinkedIn:** [Mukta Lakkawar](https://www.linkedin.com/in/mukta-lakkawar-055136288)

*Built as a deep learning project to explore AI in agriculture.*

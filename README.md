# 🌿 CropCare AI
**Deep Learning Powered Plant Disease Detection System**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-green)

CropCare AI is a web application that helps farmers identify crop diseases by uploading leaf images. It uses a fine-tuned **EfficientNetB4** model (transfer learning) to classify diseases across **9 crop types — Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato**.

## 📊 Results
- **Test Accuracy:** 91%
- **Disease Classes:** 26
- **Inference latency:** <2 seconds per prediction via Flask REST API

---

## 📸 Project Demo
*(Screenshots of the working application)*

| **Home Page** |
<img width="1880" height="882" alt="image" src="https://github.com/user-attachments/assets/75b4118b-240f-405d-b07d-22e9f4fe6f6a" />


| **Result Page** |
<img width="1869" height="830" alt="image" src="https://github.com/user-attachments/assets/a40b323b-6af2-4890-9c4f-683a0c661fe1" />

---

## 📊 Model Performance

- **Final Test Accuracy: 91%** (4,913 test images across 26 disease classes)
- **Architecture:** EfficientNetB4 (transfer learning), fine-tuned on a curated PlantVillage subset
- **Strongest classes:** Grape Leaf Blight, Corn Healthy, Grape Black Rot (all ≥99% F1-score)
- **Known limitations:** Tomato Early Blight had lower recall (33%), often confused with Tomato/Potato Late Blight due to visual similarity in leaf lesions. Tomato-healthy and Cherry-healthy leaves were also frequently confused, suggesting the augmentation pipeline could better preserve species-specific leaf texture.
- Full training run and classification report available in `crop-care-model-training.ipynb`

## 🧠 Model Architecture

Built using **TensorFlow & Keras** with transfer learning for stronger performance on a limited dataset.

* **Model Type:** EfficientNetB4 (pretrained on ImageNet) with a custom classification head
* **Approach:** Transfer learning — the pretrained EfficientNetB4 base extracts features, with **Dense** and **Dropout** layers added on top for the 26-class classification task
* **Dataset:** Trained on a curated subset of the **PlantVillage Dataset** — 22,812 training images, 4,880 validation images, 4,913 test images
* **Classes Supported:** 26 disease/healthy classes across **Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato**
---

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras, NumPy
* **Backend:** Flask (Python)
* **Frontend:** HTML5, CSS3, JavaScript
* **Image Processing:** Pillow (PIL)

---

## ⚙️ Installation & Usage
> **⚠️ Important:** The trained model file (`cropcare_model.keras`) is not included here due to GitHub's file size limits.

To run this project on your machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Mukta0702/CropCare-AI.git](https://github.com/Mukta0702/CropCare-AI.git)
    cd CropCare-AI
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup the Model**
    * Since the model file is too large for GitHub, you must either:
        * Train the model yourself using the provided dataset.
        * **OR** Download a pre-trained `.keras` model and place it in the `backend/` folder.
    * Rename the file to: `cropcare_model.keras`

4.  **Run the App**
    ```bash
    python app.py
    ```
    Open your browser to: `http://127.0.0.1:5000`

---
## 👤 Author
**Mukta**
* **GitHub:** [@Mukta0702](https://github.com/Mukta0702)
* **LinkedIn:** [Mukta Lakkawar](https://www.linkedin.com/in/mukta-lakkawar-055136288)


*Built as a Deep Learning project to explore AI in Agriculture.*

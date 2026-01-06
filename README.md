# 🌿 CropCare AI
**Deep Learning Powered Plant Disease Detection System**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-green)

CropCare AI is a web application that helps farmers identify crop diseases by uploading leaf images. It uses a custom-trained **Convolutional Neural Network (CNN)** to classify diseases in **Potatoes, Tomatoes, and Peppers** with high accuracy.

---

## 📸 Project Demo
*(Screenshots of the working application)*

| **Home Page** |
<img width="1880" height="882" alt="image" src="https://github.com/user-attachments/assets/75b4118b-240f-405d-b07d-22e9f4fe6f6a" />


| **Result Page** |
<img width="1869" height="830" alt="image" src="https://github.com/user-attachments/assets/a40b323b-6af2-4890-9c4f-683a0c661fe1" />

## 🧠 Model Architecture
The model was built from scratch using **TensorFlow & Keras**.

* **Model Type:** Custom Sequential CNN
* **Layers:** * Multiple **Conv2D** layers for feature extraction.
    * **MaxPooling2D** layers for dimensionality reduction.
    * **Flatten** & **Dense** layers for classification.
    * **Dropout** layers to prevent overfitting.
* **Dataset:** Trained on a curated subset of the **PlantVillage Dataset**.
* **Classes Supported:**
    * **Potato:** Early Blight, Late Blight, Healthy
    * **Tomato:** Bacterial Spot, Early Blight, Late Blight, Healthy, etc.
    * **Pepper:** Bacterial Spot, Healthy

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

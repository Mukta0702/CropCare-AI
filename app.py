import os
import io
import logging
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageFile
from datetime import datetime

# --- 1. MODEL CONFIGURATION ---
# EfficientNetB4 requires specific preprocessing and 380x380 input
from tensorflow.keras.applications.efficientnet import preprocess_input

# Fix for Matplotlib server-side rendering (avoids crashes)
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

# Constants
MODEL_FILE_NAME = "cropcare_model.keras"  # Ensure this matches your file
CLASSES_FILE = "classes.txt"
UPLOAD_DIR = "static/uploads"
DEFAULT_IMAGE_SIZE = (380, 380)  # STRICTLY for EfficientNetB4
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CropCare-Backend")

# Flask Setup
os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# --- 2. DETAILED REMEDIES DATA ---
# This maps your class names to specific agricultural advice.
REMEDIES = {
    "Apple___Apple_scab": "Rake and destroy fallen leaves. Apply fungicides like Captan or Myclobutanil early in the season.",
    "Apple___Black_rot": "Prune dead limbs and remove mummified fruit. Treat with fungicides during petal fall.",
    "Apple___healthy": "The plant is healthy. Maintain regular watering and pruning schedules.",
    "Cherry___Powdery_mildew": "Prune for better airflow. Apply sulfur-based fungicides or neem oil.",
    "Cherry___healthy": "Healthy cherry tree. Ensure soil drainage is good.",
    "Corn___Common_rust": "Plant resistant varieties. Apply fungicides if infection occurs early in the season.",
    "Corn___Northern_Leaf_Blight": "Rotate crops and till debris into the soil. Use resistant hybrids.",
    "Corn___healthy": "Healthy corn. Monitor for pests like armyworms.",
    "Grape___Black_rot": "Remove shriveled fruit (mummies). Apply fungicides from bud break through fruit set.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve air circulation by pruning. Apply copper-based fungicides.",
    "Grape___healthy": "Grapevine is healthy. Maintain good trellis support.",
    "Peach___Bacterial_spot": "Avoid overhead watering. Apply copper sprays during dormancy.",
    "Peach___healthy": "Peach tree is healthy. Thin fruit to ensure size and quality.",
    "Pepper_bell___Bacterial_spot": "Use disease-free seeds. Remove infected plants immediately. Apply copper sprays.",
    "Pepper_bell___healthy": "Healthy pepper plant. Ensure consistent watering to prevent blossom end rot.",
    "Potato___Early_blight": "Apply mulch to prevent soil splash. Use fungicides containing chlorothalonil or mancozeb.",
    "Potato___Late_blight": "CRITICAL: Destroy infected plants immediately. Apply fungicides like Ridomil. Avoid overhead irrigation.",
    "Potato___healthy": "Healthy potato plant. Keep tubers covered with soil.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Improve air circulation and avoid nitrogen excess.",
    "Strawberry___healthy": "Healthy strawberry plant. Mulch with straw to keep fruit off soil.",
    "Tomato___Bacterial_spot": "Use copper sprays. Remove infected leaves and avoid working with wet plants.",
    "Tomato___Early_blight": "Stake plants to keep leaves off ground. Mulch soil and rotate crops.",
    "Tomato___Leaf_Mold": "Maximize greenhouse ventilation. Reduce humidity levels below 85%.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with sticky traps or neem oil. Remove infected plants.",
    "Tomato___healthy": "Tomato plant is healthy. Maintain consistent moisture.",
    "Tomato___late_blight": "Highly contagious. Remove plants immediately. Prevent with copper fungicides.",
}

# --- 3. UTILITY FUNCTIONS ---

def load_class_names(path):
    """Loads the class list from text file."""
    if not os.path.exists(path):
        logger.error(f"Classes file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        # We assume the file is now SORTED ALPHABETICALLY
        return [l.strip() for l in f.readlines() if l.strip()]

CLASS_NAMES = load_class_names(CLASSES_FILE)
_model = None

def load_model_globally():
    """Loads the model once at startup."""
    global _model
    if _model: return _model
    
    if os.path.exists(MODEL_FILE_NAME):
        logger.info(f"Loading EfficientNet model from: {MODEL_FILE_NAME}")
        # compile=False allows loading without the optimizer state (faster/safer)
        _model = tf.keras.models.load_model(MODEL_FILE_NAME, compile=False)
        return _model
    else:
        logger.error(f"Model file {MODEL_FILE_NAME} not found.")
        return None

def preprocess_image_for_efficientnet(path):
    """
    Prepares an image specifically for EfficientNetB4.
    1. Resizes to 380x380.
    2. Uses efficientnet.preprocess_input (scales correctly).
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(DEFAULT_IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    
    # EfficientNet specific preprocessing (usually 0-255 or -1 to 1 depending on weights)
    arr = preprocess_input(arr)
    
    # Add batch dimension: (1, 380, 380, 3)
    return np.expand_dims(arr, axis=0)

# --- 4. GRAD-CAM VISUALIZATION ---
def generate_gradcam(img_array, model, save_path):
    """
    Generates a heatmap overlay showing where the model is looking.
    """
    try:
        # Find the last convolutional layer
        target_layer = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                target_layer = layer.name
                break
        
        if not target_layer:
            return None

        # create a gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(target_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Overlay on original image
        img = tf.keras.preprocessing.image.load_img(save_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        # Convert to base64
        buffer = io.BytesIO()
        superimposed_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        return None

# --- 5. ROUTES ---

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")

# --- ADD THIS MISSING BLOCK ---
@app.route("/result", methods=["GET"])
def result_page():
    # We redirect to upload.html because results are shown there dynamically now
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Load Model
    model = load_model_globally()
    if not model:
        return jsonify({"error": "Model not loaded on server."}), 500

    # 2. Validate File
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    # 3. Save File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    try:
        # 4. Preprocess
        x = preprocess_image_for_efficientnet(file_path)

        # 5. Predict
        preds = model.predict(x)
        scores = preds[0]  # Get the array of probabilities
        
        # Get Top 1 Result
        top_index = int(np.argmax(scores))
        top_confidence = float(scores[top_index])
        top_label_raw = CLASS_NAMES[top_index] if top_index < len(CLASS_NAMES) else f"Class {top_index}"
        
        # Get Top 3 Results (For detail)
        top_3_indices = np.argsort(scores)[-3:][::-1]
        top_3_details = []
        for idx in top_3_indices:
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
            prob = float(scores[idx])
            top_3_details.append({"name": name.replace("___", " - ").replace("_", " "), "probability": f"{prob*100:.2f}%"})

        # 6. Generate Heatmap
        heatmap_b64 = generate_gradcam(x, model, file_path)

        # 7. Prepare Base64 Preview of Original
        with open(file_path, "rb") as img_f:
            original_b64 = base64.b64encode(img_f.read()).decode('utf-8')

        # 8. Clean Label and Get Advice
        clean_label = top_label_raw.replace("___", " - ").replace("_", " ")
        advice = REMEDIES.get(top_label_raw, "No specific advice available for this class. Consult an agronomist.")

        # 9. Return Detailed JSON
        return jsonify({
            "status": "success",
            "predicted_class": clean_label,
            "confidence": top_confidence,
            "confidence_percent": f"{top_confidence*100:.2f}%",
            "advice": advice,
            "top_3": top_3_details,
            "image_preview": f"data:image/jpeg;base64,{original_b64}",
            "gradcam_preview": f"data:image/jpeg;base64,{heatmap_b64}" if heatmap_b64 else None
        })

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Pre-load model before starting server
    load_model_globally()
    app.run(host="0.0.0.0", port=5000, debug=True)
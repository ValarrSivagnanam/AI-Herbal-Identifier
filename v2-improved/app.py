import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_herbal_classifier.keras")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

CLASS_NAMES = ['aloevera', 'brahmi', 'lemon_grass', 'tulasi', 'wood_sorel']

IMG_SIZE = 224

# -------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------
def preprocess_image(img):
    """
    Preprocess image for model prediction
    Args:
        img: PIL Image object
    Returns:
        Preprocessed numpy array
    """
    
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    img_array = image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# -------------------------------------------
# API ENDPOINTS
# -------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "model": "Herbal Plant Classifier",
        "version": "1.0",
        "classes": CLASS_NAMES
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict plant class from uploaded image
    
    Expects:
        - Image file in request.files['image']
    
    Returns:
        - predicted_class: Name of predicted plant
        - confidence: Prediction confidence (0-1)
        - all_probabilities: Probabilities for all classes
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "message": "Please upload an image with key 'image'"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "error": "Empty filename",
                "message": "Please select a valid image file"
            }), 400
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        processed_img = preprocess_image(img)
        
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        all_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
        
        sorted_probabilities = dict(
            sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        )
        
        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "all_probabilities": sorted_probabilities
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error processing image"
        }), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Predict plant classes for multiple images
    
    Expects:
        - Multiple image files with keys 'image1', 'image2', etc.
        OR
        - Multiple files with same key 'images'
    
    Returns:
        - List of predictions for each image
    """
    try:
        files = request.files.getlist('images')
        
        if not files or files[0].filename == '':
            files = [request.files[key] for key in request.files.keys()]
        
        if not files:
            return jsonify({
                "error": "No image files provided",
                "message": "Please upload images with key 'images'"
            }), 400
        
        results = []
        
        for idx, file in enumerate(files):
            try:
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                processed_img = preprocess_image(img)
                
                predictions = model.predict(processed_img, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                
                results.append({
                    "image_index": idx,
                    "filename": file.filename,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence * 100:.2f}%"
                })
            
            except Exception as e:
                results.append({
                    "image_index": idx,
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "total_images": len(results),
            "predictions": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error processing batch prediction"
        }), 500

@app.route("/classes", methods=["GET"])
def get_classes():
    """Get list of available plant classes"""
    return jsonify({
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES)
    })

# -------------------------------------------
# RUN SERVER
# -------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Herbal Plant Classification API")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("="*50 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
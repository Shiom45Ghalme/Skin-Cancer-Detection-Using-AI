from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/best_model.h5'
model = None

# Class names
CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]

def load_model():
    global model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")

def preprocess_image(image_file):
    """Preprocess uploaded image"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Preprocess image
        img_array = preprocess_image(file)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        
        # Prepare results
        results = {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': round(confidence, 2),
            'all_probabilities': {
                CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("\n" + "="*60)
    print("🌐 Starting Skin Cancer Detection Web App")
    print("="*60)
    print("\n✓ Open your browser and go to: http://localhost:5000")
    print("\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

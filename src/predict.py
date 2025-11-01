import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

# Skin lesion class names
CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess an image for prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model_path, image_path):
    """Make prediction on a single image"""
    print("=" * 60)
    print("Skin Cancer Detection - Prediction")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    
    # Load and preprocess image
    print(f"\nProcessing image: {image_path}")
    img_array = load_and_preprocess_image(image_path)
    print("✓ Image preprocessed!")
    
    # Make prediction
    print("\nMaking prediction...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll Class Probabilities:")
    print("-" * 60)
    
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
        bar = "█" * int(prob * 50)
        print(f"{class_name:40s} {prob*100:6.2f}% {bar}")
    
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict skin lesion class')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5 or .keras)')
    parser.add_argument('--image', type=str, required=True, help='Path to image for prediction')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    predict_image(args.model, args.image)

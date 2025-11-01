import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from data_preprocessing import DataPreprocessor
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Class names
CLASS_NAMES = [
    'Actinic_keratoses',
    'Basal_cell_carcinoma',
    'Benign_keratosis',
    'Dermatofibroma',
    'Melanocytic_nevi',
    'Melanoma',
    'Vascular_lesions'
]

def evaluate_model():
    print("="*70)
    print("EVALUATING TRAINED MODEL")
    print("="*70)
    
    # Load model
    model_path = r"C:\Users\shiom\OneDrive\Desktop\Project xx\models\best_model.h5"
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    
    # Load test data
    print("\n📊 Loading test data...")
    data_dir = r"C:\Users\shiom\OneDrive\Desktop\Project xx\data"
    preprocessor = DataPreprocessor(data_dir, batch_size=32)
    _, _, test_gen = preprocessor.create_data_generators()
    
    # Evaluate
    print("\n🔍 Evaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*70)
    print("TEST SET RESULTS:")
    print("="*70)
    print(f"Test Loss:      {test_results[0]:.4f}")
    print(f"Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall:    {test_results[3]:.4f}")
    print("="*70)
    
    # Save results
    models_dir = r"C:\Users\shiom\OneDrive\Desktop\Project xx\models"
    results = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(models_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("\n✓ Saved: models/test_results.json")
    
    # Generate predictions for confusion matrix
    print("\n📈 Generating confusion matrix...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print("✓ Saved: models/confusion_matrix.png")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    with open(os.path.join(models_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    print("✓ Saved: models/classification_report.json")
    
    # Print classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/test_results.json")
    print("  - models/confusion_matrix.png")
    print("  - models/classification_report.json")

if __name__ == "__main__":
    evaluate_model()

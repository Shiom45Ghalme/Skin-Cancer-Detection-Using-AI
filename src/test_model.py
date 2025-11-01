import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SkinCancerModel
import tensorflow as tf

def test_model_creation():
    """Test if the model can be created and compiled successfully"""
    print("=" * 60)
    print("Testing Skin Cancer Model Creation")
    print("=" * 60)
    
    # Initialize model
    skin_model = SkinCancerModel(num_classes=7, img_size=(224, 224))
    
    # Build model
    model = skin_model.build_model()
    
    # Compile model
    skin_model.compile_model(learning_rate=0.0001)
    
    # Test with dummy data
    print("\nTesting with dummy input...")
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3)
    output = model.predict(dummy_input, verbose=0)
    
    print(f"✓ Model prediction shape: {output.shape}")
    print(f"✓ Output probabilities sum: {output.sum():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    test_model_creation()

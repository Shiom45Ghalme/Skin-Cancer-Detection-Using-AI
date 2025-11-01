import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras
from model import SkinCancerModel
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self, data_dir, num_classes=7, epochs=30, batch_size=32):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None
        self.model = None
        
    def train(self):
        print("="*70)
        print("STARTING SKIN CANCER DETECTION MODEL TRAINING")
        print("="*70)
        
        print("\n[1/6] Loading and preprocessing data...")
        preprocessor = DataPreprocessor(self.data_dir, batch_size=self.batch_size)
        train_gen, val_gen, test_gen = preprocessor.create_data_generators()
        
        print("\n[2/6] Building model architecture...")
        model_builder = SkinCancerModel(num_classes=self.num_classes, use_pretrained=True)
        self.model = model_builder.build_model()
        model_builder.compile_model(learning_rate=0.0001)
        
        print("\n[3/6] Setting up training callbacks...")
        callbacks = self._create_callbacks()
        
        print("\n[4/6] Starting training...")
        print(f"Training for {self.epochs} epochs...")
        print("This may take 30-60 minutes depending on your hardware")
        print("-" * 70)
        
        self.history = self.model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n[5/6] Evaluating on test set...")
        test_results = self.model.evaluate(test_gen, verbose=1)
        
        print("\n" + "="*70)
        print("TEST SET RESULTS:")
        print("="*70)
        print(f"Test Loss:      {test_results[0]:.4f}")
        print(f"Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"Test Precision: {test_results[2]:.4f}")
        print(f"Test Recall:    {test_results[3]:.4f}")
        print("="*70)
        
        print("\n[6/6] Generating reports and saving model...")
        self._save_results(test_results)
        self._plot_training_history()
        self._save_training_config()
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print("\nSaved files:")
        print("  - models/best_model.h5")
        print("  - models/skin_cancer_model.h5")
        print("  - models/training_history.png")
        print("  - models/test_results.json")
        
        return self.model
    
    def _create_callbacks(self):
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, 'skin_cancer_model.h5'),
                monitor='val_accuracy',
                save_best_only=False,
                verbose=0
            )
        ]
        return callbacks
    
    def _plot_training_history(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'training_history.png'), dpi=300)
        plt.close()
        
    def _save_results(self, test_results):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        results = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(models_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    def _save_training_config(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        config = {
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(models_dir, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    # Create models directory
    models_dir = r"C:\Users\shiom\OneDrive\Desktop\Project xx\models"
    os.makedirs(models_dir, exist_ok=True)
    
    trainer = ModelTrainer(
        data_dir=r"C:\Users\shiom\OneDrive\Desktop\Project xx\data",
        num_classes=7,
        epochs=30,
        batch_size=32
    )
    model = trainer.train()
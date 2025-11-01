import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

class SkinCancerModel:
    def __init__(self, num_classes=7, img_size=(224, 224), use_pretrained=True):
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_pretrained = use_pretrained
        self.model = None
        
    def build_model(self):
        if self.use_pretrained:
            print("Building model with Transfer Learning...")
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            model = models.Sequential([
                layers.Input(shape=(*self.img_size, 3)),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        self.model = model
        print("\n✓ Model architecture created!")
        return model
    
    def compile_model(self, learning_rate=0.0001):
        print("\nCompiling model...")
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print("✓ Model compiled!")
        self.model.summary()
        
    def get_model(self):
        return self.model
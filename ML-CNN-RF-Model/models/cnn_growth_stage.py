"""
CNN model for agricultural growth stage classification using Sentinel-2 data.
Enhanced architecture with attention mechanisms and batch normalization.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


def build_cnn_model(input_shape=(256, 256, 5), num_classes=3):
    """
    Build a CNN model for growth stage classification.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes (growth stages)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling instead of Flatten to reduce parameters
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_attention_cnn_model(input_shape=(256, 256, 5), num_classes=3):
    """
    Build CTANet-inspired CNN model with advanced attention mechanism
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes (growth stages)
        
    Returns:
        Compiled Keras model with advanced attention
    """
    inputs = layers.Input(shape=input_shape)
    
    # Multi-scale feature extraction (CTANet-inspired)
    # Branch 1: Fine features
    branch1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    
    # Branch 2: Medium features
    branch2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(branch2)
    
    # Branch 3: Coarse features
    branch3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(branch3)
    
    # Concatenate branches
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Deeper feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Advanced attention mechanism (Channel + Spatial)
    # Channel attention
    channel_attention = layers.GlobalAveragePooling2D()(x)
    channel_attention = layers.Dense(256 // 8, activation='relu')(channel_attention)
    channel_attention = layers.Dense(256, activation='sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, 256))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
    
    # Spatial attention
    spatial_attention = layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)
    x = layers.Multiply()([x, spatial_attention])
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_save_path='models/trained_cnn.h5'):
    """
    Get training callbacks for model optimization.
    
    Args:
        model_save_path: Path to save the best model
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def create_data_augmentation():
    """
    Create data augmentation pipeline for training.
    
    Returns:
        Keras Sequential model for data augmentation
    """
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


def evaluate_model(model, test_data, test_labels):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test input data
        test_labels: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    class_report = classification_report(true_classes, predicted_classes, 
                                       target_names=['Early', 'Mid', 'Late'],
                                       output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }


def print_model_summary(model):
    """Print model architecture summary."""
    print("Model Architecture:")
    print("="*50)
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN model creation...")
    
    # Test basic model
    model = build_cnn_model()
    print_model_summary(model)
    
    # Test attention model
    print("\n" + "="*50)
    print("Testing Attention CNN model...")
    attention_model = build_attention_cnn_model()
    print_model_summary(attention_model)
    
    # Test callbacks
    callbacks = get_callbacks()
    print(f"\nCreated {len(callbacks)} callbacks for training")
    
    # Test data augmentation
    aug_model = create_data_augmentation()
    print(f"\nCreated data augmentation pipeline with {len(aug_model.layers)} layers")
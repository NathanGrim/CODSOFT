"""
Training script for Image Captioning model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

import config
from feature_extractor import FeatureExtractor
from caption_model import create_model
from data_loader import DataLoader, create_tf_dataset, create_sample_dataset


class CaptionTrainer:
    """
    Trainer class for image captioning model
    """
    
    def __init__(self, use_sample_data=False):
        """
        Initialize trainer
        
        Args:
            use_sample_data: If True, create and use sample data
        """
        self.use_sample_data = use_sample_data
        self.feature_extractor = None
        self.model = None
        self.tokenizer = None
        self.max_length = 0
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Create directories
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
    
    def extract_features(self, image_names):
        """
        Extract features from all images
        
        Args:
            image_names: List of image filenames
            
        Returns:
            Dictionary mapping image names to features
        """
        print("\nExtracting image features...")
        
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(config.FEATURE_EXTRACTOR)
        
        features_dict = {}
        unique_images = list(set(image_names))
        
        for img_name in tqdm(unique_images, desc="Extracting features"):
            img_path = os.path.join(config.IMAGES_DIR, img_name)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            try:
                features = self.feature_extractor.extract_features(img_path)
                features_dict[img_name] = features
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        
        print(f"Extracted features for {len(features_dict)} images")
        return features_dict
    
    def prepare_data(self):
        """
        Load and prepare data for training
        
        Returns:
            train_dataset, val_dataset, vocab_size, feature_shape
        """
        # Create sample data if needed
        if self.use_sample_data:
            print("Creating sample dataset...")
            create_sample_dataset()
        
        # Load captions
        loader = DataLoader()
        captions_dict = loader.load_captions()
        
        if not captions_dict:
            raise ValueError(f"No captions found. Please ensure captions file exists at {config.CAPTIONS_FILE}")
        
        # Preprocess captions
        tokenizer, image_names, sequences, max_length = loader.preprocess_captions(captions_dict)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Save tokenizer
        tokenizer_path = os.path.join(config.MODEL_DIR, 'tokenizer.pkl')
        loader.save_tokenizer(tokenizer_path)
        
        # Extract features
        features_dict = self.extract_features(image_names)
        
        if not features_dict:
            raise ValueError("No features extracted. Please check image paths.")
        
        # Create dataset
        caption_sequences = list(zip(image_names, sequences))
        
        # Split data
        train_data, val_data = train_test_split(
            caption_sequences,
            test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
            random_state=42
        )
        
        print(f"\nTraining samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create TensorFlow datasets
        train_dataset = create_tf_dataset(
            features_dict, train_data, max_length, config.BATCH_SIZE
        )
        val_dataset = create_tf_dataset(
            features_dict, val_data, max_length, config.BATCH_SIZE
        )
        
        vocab_size = len(tokenizer.word_index) + 1
        feature_shape = self.feature_extractor.feature_shape
        
        return train_dataset, val_dataset, vocab_size, feature_shape
    
    def build_model(self, vocab_size, feature_shape):
        """
        Build the captioning model
        
        Args:
            vocab_size: Size of vocabulary
            feature_shape: Shape of image features
        """
        print("\nBuilding model...")
        self.model = create_model(vocab_size, feature_shape)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        print("Model built and compiled successfully")
    
    def train(self, train_dataset, val_dataset, epochs=None):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs to train
        """
        epochs = epochs or config.EPOCHS
        
        print(f"\nStarting training for {epochs} epochs...")
        
        # Callbacks
        checkpoint_path = os.path.join(
            config.CHECKPOINTS_DIR,
            'model_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5'
        )
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.CHECKPOINTS_DIR, 'best_model.weights.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        self.history = history.history
        history_path = os.path.join(config.LOGS_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {min(self.history['val_loss']):.4f}")
    
    def save_model(self):
        """Save the complete model"""
        model_path = os.path.join(config.MODEL_DIR, 'caption_model.weights.h5')
        self.model.save_weights(model_path)
        
        # Save configuration
        model_config = {
            'vocab_size': self.model.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'lstm_units': self.model.lstm_units,
            'feature_shape': self.model.feature_shape,
            'max_length': self.max_length,
            'feature_extractor': config.FEATURE_EXTRACTOR
        }
        
        config_path = os.path.join(config.MODEL_DIR, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"\nModel saved to {model_path}")
        print(f"Configuration saved to {config_path}")


def main():
    """Main training function"""
    print("=" * 60)
    print("Image Captioning Model - Training")
    print("=" * 60)
    
    # Check if sample data should be used
    use_sample = not os.path.exists(config.CAPTIONS_FILE)
    if use_sample:
        print("\nNo captions file found. Using sample data for demonstration.")
        print("To train on your own data, prepare:")
        print(f"  1. Images in: {config.IMAGES_DIR}")
        print(f"  2. Captions file: {config.CAPTIONS_FILE}")
        print("     Format: image_name.jpg,caption text here")
    
    # Initialize trainer
    trainer = CaptionTrainer(use_sample_data=use_sample)
    
    # Prepare data
    train_dataset, val_dataset, vocab_size, feature_shape = trainer.prepare_data()
    
    # Build model
    trainer.build_model(vocab_size, feature_shape)
    
    # Train
    trainer.train(train_dataset, val_dataset)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel files saved in: {config.MODEL_DIR}")
    print("\nYou can now use inference.py to generate captions for new images.")


if __name__ == '__main__':
    main()


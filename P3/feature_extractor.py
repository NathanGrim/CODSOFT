"""
Image Feature Extraction using pre-trained CNN models
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import numpy as np
from PIL import Image
import config


class FeatureExtractor:
    """
    Extract features from images using pre-trained CNN models
    """
    
    def __init__(self, model_name='resnet50'):
        """
        Initialize feature extractor with specified model
        
        Args:
            model_name: 'resnet50', 'vgg16', or 'inceptionv3'
        """
        self.model_name = model_name.lower()
        self.model = None
        self.preprocess_fn = None
        self.feature_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model without top layers"""
        print(f"Loading {self.model_name} model...")
        
        if self.model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_fn = resnet_preprocess
            self.feature_shape = 2048
            
        elif self.model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_fn = vgg_preprocess
            self.feature_shape = 512
            
        elif self.model_name == 'inceptionv3':
            base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_fn = inception_preprocess
            self.feature_shape = 2048
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Freeze the base model
        base_model.trainable = False
        self.model = base_model
        
        print(f"Model loaded. Feature shape: {self.feature_shape}")
    
    def preprocess_image(self, img_path):
        """
        Load and preprocess an image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(config.IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess_fn(img_array)
        
        return img_array
    
    def extract_features(self, img_path):
        """
        Extract features from a single image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Feature vector
        """
        img_array = self.preprocess_image(img_path)
        features = self.model.predict(img_array, verbose=0)
        
        return features.squeeze()
    
    def extract_features_batch(self, img_paths):
        """
        Extract features from multiple images
        
        Args:
            img_paths: List of image paths
            
        Returns:
            Array of feature vectors
        """
        images = []
        for img_path in img_paths:
            img_array = self.preprocess_image(img_path)
            images.append(img_array)
        
        images = np.vstack(images)
        features = self.model.predict(images, verbose=0)
        
        return features


def test_feature_extractor():
    """Test feature extraction"""
    import os
    
    # Create a sample image for testing
    os.makedirs('data/images', exist_ok=True)
    test_img = Image.new('RGB', (224, 224), color='red')
    test_img.save('data/images/test.jpg')
    
    # Test feature extraction
    extractor = FeatureExtractor('resnet50')
    features = extractor.extract_features('data/images/test.jpg')
    print(f"Extracted features shape: {features.shape}")
    print(f"Sample features (first 10): {features[:10]}")


if __name__ == '__main__':
    test_feature_extractor()


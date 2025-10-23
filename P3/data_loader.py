"""
Data loading and preprocessing utilities
"""

import os
import numpy as np
import pickle
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import config


class DataLoader:
    """
    Load and preprocess caption data
    """
    
    def __init__(self, captions_file=None, images_dir=None):
        """
        Initialize data loader
        
        Args:
            captions_file: Path to captions file
            images_dir: Path to images directory
        """
        self.captions_file = captions_file or config.CAPTIONS_FILE
        self.images_dir = images_dir or config.IMAGES_DIR
        self.tokenizer = None
        self.max_caption_length = 0
        
    def load_captions(self):
        """
        Load captions from file
        
        Expected format (one caption per line):
        image_name.jpg,caption text here
        or
        image_name.jpg caption text here
        
        Returns:
            Dictionary mapping image names to list of captions
        """
        captions_dict = {}
        
        if not os.path.exists(self.captions_file):
            print(f"Warning: Captions file not found at {self.captions_file}")
            return captions_dict
        
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Try different delimiters
                if ',' in line:
                    parts = line.split(',', 1)
                elif '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(' ', 1)
                
                if len(parts) == 2:
                    image_name, caption = parts
                    image_name = image_name.strip()
                    caption = caption.strip().lower()
                    
                    # Add start and end tokens
                    caption = f"{config.START_TOKEN} {caption} {config.END_TOKEN}"
                    
                    if image_name not in captions_dict:
                        captions_dict[image_name] = []
                    captions_dict[image_name].append(caption)
        
        print(f"Loaded captions for {len(captions_dict)} images")
        return captions_dict
    
    def create_tokenizer(self, captions, vocab_size=None):
        """
        Create tokenizer from captions
        
        Args:
            captions: List of caption strings
            vocab_size: Maximum vocabulary size
            
        Returns:
            Tokenizer object
        """
        vocab_size = vocab_size or config.VOCAB_SIZE
        
        tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token=config.UNK_TOKEN,
            filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        )
        tokenizer.fit_on_texts(captions)
        
        # Ensure special tokens are in the vocabulary
        for token in [config.START_TOKEN, config.END_TOKEN, config.PAD_TOKEN]:
            if token not in tokenizer.word_index:
                tokenizer.word_index[token] = len(tokenizer.word_index) + 1
        
        # Create reverse index
        tokenizer.index_word = {v: k for k, v in tokenizer.word_index.items()}
        
        self.tokenizer = tokenizer
        print(f"Vocabulary size: {len(tokenizer.word_index)}")
        
        return tokenizer
    
    def preprocess_captions(self, captions_dict):
        """
        Preprocess captions: create tokenizer and sequences
        
        Args:
            captions_dict: Dictionary of image -> captions
            
        Returns:
            tokenizer, image_names, caption_sequences, max_length
        """
        # Collect all captions
        all_captions = []
        image_names = []
        
        for img_name, captions in captions_dict.items():
            for caption in captions:
                all_captions.append(caption)
                image_names.append(img_name)
        
        # Create tokenizer
        tokenizer = self.create_tokenizer(all_captions)
        
        # Convert captions to sequences
        sequences = tokenizer.texts_to_sequences(all_captions)
        
        # Find max length
        max_length = max(len(seq) for seq in sequences)
        max_length = min(max_length, config.MAX_CAPTION_LENGTH)
        self.max_caption_length = max_length
        
        print(f"Maximum caption length: {max_length}")
        print(f"Total training samples: {len(sequences)}")
        
        return tokenizer, image_names, sequences, max_length
    
    def save_tokenizer(self, filepath):
        """Save tokenizer to file"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'max_length': self.max_caption_length
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    @staticmethod
    def load_tokenizer(filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")
        return data['tokenizer'], data['max_length']


def create_tf_dataset(image_features, caption_sequences, max_length, batch_size):
    """
    Create TensorFlow dataset for training
    
    Args:
        image_features: Dictionary mapping image names to feature vectors
        caption_sequences: List of (image_name, caption_sequence) tuples
        max_length: Maximum caption length
        batch_size: Batch size
        
    Returns:
        tf.data.Dataset
    """
    features_list = []
    input_seqs = []
    target_seqs = []
    
    for img_name, seq in caption_sequences:
        if img_name not in image_features:
            continue
        
        # Pad sequence
        padded_seq = pad_sequences([seq], maxlen=max_length, padding='post')[0]
        
        # Input: all tokens except last
        # Target: all tokens except first
        input_seq = padded_seq[:-1]
        target_seq = padded_seq[1:]
        
        features_list.append(image_features[img_name])
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)
    input_array = np.array(input_seqs, dtype=np.int32)
    target_array = np.array(target_seqs, dtype=np.int32)
    
    # Create dataset
    # Format: (inputs, targets) where inputs = (features, input_seq)
    dataset = tf.data.Dataset.from_tensor_slices((
        (features_array, input_array),
        target_array
    ))
    
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_sample_dataset():
    """
    Create a sample dataset for testing/demo purposes
    """
    from PIL import Image
    import random
    
    # Create directories
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    
    # Sample captions
    sample_data = [
        ("sample1.jpg", ["a red car on the street", "red vehicle parked outside"]),
        ("sample2.jpg", ["a dog playing in the park", "happy dog running on grass"]),
        ("sample3.jpg", ["a beautiful sunset over the ocean", "sun setting over water"]),
        ("sample4.jpg", ["a person riding a bicycle", "cyclist on the road"]),
        ("sample5.jpg", ["a cat sitting on a chair", "cat resting on furniture"]),
    ]
    
    # Create sample images
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    for i, (img_name, captions) in enumerate(sample_data):
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            # Create a colored image
            color = colors[i % len(colors)]
            img = Image.new('RGB', (224, 224), color=color)
            img.save(img_path)
    
    # Create captions file
    with open(config.CAPTIONS_FILE, 'w', encoding='utf-8') as f:
        for img_name, captions in sample_data:
            for caption in captions:
                f.write(f"{img_name},{caption}\n")
    
    print(f"Created sample dataset with {len(sample_data)} images")
    print(f"Images saved to: {config.IMAGES_DIR}")
    print(f"Captions saved to: {config.CAPTIONS_FILE}")


if __name__ == '__main__':
    # Create sample dataset
    os.makedirs('data', exist_ok=True)
    create_sample_dataset()
    
    # Test data loader
    loader = DataLoader()
    captions_dict = loader.load_captions()
    
    if captions_dict:
        tokenizer, image_names, sequences, max_length = loader.preprocess_captions(captions_dict)
        print(f"\nSample caption: {captions_dict[list(captions_dict.keys())[0]][0]}")
        print(f"Sample sequence: {sequences[0]}")


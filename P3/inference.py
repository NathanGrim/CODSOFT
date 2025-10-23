"""
Inference script for generating captions for new images
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

import config
from feature_extractor import FeatureExtractor
from caption_model import create_model
from data_loader import DataLoader


class CaptionGenerator:
    """
    Generate captions for images using trained model
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize caption generator
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.model_dir = model_dir or config.MODEL_DIR
        self.feature_extractor = None
        self.model = None
        self.tokenizer = None
        self.max_length = 0
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and tokenizer"""
        print("Loading model...")
        
        # Load model configuration
        config_path = os.path.join(self.model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model configuration not found at {config_path}")
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer, self.max_length = DataLoader.load_tokenizer(tokenizer_path)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(model_config['feature_extractor'])
        
        # Build model
        self.model = create_model(
            vocab_size=model_config['vocab_size'],
            feature_shape=model_config['feature_shape']
        )
        
        # Load weights
        model_path = os.path.join(self.model_dir, 'caption_model.weights.h5')
        if not os.path.exists(model_path):
            # Try loading from best checkpoint
            model_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model.weights.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Build the model with a dummy input first
        dummy_features = np.random.randn(1, model_config['feature_shape']).astype(np.float32)
        dummy_captions = np.zeros((1, self.max_length), dtype=np.int32)
        _ = self.model((dummy_features, dummy_captions), training=False)
        
        # Load weights
        self.model.load_weights(model_path)
        
        print("Model loaded successfully!")
        print(f"Vocabulary size: {model_config['vocab_size']}")
        print(f"Max caption length: {self.max_length}")
    
    def generate_caption(self, image_path, show_image=False):
        """
        Generate caption for a single image
        
        Args:
            image_path: Path to image file
            show_image: If True, display the image with caption
            
        Returns:
            Generated caption string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Extract features
        features = self.feature_extractor.extract_features(image_path)
        
        # Generate caption
        caption = self.model.generate_caption(
            features,
            self.tokenizer,
            max_length=self.max_length
        )
        
        # Display if requested
        if show_image:
            self.display_image_with_caption(image_path, caption)
        
        return caption
    
    def generate_captions_batch(self, image_paths):
        """
        Generate captions for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of captions
        """
        captions = []
        
        for img_path in image_paths:
            try:
                caption = self.generate_caption(img_path, show_image=False)
                captions.append(caption)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                captions.append("")
        
        return captions
    
    def display_image_with_caption(self, image_path, caption):
        """
        Display image with generated caption
        
        Args:
            image_path: Path to image
            caption: Generated caption
        """
        img = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(caption, fontsize=14, wrap=True, pad=20)
        plt.tight_layout()
        plt.show()
    
    def generate_with_beam_search(self, image_path, beam_width=3, show_image=False):
        """
        Generate caption using beam search (more accurate but slower)
        
        Args:
            image_path: Path to image file
            beam_width: Number of beams to keep
            show_image: If True, display the image with caption
            
        Returns:
            Generated caption string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Extract features
        features = self.feature_extractor.extract_features(image_path)
        features = tf.expand_dims(features, 0)
        features = self.model.feature_fc(features)
        
        # Initialize beams
        start_token_id = self.tokenizer.word_index.get(config.START_TOKEN, 1)
        end_token_id = self.tokenizer.word_index.get(config.END_TOKEN, 2)
        
        beams = [([start_token_id], 0.0)]  # (sequence, score)
        completed_beams = []
        
        for _ in range(self.max_length):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == end_token_id:
                    completed_beams.append((seq, score))
                    continue
                
                # Prepare input
                input_seq = tf.expand_dims(seq, 0)
                embeddings = self.model.embedding(input_seq)
                
                # Get predictions
                hidden = tf.zeros((1, self.model.lstm_units))
                context_vector, _ = self.model.attention(features[0], hidden[0])
                context_vector = tf.expand_dims(tf.expand_dims(context_vector, 0), 1)
                context_vector = tf.tile(context_vector, [1, tf.shape(embeddings)[1], 1])
                
                lstm_input = tf.concat([context_vector, embeddings], axis=-1)
                lstm_out, _, _ = self.model.lstm(lstm_input, initial_state=[hidden, hidden])
                output = self.model.fc1(lstm_out[:, -1, :])
                predictions = self.model.fc2(output)
                
                # Get top-k predictions
                probs = tf.nn.softmax(predictions[0]).numpy()
                top_k_ids = np.argsort(probs)[-beam_width:]
                
                for token_id in top_k_ids:
                    new_seq = seq + [token_id]
                    new_score = score - np.log(probs[token_id] + 1e-10)
                    candidates.append((new_seq, new_score))
            
            # Keep top beams
            candidates.sort(key=lambda x: x[1])
            beams = candidates[:beam_width]
            
            if len(completed_beams) >= beam_width:
                break
        
        # Get best completed beam
        if completed_beams:
            best_seq, _ = min(completed_beams, key=lambda x: x[1])
        else:
            best_seq, _ = min(beams, key=lambda x: x[1])
        
        # Convert to caption
        words = []
        for idx in best_seq:
            word = self.tokenizer.index_word.get(idx, config.UNK_TOKEN)
            if word not in [config.START_TOKEN, config.END_TOKEN, config.PAD_TOKEN]:
                words.append(word)
        
        caption = ' '.join(words)
        
        if show_image:
            self.display_image_with_caption(image_path, caption)
        
        return caption


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--images_dir', type=str, help='Directory containing images')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                       help='Directory containing model files')
    parser.add_argument('--beam_search', action='store_true',
                       help='Use beam search (slower but more accurate)')
    parser.add_argument('--show', action='store_true',
                       help='Display images with captions')
    parser.add_argument('--output', type=str, help='Output file for captions')
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = CaptionGenerator(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease ensure the model is trained first by running: python train.py")
        return
    
    results = []
    
    # Process single image
    if args.image:
        print(f"\nGenerating caption for: {args.image}")
        
        if args.beam_search:
            caption = generator.generate_with_beam_search(args.image, show_image=args.show)
        else:
            caption = generator.generate_caption(args.image, show_image=args.show)
        
        print(f"Caption: {caption}")
        results.append((args.image, caption))
    
    # Process directory of images
    elif args.images_dir:
        if not os.path.exists(args.images_dir):
            print(f"Error: Directory not found: {args.images_dir}")
            return
        
        image_files = [f for f in os.listdir(args.images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing {len(image_files)} images from {args.images_dir}...")
        
        for img_file in image_files:
            img_path = os.path.join(args.images_dir, img_file)
            try:
                if args.beam_search:
                    caption = generator.generate_with_beam_search(img_path, show_image=False)
                else:
                    caption = generator.generate_caption(img_path, show_image=False)
                
                print(f"{img_file}: {caption}")
                results.append((img_file, caption))
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    else:
        print("Please provide either --image or --images_dir")
        return
    
    # Save results if output file specified
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            for img_name, caption in results:
                f.write(f"{img_name}\t{caption}\n")
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()


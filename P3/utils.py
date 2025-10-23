"""
Utility functions for image captioning
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_training_history(history_file='logs/training_history.json', save_path='training_plot.png'):
    """
    Plot training history
    
    Args:
        history_file: Path to training history JSON file
        save_path: Path to save the plot
    """
    if not os.path.exists(history_file):
        print(f"History file not found: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Accuracy')
        axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to: {save_path}")
    plt.show()


def analyze_captions(captions_file):
    """
    Analyze caption statistics
    
    Args:
        captions_file: Path to captions file
    """
    if not os.path.exists(captions_file):
        print(f"Captions file not found: {captions_file}")
        return
    
    captions = []
    word_counts = Counter()
    caption_lengths = []
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',', 1) if ',' in line else line.split('\t', 1)
            if len(parts) == 2:
                caption = parts[1].strip().lower()
                captions.append(caption)
                
                words = caption.split()
                caption_lengths.append(len(words))
                word_counts.update(words)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Caption Statistics")
    print("=" * 60)
    print(f"Total captions: {len(captions)}")
    print(f"Unique words: {len(word_counts)}")
    print(f"Average caption length: {np.mean(caption_lengths):.2f} words")
    print(f"Min caption length: {min(caption_lengths)} words")
    print(f"Max caption length: {max(caption_lengths)} words")
    print(f"Median caption length: {np.median(caption_lengths):.2f} words")
    
    print("\nTop 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    # Plot caption length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(caption_lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Caption Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Caption Lengths')
    plt.grid(True, alpha=0.3)
    plt.savefig('caption_length_distribution.png', dpi=150, bbox_inches='tight')
    print("\nCaption length distribution saved to: caption_length_distribution.png")
    plt.show()


def visualize_attention(image, attention_weights, caption_words, save_path='attention_viz.png'):
    """
    Visualize attention weights on image
    
    Args:
        image: PIL Image or numpy array
        attention_weights: Attention weights for each word
        caption_words: List of caption words
        save_path: Path to save visualization
    """
    n_words = len(caption_words)
    cols = min(4, n_words)
    rows = (n_words + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if n_words == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, (word, weights) in enumerate(zip(caption_words, attention_weights)):
        if i >= len(axes):
            break
        
        axes[i].imshow(image)
        
        # Overlay attention (simplified visualization)
        # In practice, you'd need to reshape attention to match image dimensions
        axes[i].set_title(f'"{word}"', fontsize=12)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_words, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention visualization saved to: {save_path}")
    plt.show()


def create_caption_dataset_from_annotations(annotations_file, images_dir, output_file):
    """
    Convert common annotation formats to our caption format
    
    Args:
        annotations_file: Path to annotations (JSON or text file)
        images_dir: Directory containing images
        output_file: Output captions file
    """
    # Handle JSON format (COCO-style)
    if annotations_file.endswith('.json'):
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Example for COCO format
        if 'annotations' in data and 'images' in data:
            image_id_to_name = {img['id']: img['file_name'] for img in data['images']}
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for ann in data['annotations']:
                    image_id = ann['image_id']
                    caption = ann['caption']
                    image_name = image_id_to_name.get(image_id, '')
                    
                    if image_name:
                        f.write(f"{image_name},{caption}\n")
            
            print(f"Created captions file: {output_file}")
    
    else:
        print("Unsupported annotation format")


def evaluate_model(generator, test_images, reference_captions):
    """
    Evaluate model using BLEU score (simplified)
    
    Args:
        generator: CaptionGenerator instance
        test_images: List of test image paths
        reference_captions: List of reference captions for each image
    
    Returns:
        Dictionary with evaluation metrics
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    smoothing = SmoothingFunction()
    bleu_scores = []
    
    for img_path, references in zip(test_images, reference_captions):
        # Generate caption
        generated = generator.generate_caption(img_path, show_image=False)
        generated_tokens = generated.lower().split()
        
        # Prepare references
        reference_tokens = [ref.lower().split() for ref in references]
        
        # Calculate BLEU
        score = sentence_bleu(
            reference_tokens,
            generated_tokens,
            smoothing_function=smoothing.method1
        )
        bleu_scores.append(score)
    
    return {
        'bleu_avg': np.mean(bleu_scores),
        'bleu_std': np.std(bleu_scores),
        'bleu_scores': bleu_scores
    }


def export_model_to_tflite(model_dir='models', output_file='model.tflite'):
    """
    Export model to TensorFlow Lite format for mobile deployment
    
    Args:
        model_dir: Directory containing model files
        output_file: Output TFLite file path
    """
    import tensorflow as tf
    from caption_model import create_model
    import json
    
    # Load model config
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create model
    model = create_model(
        vocab_size=model_config['vocab_size'],
        feature_shape=model_config['feature_shape']
    )
    
    # Load weights
    model_path = os.path.join(model_dir, 'caption_model.h5')
    model.load_weights(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model exported to TensorFlow Lite: {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'plot_history':
            plot_training_history()
        
        elif command == 'analyze':
            captions_file = sys.argv[2] if len(sys.argv) > 2 else 'data/captions.txt'
            analyze_captions(captions_file)
        
        else:
            print("Unknown command. Available commands:")
            print("  plot_history - Plot training history")
            print("  analyze <captions_file> - Analyze caption statistics")
    
    else:
        print("Usage: python utils.py <command>")
        print("\nAvailable commands:")
        print("  plot_history - Plot training history")
        print("  analyze <captions_file> - Analyze caption statistics")


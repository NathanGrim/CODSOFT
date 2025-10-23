# Image Captioning AI

A deep learning-based image captioning system that combines computer vision and natural language processing to automatically generate descriptive captions for images.

## Features

- **Pre-trained CNN Models**: Uses ResNet50, VGG16, or InceptionV3 for image feature extraction
- **LSTM with Attention**: Implements attention mechanism for better caption generation
- **Flexible Architecture**: Easily configurable model parameters
- **Beam Search**: Supports both greedy and beam search decoding
- **Interactive Demo**: User-friendly demo application
- **Sample Data**: Includes sample dataset for quick testing

## Architecture

The system consists of two main components:

### 1. Image Encoder
- Pre-trained CNN (ResNet50/VGG16/InceptionV3) extracts visual features
- Features are passed through a fully connected layer
- Models are loaded with ImageNet weights and kept frozen

### 2. Caption Decoder
- Word embeddings for text representation
- LSTM with attention mechanism for sequential caption generation
- Attention helps the model focus on relevant image regions
- Dropout layers for regularization

## Project Structure

```
p3/
├── config.py              # Configuration settings
├── feature_extractor.py   # CNN-based feature extraction
├── caption_model.py       # LSTM captioning model with attention
├── data_loader.py         # Data loading and preprocessing
├── train.py              # Training script
├── inference.py          # Inference script for generating captions
├── demo.py               # Interactive demo application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Data directory
│   ├── images/         # Training images
│   └── captions.txt    # Image captions
└── models/             # Saved models and checkpoints
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (required for text preprocessing):
```python
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### Option 1: Run the Interactive Demo

The easiest way to get started is to run the demo, which will automatically train on sample data if no model exists:

```bash
python demo.py
```

This will:
- Check if a trained model exists
- If not, create sample data and train a model
- Launch an interactive menu for generating captions

### Option 2: Train on Your Own Data

1. **Prepare your dataset**:
   - Place images in `data/images/`
   - Create `data/captions.txt` with the format:
     ```
     image1.jpg,A description of the image
     image1.jpg,Another description of the same image
     image2.jpg,Description of second image
     ```

2. **Train the model**:
```bash
python train.py
```

3. **Generate captions**:
```bash
# Single image
python inference.py --image path/to/image.jpg --show

# Directory of images
python inference.py --images_dir path/to/images/ --output captions.txt

# Use beam search for better quality
python inference.py --image path/to/image.jpg --beam_search --show
```

## Usage Examples

### Training

```python
from train import CaptionTrainer

# Initialize trainer
trainer = CaptionTrainer(use_sample_data=False)

# Prepare data
train_dataset, val_dataset, vocab_size, feature_shape = trainer.prepare_data()

# Build and train model
trainer.build_model(vocab_size, feature_shape)
trainer.train(train_dataset, val_dataset, epochs=20)

# Save model
trainer.save_model()
```

### Inference

```python
from inference import CaptionGenerator

# Load trained model
generator = CaptionGenerator()

# Generate caption for single image
caption = generator.generate_caption('path/to/image.jpg', show_image=True)
print(f"Caption: {caption}")

# Use beam search for better quality
caption = generator.generate_with_beam_search('path/to/image.jpg', beam_width=5)
print(f"Caption: {caption}")

# Process multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
captions = generator.generate_captions_batch(image_paths)
```

### Feature Extraction

```python
from feature_extractor import FeatureExtractor

# Initialize with ResNet50
extractor = FeatureExtractor('resnet50')

# Extract features from single image
features = extractor.extract_features('path/to/image.jpg')
print(f"Feature shape: {features.shape}")  # (2048,)

# Extract features from multiple images
image_paths = ['image1.jpg', 'image2.jpg']
features_batch = extractor.extract_features_batch(image_paths)
print(f"Batch features shape: {features_batch.shape}")  # (2, 2048)
```

## Configuration

Edit `config.py` to customize the model:

```python
# Model architecture
EMBEDDING_DIM = 256        # Word embedding dimension
LSTM_UNITS = 512          # LSTM hidden units
ATTENTION_UNITS = 256     # Attention mechanism units
VOCAB_SIZE = 5000         # Maximum vocabulary size

# Feature extraction
FEATURE_EXTRACTOR = 'resnet50'  # Options: 'resnet50', 'vgg16', 'inceptionv3'
IMAGE_SIZE = (224, 224)

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
```

## Model Architecture Details

### Attention Mechanism

The model uses Bahdanau attention, which:
- Computes attention weights based on image features and decoder hidden state
- Creates a context vector that highlights relevant image regions
- Improves caption quality by allowing the model to "focus" on different parts of the image

### Training Process

1. **Feature Extraction**: Pre-trained CNN extracts fixed-length feature vectors from images
2. **Caption Preprocessing**: Text captions are tokenized and converted to sequences
3. **Model Training**: LSTM learns to generate captions word-by-word
4. **Validation**: Model performance is monitored on validation set
5. **Checkpointing**: Best model is saved based on validation loss

## Dataset Format

### Captions File Format

The `captions.txt` file should contain one caption per line:

```
image_name.jpg,caption text here
image_name.jpg,another caption for same image
another_image.jpg,description of this image
```

Alternatively, you can use tab or space separation:

```
image_name.jpg    caption text here
```

### Tips for Better Results

1. **Dataset Size**: Use at least 1000+ image-caption pairs for good results
2. **Caption Quality**: Clean, descriptive captions lead to better models
3. **Multiple Captions**: Multiple captions per image improve generalization
4. **Image Quality**: Use clear, high-quality images
5. **Training Time**: More epochs generally improve quality (with early stopping)

## Advanced Usage

### Custom Model Configuration

```python
from caption_model import ImageCaptioningModel

# Create custom model
model = ImageCaptioningModel(
    vocab_size=10000,
    embedding_dim=512,
    lstm_units=1024,
    feature_shape=2048,
    attention_units=512,
    dropout_rate=0.3
)
```

### Using Different CNN Backbones

```python
from feature_extractor import FeatureExtractor

# Use VGG16
extractor = FeatureExtractor('vgg16')

# Use InceptionV3
extractor = FeatureExtractor('inceptionv3')

# Use ResNet50 (default)
extractor = FeatureExtractor('resnet50')
```

### Batch Inference

```python
import os
from inference import CaptionGenerator

generator = CaptionGenerator()

# Process all images in a directory
images_dir = 'path/to/images'
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    caption = generator.generate_caption(img_path)
    print(f"{img_file}: {caption}")
```

## Performance Optimization

### For Training
- Use GPU for faster training (TensorFlow will automatically use GPU if available)
- Increase batch size if you have sufficient memory
- Use mixed precision training for faster computation

### For Inference
- Use greedy search for faster inference (default)
- Use beam search for better quality (slower)
- Batch process multiple images together

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
- Reduce `BATCH_SIZE` in `config.py`
- Use a smaller vocabulary size
- Reduce `LSTM_UNITS` or `EMBEDDING_DIM`

**2. Model Not Training**
- Check that images and captions are properly aligned
- Ensure captions file format is correct
- Verify images can be loaded (correct paths and formats)

**3. Poor Caption Quality**
- Train for more epochs
- Use a larger dataset
- Try beam search instead of greedy search
- Increase model capacity (more LSTM units)

**4. Model Files Not Found**
- Ensure you've run `train.py` before inference
- Check that `models/` directory contains the required files
- Verify file paths in configuration

## Technical Details

### Dependencies

- **TensorFlow**: Deep learning framework
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **NLTK**: Text preprocessing
- **Matplotlib**: Visualization
- **scikit-learn**: Data splitting

### Model Files

After training, the following files are created:

- `models/caption_model.h5`: Model weights
- `models/model_config.json`: Model configuration
- `models/tokenizer.pkl`: Tokenizer for text processing
- `models/checkpoints/best_model.h5`: Best model checkpoint
- `logs/training_history.json`: Training history

## Evaluation Metrics

To evaluate your model, you can implement standard metrics:

- **BLEU**: Measures n-gram overlap with reference captions
- **METEOR**: Considers synonyms and word order
- **CIDEr**: Consensus-based metric for image description
- **ROUGE**: Recall-oriented metric

## Future Improvements

Possible enhancements:
- Implement transformer-based decoder
- Add object detection integration
- Support for multiple languages
- Visual attention visualization
- Fine-tuning pre-trained models
- Implement evaluation metrics (BLEU, METEOR, CIDEr)

## References

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Bottom-Up and Top-Down Attention for Image Captioning](https://arxiv.org/abs/1707.07998)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Contact

For questions or suggestions, please create an issue in the repository.

---

**Note**: This is a educational implementation of image captioning. For production use, consider using more sophisticated architectures and larger datasets like COCO or Flickr30k.


"""
Configuration file for Image Captioning model
"""

# Model Configuration
EMBEDDING_DIM = 256
LSTM_UNITS = 512
ATTENTION_UNITS = 256
VOCAB_SIZE = 5000  # Will be updated during training
MAX_CAPTION_LENGTH = 40

# Feature extraction
IMAGE_SIZE = (224, 224)
FEATURE_EXTRACTOR = 'resnet50'  # Options: 'resnet50', 'vgg16', 'inceptionv3'

# Training Configuration
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# Paths
DATA_DIR = 'data'
IMAGES_DIR = 'data/images'
CAPTIONS_FILE = 'data/captions.txt'
MODEL_DIR = 'models'
CHECKPOINTS_DIR = 'models/checkpoints'
LOGS_DIR = 'logs'

# Tokenizer settings
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# Dataset split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


"""
Image Captioning Model with Attention Mechanism
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


class BahdanauAttention(layers.Layer):
    """
    Bahdanau Attention mechanism
    """
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, features, hidden):
        """
        Args:
            features: Image features (batch_size, feature_dim)
            hidden: Previous hidden state (batch_size, hidden_dim)
            
        Returns:
            context_vector: Weighted features
            attention_weights: Attention weights
        """
        # Expand dims for broadcasting
        # features shape: (batch_size, 1, feature_dim)
        # hidden shape: (batch_size, hidden_dim)
        
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # Score shape: (batch_size, 1, attention_units)
        score = tf.nn.tanh(self.W1(tf.expand_dims(features, 1)) + 
                          self.W2(hidden_with_time_axis))
        
        # Attention weights shape: (batch_size, 1, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Context vector shape: (batch_size, feature_dim)
        context_vector = attention_weights * tf.expand_dims(features, 1)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class ImageCaptioningModel(keras.Model):
    """
    Image Captioning Model with LSTM and Attention
    """
    
    def __init__(self, vocab_size, embedding_dim, lstm_units, 
                 feature_shape, attention_units=256, dropout_rate=0.5):
        """
        Initialize the captioning model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            feature_shape: Shape of image features
            attention_units: Units for attention mechanism
            dropout_rate: Dropout rate for regularization
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.feature_shape = feature_shape
        
        # Image feature processing
        self.feature_fc = layers.Dense(embedding_dim, activation='relu')
        self.feature_dropout = layers.Dropout(dropout_rate)
        
        # Word embedding
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.embedding_dropout = layers.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = BahdanauAttention(attention_units)
        
        # LSTM layers
        self.lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm_dropout = layers.Dropout(dropout_rate)
        
        # Output layer
        self.fc1 = layers.Dense(lstm_units, activation='relu')
        self.fc_dropout = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        """
        Forward pass
        
        Args:
            inputs: Tuple of (features, captions) where:
                features: Image features (batch_size, feature_shape)
                captions: Caption sequences (batch_size, max_length)
            training: Training mode flag
            
        Returns:
            predictions: Predicted word probabilities
        """
        # Unpack inputs
        if isinstance(inputs, tuple):
            features, captions = inputs
        else:
            # For backwards compatibility
            features = inputs
            captions = None
        # Process image features
        features = self.feature_fc(features)
        if training:
            features = self.feature_dropout(features, training=training)
        
        # Initialize LSTM hidden state with image features
        batch_size = tf.shape(features)[0]
        hidden = tf.zeros((batch_size, self.lstm_units))
        cell = tf.zeros((batch_size, self.lstm_units))
        
        # Embed captions
        embeddings = self.embedding(captions)
        if training:
            embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Apply attention
        context_vector, attention_weights = self.attention(features, hidden)
        context_vector = tf.expand_dims(context_vector, 1)
        context_vector = tf.tile(context_vector, [1, tf.shape(embeddings)[1], 1])
        
        # Concatenate context with embeddings
        lstm_input = tf.concat([context_vector, embeddings], axis=-1)
        
        # LSTM
        lstm_out, hidden, cell = self.lstm(lstm_input, initial_state=[hidden, cell])
        if training:
            lstm_out = self.lstm_dropout(lstm_out, training=training)
        
        # Output layers
        output = self.fc1(lstm_out)
        if training:
            output = self.fc_dropout(output, training=training)
        predictions = self.fc2(output)
        
        return predictions
    
    def generate_caption(self, features, tokenizer, max_length=40):
        """
        Generate caption for an image using greedy search
        
        Args:
            features: Image features (feature_shape,)
            tokenizer: Tokenizer object
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        # Add batch dimension to features
        features_batch = tf.expand_dims(features, 0)  # (1, feature_shape)
        
        # Initialize with start token
        caption = [tokenizer.word_index.get(config.START_TOKEN, 1)]
        
        for _ in range(max_length):
            # Prepare caption sequence
            caption_seq = tf.constant([caption], dtype=tf.int32)  # (1, seq_len)
            
            # Get predictions using the model's forward pass
            # The model will process features internally
            predictions = self((features_batch, caption_seq), training=False)  # (1, seq_len, vocab_size)
            
            # Get the last word prediction
            last_pred = predictions[0, -1, :]  # (vocab_size,)
            predicted_id = tf.argmax(last_pred).numpy()
            
            # Check for end token
            if predicted_id == tokenizer.word_index.get(config.END_TOKEN, 2):
                break
            
            caption.append(int(predicted_id))
        
        # Convert to words
        words = []
        for idx in caption:
            word = tokenizer.index_word.get(idx, config.UNK_TOKEN)
            if word not in [config.START_TOKEN, config.END_TOKEN, config.PAD_TOKEN]:
                words.append(word)
        
        return ' '.join(words) if words else "unable to generate caption"


def create_model(vocab_size, feature_shape):
    """
    Create and return a captioning model
    
    Args:
        vocab_size: Size of vocabulary
        feature_shape: Shape of image features
        
    Returns:
        ImageCaptioningModel instance
    """
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        feature_shape=feature_shape,
        attention_units=config.ATTENTION_UNITS,
        dropout_rate=config.DROPOUT_RATE
    )
    
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model(vocab_size=5000, feature_shape=2048)
    print("Model created successfully")
    print(f"Model architecture:")
    
    # Test with dummy data
    import numpy as np
    dummy_features = np.random.randn(2, 2048).astype(np.float32)
    dummy_captions = np.random.randint(0, 5000, (2, 20))
    
    output = model((dummy_features, dummy_captions), training=False)
    print(f"Output shape: {output.shape}")


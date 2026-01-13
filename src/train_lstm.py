"""
Train LSTM model for sequence-based anomaly detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import argparse


class LSTMTrainer:
    """
    Trainer for LSTM-based anomaly detection model.
    """
    
    def __init__(self, 
                 sequence_length,
                 n_features,
                 lstm_units=[64, 32],
                 learning_rate=0.001):
        """
        Initialize the LSTM trainer.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            lstm_units: List of LSTM layer units
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture."""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(0.2))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(0.2))
        
        # Output layer
        self.model.add(Dense(self.n_features))
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def create_sequences(self, data, sequence_length):
        """
        Create sequences from time series data.
        
        Args:
            data: Input time series data
            sequence_length: Length of each sequence
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length])
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def compute_prediction_error(self, X, y):
        """
        Compute prediction error for anomaly detection.
        
        Args:
            X: Input sequences
            y: True targets
            
        Returns:
            Prediction errors
        """
        predictions = self.model.predict(X)
        mse = np.mean(np.power(y - predictions, 2), axis=1)
        return mse
    
    def save(self, path: str):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='../models/lstm_model.h5',
                        help='Path to save the model')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    print("Training LSTM model...")
    # Training implementation here

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
    parser.add_argument('--data', type=str, default='data/X_phase2.npy',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models/lstm_autoencoder.keras',
                        help='Path to save the model')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.endswith('.npy'):
        X = np.load(args.data, allow_pickle=True)
    elif args.data.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(args.data)
        X = df.select_dtypes(include=[np.number]).values
    else:
        raise ValueError("Data file must be .npy or .csv")
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Create trainer
    trainer = LSTMTrainer(
        sequence_length=args.sequence_length,
        n_features=X.shape[1]
    )
    
    # Create sequences
    print(f"\nCreating sequences with length {args.sequence_length}...")
    X_seq, y_seq = trainer.create_sequences(X, args.sequence_length)
    print(f"Created {len(X_seq)} sequences")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, 
        test_size=args.val_split, 
        random_state=42
    )
    
    print(f"Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
    
    # Train model
    print(f"\nTraining LSTM model...")
    print(f"  - sequence_length: {args.sequence_length}")
    print(f"  - n_features: {X.shape[1]}")
    print(f"  - epochs: {args.epochs}")
    print(f"  - batch_size: {args.batch_size}")
    
    history = trainer.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    
    # Compute prediction errors on validation set
    print("\nComputing prediction errors on validation set...")
    errors = trainer.compute_prediction_error(X_val[:10], y_val[:10])
    
    print("\nSample Prediction Errors (first 10):")
    for i, error in enumerate(errors):
        print(f"  Sequence {i}: MSE={error:.6f}")
    
    print(f"\nMean prediction error: {np.mean(errors):.6f}")
    print(f"Std prediction error: {np.std(errors):.6f}")
    print(f"\nTraining complete! Model saved to {args.output}")

"""
Train Autoencoder model for anomaly detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
from pathlib import Path


class AutoencoderTrainer:
    """
    Trainer for Autoencoder anomaly detection model.
    """
    
    def __init__(self, input_dim, encoding_dim=32, learning_rate=0.001):
        """
        Initialize the Autoencoder trainer.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the encoding layer
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.encoder = None
        self.decoder = None
        self._build_model()
    
    def _build_model(self):
        """Build the autoencoder architecture."""
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        self.model = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32):
        """
        Train the autoencoder.
        
        Args:
            X_train: Training data
            X_val: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def compute_reconstruction_error(self, X):
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            X: Input data
            
        Returns:
            Reconstruction errors
        """
        reconstructions = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
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
    parser = argparse.ArgumentParser(description='Train Autoencoder model')
    parser.add_argument('--data', type=str, default='data/X_phase2.npy',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models/autoencoder.keras',
                        help='Path to save the model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--encoding-dim', type=int, default=32, help='Encoding dimension')
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
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(X, test_size=args.val_split, random_state=42)
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create and train model
    print(f"\nTraining Autoencoder model...")
    print(f"  - encoding_dim: {args.encoding_dim}")
    print(f"  - epochs: {args.epochs}")
    print(f"  - batch_size: {args.batch_size}")
    
    trainer = AutoencoderTrainer(
        input_dim=X.shape[1],
        encoding_dim=args.encoding_dim
    )
    
    history = trainer.train(
        X_train, 
        X_val=X_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    
    # Compute reconstruction errors on validation set
    print("\nComputing reconstruction errors on validation set...")
    errors = trainer.compute_reconstruction_error(X_val[:10])
    
    print("\nSample Reconstruction Errors (first 10):")
    for i, error in enumerate(errors):
        print(f"  Sample {i}: MSE={error:.6f}")
    
    print(f"\nMean reconstruction error: {np.mean(errors):.6f}")
    print(f"Std reconstruction error: {np.std(errors):.6f}")
    print(f"\nTraining complete! Model saved to {args.output}")

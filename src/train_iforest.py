"""
Train Isolation Forest model for anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import argparse


class IsolationForestTrainer:
    """
    Trainer for Isolation Forest anomaly detection model.
    """
    
    def __init__(self, 
                 n_estimators=100,
                 contamination=0.1,
                 max_samples='auto',
                 random_state=42):
        """
        Initialize the Isolation Forest trainer.
        
        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of anomalies
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        
    def train(self, X_train: np.ndarray) -> 'IsolationForestTrainer':
        """
        Train the Isolation Forest model.
        
        Args:
            X_train: Training features
            
        Returns:
            Self
        """
        self.model.fit(X_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Input features
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (lower is more anomalous)
        """
        return self.model.score_samples(X)
    
    def save(self, path: str):
        """Save the trained model."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Isolation Forest model')
    parser.add_argument('--data', type=str, default='data/X_phase2.npy',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models/isolation_forest.pkl', 
                        help='Path to save the model')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Expected proportion of anomalies')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in the forest')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.endswith('.npy'):
        X_train = np.load(args.data, allow_pickle=True)
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
        # Assume all numeric columns are features
        X_train = df.select_dtypes(include=[np.number]).values
    else:
        raise ValueError("Data file must be .npy or .csv")
    
    print(f"Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Create and train model
    print(f"\nTraining Isolation Forest model...")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - contamination: {args.contamination}")
    
    trainer = IsolationForestTrainer(
        n_estimators=args.n_estimators,
        contamination=args.contamination
    )
    
    trainer.train(X_train)
    
    # Save model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    
    # Compute and display sample predictions
    print("\nComputing sample predictions...")
    scores = trainer.score_samples(X_train[:10])
    predictions = trainer.predict(X_train[:10])
    
    print("\nSample Anomaly Scores (first 10):")
    for i, (score, pred) in enumerate(zip(scores, predictions)):
        label = "Anomaly" if pred == -1 else "Normal"
        print(f"  Sample {i}: score={score:.4f}, prediction={label}")
    
    print(f"\nTraining complete! Model saved to {args.output}")

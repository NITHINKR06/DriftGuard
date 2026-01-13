"""
Inference module for anomaly detection models.
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from pathlib import Path
from typing import Dict, Union, List


class AnomalyDetector:
    """
    Unified interface for anomaly detection inference.
    """
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.models = {}
        self.feature_engineer = None
    
    def load_model(self, model_name: str, model_path: str, model_type: str = 'sklearn'):
        """
        Load a trained model.
        
        Args:
            model_name: Name to identify the model
            model_path: Path to the saved model
            model_type: Type of model ('sklearn' or 'keras')
        """
        if model_type == 'sklearn':
            model = joblib.load(model_path)
        elif model_type == 'keras':
            model = keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.models[model_name] = {
            'model': model,
            'type': model_type
        }
        print(f"Loaded {model_name} from {model_path}")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using loaded models.
        
        Args:
            X: Input features
            model_name: Specific model to use (if None, uses all models)
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        
        models_to_use = [model_name] if model_name else self.models.keys()
        
        for name in models_to_use:
            if name not in self.models:
                print(f"Warning: Model {name} not loaded")
                continue
            
            model_info = self.models[name]
            model = model_info['model']
            
            if model_info['type'] == 'sklearn':
                if hasattr(model, 'predict'):
                    predictions[name] = model.predict(X)
                if hasattr(model, 'score_samples'):
                    predictions[f'{name}_scores'] = model.score_samples(X)
            elif model_info['type'] == 'keras':
                predictions[name] = model.predict(X)
        
        return predictions
    
    def detect_anomalies(self, 
                        X: np.ndarray,
                        threshold: float = None,
                        percentile: float = 95) -> np.ndarray:
        """
        Detect anomalies using all loaded models.
        
        Args:
            X: Input features
            threshold: Threshold for anomaly detection (if None, uses percentile)
            percentile: Percentile to use for threshold (default: 95)
            
        Returns:
            Binary anomaly labels (1 for anomaly, 0 for normal)
        """
        predictions = self.predict(X)
        
        # Combine predictions from all models
        # This is a simple implementation - can be made more sophisticated
        all_scores = []
        for key, values in predictions.items():
            if 'score' in key.lower():
                all_scores.append(values)
        
        if not all_scores:
            raise ValueError("No anomaly scores available from models")
        
        # Average scores
        avg_scores = np.mean(all_scores, axis=0)
        
        # Determine threshold
        if threshold is None:
            threshold = np.percentile(avg_scores, percentile)
        
        # Anomalies have lower scores in isolation forest
        anomalies = (avg_scores < threshold).astype(int)
        
        return anomalies
    
    def batch_predict(self, 
                     data_path: str,
                     output_path: str = None,
                     batch_size: int = 1000) -> pd.DataFrame:
        """
        Perform batch predictions on a dataset.
        
        Args:
            data_path: Path to input data
            output_path: Path to save results (optional)
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Process in batches
        results = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            # Assuming features are all numeric columns
            X = batch.select_dtypes(include=[np.number]).values
            
            predictions = self.predict(X)
            batch_results = batch.copy()
            
            for model_name, preds in predictions.items():
                batch_results[f'pred_{model_name}'] = preds
            
            results.append(batch_results)
        
        final_results = pd.concat(results, ignore_index=True)
        
        if output_path:
            final_results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return final_results


if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector()
    
    # Load models
    # detector.load_model('iforest', '../models/iforest_model.pkl', 'sklearn')
    # detector.load_model('autoencoder', '../models/autoencoder_model.h5', 'keras')
    
    # Make predictions
    # predictions = detector.predict(X_test)

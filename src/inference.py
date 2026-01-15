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
    import sys
    import argparse
    
    # Add parent directory to path for imports
    sys.path.append('..')
    from src import data_loader, risk_scoring, features
    
    parser = argparse.ArgumentParser(description='Run anomaly detection inference')
    parser.add_argument('--data', type=str, default='data/X_phase2.npy',
                        help='Path to input data')
    parser.add_argument('--output', type=str, default='data/inference_results.csv',
                        help='Path to save results')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ANOMALY DETECTION INFERENCE PIPELINE")
    print("="*70)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Load all models
    print("\n[1/5] Loading trained models...")
    try:
        detector.load_model('iforest', f'{args.models_dir}/isolation_forest.pkl', 'sklearn')
        detector.load_model('autoencoder', f'{args.models_dir}/autoencoder.keras', 'keras')
        detector.load_model('lstm', f'{args.models_dir}/lstm_autoencoder.keras', 'keras')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure models are trained and saved in the models directory.")
        sys.exit(1)
    
    # Load data
    print(f"\n[2/5] Loading input data from {args.data}...")
    try:
        if args.data.endswith('.npy'):
            X = np.load(args.data, allow_pickle=True)
        elif args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
            X = df.select_dtypes(include=[np.number]).values
        else:
            raise ValueError("Data must be .npy or .csv")
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run predictions
    print("\n[3/5] Running model predictions...")
    predictions = detector.predict(X)
    
    # Extract scores for risk computation
    print("\n[4/5] Computing risk scores...")
    model_scores = {}
    
    # Get Isolation Forest scores
    if 'iforest_scores' in predictions:
        model_scores['iforest'] = predictions['iforest_scores']
        print(f"  - Isolation Forest: {len(model_scores['iforest'])} scores")
    
    # Get Autoencoder reconstruction errors
    if 'autoencoder' in predictions:
        autoencoder_model = detector.models['autoencoder']['model']
        reconstructions = predictions['autoencoder']
        errors = np.mean(np.power(X - reconstructions, 2), axis=1)
        model_scores['autoencoder'] = errors
        print(f"  - Autoencoder: {len(errors)} reconstruction errors")
    
    # Get LSTM prediction errors (simplified - using first n_features)
    if 'lstm' in predictions:
        lstm_pred = predictions['lstm']
        # For full LSTM evaluation, you'd need sequences, but this is simplified
        errors = np.mean(np.power(lstm_pred[:, :X.shape[1]] - X[:len(lstm_pred)], 2), axis=1)
        model_scores['lstm'] = errors[:len(X)]
        print(f"  - LSTM: {len(errors)} prediction errors")
    
    # Use risk scoring module
    from src.risk_scoring import RiskScorer
    
    scorer = RiskScorer()
    scorer.fit(model_scores)  # Fit on current data (in production, load pre-fitted)
    
    risk_scores, severity = scorer.score_and_classify(model_scores)
    
    # Create results dataframe
    print("\n[5/5] Generating results...")
    results_df = pd.DataFrame({
        'sample_id': range(len(risk_scores)),
        'risk_score': risk_scores,
        'severity': severity
    })
    
    # Add model scores
    for model_name, scores in model_scores.items():
        results_df[f'{model_name}_score'] = scores
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    scorer.print_summary(risk_scores, severity)
    
    # Show top risks
    print("\nTop 10 High-Risk Alerts:")
    print("-" * 70)
    top_alerts = results_df.nlargest(10, 'risk_score')
    for idx, row in top_alerts.iterrows():
        print(f"Sample {row['sample_id']:5d} | Risk: {row['risk_score']:.4f} | Severity: {row['severity']:6s}")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)

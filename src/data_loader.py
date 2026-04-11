"""
Data Loader Module
Central place to load datasets, models, and saved artifacts.
Avoids repeating np.load, pd.read_csv everywhere.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow import keras
from typing import Tuple, Optional, Dict, Any


# ==================== Raw Data Loading ====================

def load_raw_cicids_data(file_name: str = None) -> pd.DataFrame:
    """
    Load raw CICIDS2017 dataset.
    
    Args:
        file_name: Specific CSV file to load. If None, loads processed_phase1.csv
        
    Returns:
        DataFrame with raw network traffic data
    """
    if file_name is None:
        file_path = "data/processed_phase1.csv"
    else:
        file_path = f"data/CICIDS2017/{file_name}"
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {file_path}")
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load preprocessed Phase 1 data.
    
    Returns:
        DataFrame with processed features
    """
    df = pd.read_csv("data/processed_phase1.csv")
    print(f"Loaded {len(df)} processed records")
    return df


# ==================== Feature Data Loading ====================

def load_phase2_features() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Phase 2 processed features and labels.
    
    Returns:
        Tuple of (X_phase2, y_phase2)
    """
    X = np.load("data/X_phase2.npy", allow_pickle=True)
    y = np.load("data/y_phase2.npy", allow_pickle=True)
    print(f"Loaded Phase 2 data: X shape={X.shape}, y shape={y.shape}")
    return X, y


# ==================== Model Outputs Loading ====================

def load_iforest_scores() -> np.ndarray:
    """
    Load Isolation Forest anomaly scores.
    
    Returns:
        Array of anomaly scores from Isolation Forest
    """
    scores = np.load("data/iforest_scores.npy", allow_pickle=True)
    print(f"Loaded {len(scores)} Isolation Forest scores")
    return scores


def load_autoencoder_errors() -> np.ndarray:
    """
    Load Autoencoder reconstruction errors.
    
    Returns:
        Array of reconstruction errors from Autoencoder
    """
    errors = np.load("data/autoencoder_errors.npy", allow_pickle=True)
    print(f"Loaded {len(errors)} Autoencoder errors")
    return errors


def load_lstm_errors() -> np.ndarray:
    """
    Load LSTM sequence prediction errors.
    
    Returns:
        Array of prediction errors from LSTM
    """
    errors = np.load("data/lstm_sequence_errors.npy", allow_pickle=True)
    print(f"Loaded {len(errors)} LSTM sequence errors")
    return errors


def load_all_model_scores() -> Dict[str, np.ndarray]:
    """
    Load all model scores/errors in one call.
    
    Returns:
        Dictionary with keys: 'iforest', 'autoencoder', 'lstm'
    """
    return {
        'iforest': load_iforest_scores(),
        'autoencoder': load_autoencoder_errors(),
        'lstm': load_lstm_errors()
    }


# ==================== Final Outputs Loading ====================

def load_final_outputs() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load final risk scores and ground truth labels.
    Handles array length mismatch by truncating to the shorter length.
    
    Returns:
        Tuple of (risk_scores, labels)
    """
    risk_scores = np.load("data/final_risk_scores.npy", allow_pickle=True)
    labels = np.load("data/y_phase2.npy", allow_pickle=True)
    
    # Handle length mismatch by truncating to minimum length
    min_len = min(len(risk_scores), len(labels))
    if len(risk_scores) != len(labels):
        print(f"Warning: Array length mismatch detected!")
        print(f"  risk_scores: {len(risk_scores)} samples")
        print(f"  labels: {len(labels)} samples")
        print(f"  Truncating both to {min_len} samples")
        risk_scores = risk_scores[:min_len]
        labels = labels[:min_len]
    
    print(f"Loaded {len(risk_scores)} final risk scores and labels")
    return risk_scores, labels


def load_severity_labels() -> np.ndarray:
    """
    Load severity classification labels (HIGH/MEDIUM/LOW).
    
    Returns:
        Array of severity labels
    """
    severity = np.load("data/final_severity_labels.npy", allow_pickle=True)
    print(f"Loaded {len(severity)} severity labels")
    return severity


# ==================== Trained Models Loading ====================

def load_iforest_model(model_path: str = "models/isolation_forest.pkl"):
    """
    Load trained Isolation Forest model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Isolation Forest model
    """
    model = joblib.load(model_path)
    print(f"Loaded Isolation Forest model from {model_path}")
    return model


def load_autoencoder_model(model_path: str = "models/autoencoder.keras"):
    """
    Load trained Autoencoder model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras Autoencoder model
    """
    model = keras.models.load_model(model_path)
    print(f"Loaded Autoencoder model from {model_path}")
    return model


def load_lstm_model(model_path: str = "models/lstm_autoencoder.keras"):
    """
    Load trained LSTM model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras LSTM model
    """
    model = keras.models.load_model(model_path)
    print(f"Loaded LSTM model from {model_path}")
    return model


def load_all_models(models_dir: str = "models") -> Dict[str, Any]:
    """
    Load all trained models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary with keys: 'iforest', 'autoencoder', 'lstm'
    """
    return {
        'iforest': load_iforest_model(f"{models_dir}/isolation_forest.pkl"),
        'autoencoder': load_autoencoder_model(f"{models_dir}/autoencoder.keras"),
        'lstm': load_lstm_model(f"{models_dir}/lstm_autoencoder.keras")
    }


# ==================== SHAP Explanations Loading ====================

def load_shap_values(file_path: str = "data/shap_values_high_risk.npy") -> np.ndarray:
    """
    Load SHAP values for model explainability.
    
    Args:
        file_path: Path to SHAP values file
        
    Returns:
        Array of SHAP values
    """
    shap_values = np.load(file_path, allow_pickle=True)
    print(f"Loaded SHAP values from {file_path}")
    return shap_values


# ==================== Utility Functions ====================

def check_data_availability() -> Dict[str, bool]:
    """
    Check which data files are available.
    
    Returns:
        Dictionary showing availability of each data file
    """
    files_to_check = {
        'processed_data': 'data/processed_phase1.csv',
        'X_phase2': 'data/X_phase2.npy',
        'y_phase2': 'data/y_phase2.npy',
        'iforest_scores': 'data/iforest_scores.npy',
        'autoencoder_errors': 'data/autoencoder_errors.npy',
        'lstm_errors': 'data/lstm_sequence_errors.npy',
        'final_risk_scores': 'data/final_risk_scores.npy',
        'severity_labels': 'data/final_severity_labels.npy',
        'iforest_model': 'models/isolation_forest.pkl',
        'autoencoder_model': 'models/autoencoder.keras',
        'lstm_model': 'models/lstm_autoencoder.keras'
    }
    
    availability = {}
    for name, path in files_to_check.items():
        availability[name] = Path(path).exists()
    
    return availability


def get_data_summary() -> None:
    """Print summary of available data and models."""
    print("\n" + "="*50)
    print("DATA AVAILABILITY SUMMARY")
    print("="*50)
    
    availability = check_data_availability()
    for name, is_available in availability.items():
        status = "✓" if is_available else "✗"
        print(f"{status} {name}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test data loading
    get_data_summary()
    
    # Example: Load final outputs
    try:
        risk_scores, labels = load_final_outputs()
        print(f"\nSuccessfully loaded {len(risk_scores)} risk scores")
    except Exception as e:
        print(f"Error loading data: {e}")

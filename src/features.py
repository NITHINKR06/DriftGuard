"""
Feature Engineering Module
Feature preprocessing logic shared across training & inference.
Prevents data leakage by ensuring identical transformations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
from typing import Tuple, Optional, List
from pathlib import Path


class FeaturePipeline:
    """
    Feature transformation pipeline for anomaly detection.
    Ensures consistent preprocessing between training and inference.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize feature pipeline.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
        self.feature_names = None
        self.variance_selector = None
        self.is_fitted = False
    
    def _create_scaler(self):
        """Create the appropriate scaler based on scaler_type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeaturePipeline':
        """
        Fit the feature pipeline on training data.
        
        Args:
            X: Training features
            feature_names: Optional list of feature names
            
        Returns:
            Self
        """
        self.feature_names = feature_names
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Fit variance selector (remove near-zero variance features)
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self.variance_selector.fit(X)
        
        self.is_fitted = True
        print(f"Feature pipeline fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Apply scaling
        X_scaled = self.scaler.transform(X)
        
        # Apply variance filtering
        X_filtered = self.variance_selector.transform(X_scaled)
        
        return X_filtered
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Training features
            feature_names: Optional list of feature names
            
        Returns:
            Transformed features
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def save(self, path: str):
        """Save the fitted pipeline."""
        joblib.dump({
            'scaler': self.scaler,
            'variance_selector': self.variance_selector,
            'feature_names': self.feature_names,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Feature pipeline saved to {path}")
    
    def load(self, path: str):
        """Load a fitted pipeline."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.variance_selector = data['variance_selector']
        self.feature_names = data['feature_names']
        self.scaler_type = data['scaler_type']
        self.is_fitted = data['is_fitted']
        print(f"Feature pipeline loaded from {path}")


# ==================== Standalone Functions ====================

def scale_features(df: pd.DataFrame, scaler_type: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    Scale features using specified scaler.
    
    Args:
        df: DataFrame or array with features
        scaler_type: Type of scaler to use
        
    Returns:
        Tuple of (scaled_features, fitted_scaler)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler


def apply_scaling(X: np.ndarray, scaler: object) -> np.ndarray:
    """
    Apply pre-fitted scaler to features.
    
    Args:
        X: Features to scale
        scaler: Fitted scaler object
        
    Returns:
        Scaled features
    """
    return scaler.transform(X)


def remove_low_variance_features(X: np.ndarray, threshold: float = 0.01) -> Tuple[np.ndarray, object]:
    """
    Remove features with low variance.
    
    Args:
        X: Feature array
        threshold: Variance threshold
        
    Returns:
        Tuple of (filtered_features, selector)
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    print(f"Removed {X.shape[1] - X_filtered.shape[1]} low variance features")
    return X_filtered, selector


def prepare_features_from_dataframe(df: pd.DataFrame, 
                                   label_column: str = ' Label',
                                   drop_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and labels from a DataFrame.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        drop_columns: Additional columns to drop
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Drop label and any specified columns
    columns_to_drop = [label_column]
    if drop_columns:
        columns_to_drop.extend(drop_columns)
    
    # Remove columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Extract features
    X = df.drop(columns=columns_to_drop)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy
    X = X.values
    
    # Extract labels if available
    if label_column in df.columns:
        y = df[label_column].values
    else:
        y = None
    
    return X, y, feature_names


def handle_missing_values(X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Handle missing values in features.
    
    Args:
        X: Feature array
        strategy: Imputation strategy ('mean', 'median', 'zero')
        
    Returns:
        Array with missing values handled
    """
    X_copy = X.copy()
    
    if strategy == 'mean':
        col_mean = np.nanmean(X_copy, axis=0)
        inds = np.where(np.isnan(X_copy))
        X_copy[inds] = np.take(col_mean, inds[1])
    elif strategy == 'median':
        col_median = np.nanmedian(X_copy, axis=0)
        inds = np.where(np.isnan(X_copy))
        X_copy[inds] = np.take(col_median, inds[1])
    elif strategy == 'zero':
        X_copy = np.nan_to_num(X_copy, nan=0.0)
    
    return X_copy


def handle_infinite_values(X: np.ndarray) -> np.ndarray:
    """
    Replace infinite values with large finite values.
    
    Args:
        X: Feature array
        
    Returns:
        Array with infinite values handled
    """
    X_copy = X.copy()
    X_copy = np.nan_to_num(X_copy, posinf=1e10, neginf=-1e10)
    return X_copy


def preprocess_features(X: np.ndarray, 
                       handle_missing: bool = True,
                       handle_infinite: bool = True,
                       scale: bool = True,
                       scaler=None) -> Tuple[np.ndarray, Optional[object]]:
    """
    Complete preprocessing pipeline for features.
    
    Args:
        X: Raw features
        handle_missing: Whether to handle missing values
        handle_infinite: Whether to handle infinite values
        scale: Whether to scale features
        scaler: Pre-fitted scaler (if None, creates and fits new one)
        
    Returns:
        Tuple of (preprocessed_features, scaler)
    """
    X_processed = X.copy()
    
    # Handle missing values
    if handle_missing:
        X_processed = handle_missing_values(X_processed)
    
    # Handle infinite values
    if handle_infinite:
        X_processed = handle_infinite_values(X_processed)
    
    # Scale features
    fitted_scaler = None
    if scale:
        if scaler is None:
            # Fit new scaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            fitted_scaler = scaler
        else:
            # Use existing scaler
            X_processed = scaler.transform(X_processed)
            fitted_scaler = scaler
    
    return X_processed, fitted_scaler


def create_sequence_data(X: np.ndarray, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM model.
    
    Args:
        X: Feature array
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (X_sequences, y_targets)
    """
    sequences = []
    targets = []
    
    for i in range(len(X) - sequence_length):
        sequences.append(X[i:i+sequence_length])
        targets.append(X[i+sequence_length])
    
    return np.array(sequences), np.array(targets)


# ==================== Save/Load Utilities ====================

def save_scaler(scaler: object, path: str):
    """Save a fitted scaler."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path: str) -> object:
    """Load a fitted scaler."""
    scaler = joblib.load(path)
    print(f"Scaler loaded from {path}")
    return scaler


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("="*50)
    
    # Create sample data
    X_sample = np.random.randn(100, 20)
    
    # Test feature pipeline
    pipeline = FeaturePipeline(scaler_type='standard')
    X_transformed = pipeline.fit_transform(X_sample)
    
    print(f"Original shape: {X_sample.shape}")
    print(f"Transformed shape: {X_transformed.shape}")

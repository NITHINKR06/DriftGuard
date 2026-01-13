"""
Feature engineering utilities for anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureEngineer:
    """
    Feature engineering pipeline for anomaly detection.
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self._init_scaler()
        
    def _init_scaler(self):
        """Initialize the scaler based on type."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler and transform features.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        return self.scaler.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        return self.scaler.transform(X)
    
    def create_time_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            time_column: Name of the datetime column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        df['hour'] = df[time_column].dt.hour
        df['day_of_week'] = df[time_column].dt.dayofweek
        df['day_of_month'] = df[time_column].dt.day
        df['month'] = df[time_column].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                                columns: list, 
                                windows: list = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for col in columns:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
        
        return df


if __name__ == "__main__":
    # Example usage
    pass

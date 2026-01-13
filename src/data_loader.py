"""
Data loading utilities for the anomaly detection project.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments to pass to pandas read functions
        
    Returns:
        DataFrame containing the loaded data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif file_path.suffix == '.json':
        return pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def train_test_split_temporal(df: pd.DataFrame, 
                               test_size: float = 0.2,
                               time_column: str = None) -> tuple:
    """
    Split data temporally (maintaining time order).
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing
        time_column: Name of the time column (if None, uses index)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if time_column:
        df = df.sort_values(time_column)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    pass

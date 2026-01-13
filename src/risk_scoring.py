"""
Risk scoring and ensemble methods for anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class RiskScorer:
    """
    Combine multiple anomaly detection models into a unified risk score.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the risk scorer.
        
        Args:
            weights: Dictionary of model weights (if None, uses equal weights)
        """
        self.weights = weights
        self.scaler = MinMaxScaler()
    
    def compute_ensemble_score(self, scores_dict):
        """
        Compute ensemble risk score from multiple models.
        
        Args:
            scores_dict: Dictionary of {model_name: anomaly_scores}
            
        Returns:
            Ensemble risk scores normalized to [0, 1]
        """
        # Normalize each model's scores to [0, 1]
        normalized_scores = {}
        for model_name, scores in scores_dict.items():
            scores = np.array(scores).reshape(-1, 1)
            normalized_scores[model_name] = self.scaler.fit_transform(scores).flatten()
        
        # Apply weights
        if self.weights is None:
            self.weights = {name: 1.0/len(scores_dict) for name in scores_dict.keys()}
        
        # Compute weighted average
        ensemble_score = np.zeros(len(list(scores_dict.values())[0]))
        for model_name, scores in normalized_scores.items():
            weight = self.weights.get(model_name, 1.0/len(scores_dict))
            ensemble_score += weight * scores
        
        return ensemble_score
    
    def assign_risk_categories(self, scores, thresholds=None):
        """
        Assign risk categories based on scores.
        
        Args:
            scores: Risk scores
            thresholds: Dictionary of thresholds for categories
                       Default: {'low': 0.33, 'medium': 0.66, 'high': 1.0}
        
        Returns:
            Array of risk categories
        """
        if thresholds is None:
            thresholds = {'low': 0.33, 'medium': 0.66, 'high': 1.0}
        
        categories = []
        for score in scores:
            if score <= thresholds['low']:
                categories.append('low')
            elif score <= thresholds['medium']:
                categories.append('medium')
            else:
                categories.append('high')
        
        return np.array(categories)
    
    def get_top_anomalies(self, scores, n=10, return_indices=False):
        """
        Get the top N anomalies based on risk scores.
        
        Args:
            scores: Risk scores
            n: Number of top anomalies to return
            return_indices: If True, return indices instead of scores
            
        Returns:
            Top N anomaly scores or indices
        """
        top_indices = np.argsort(scores)[-n:][::-1]
        
        if return_indices:
            return top_indices
        else:
            return scores[top_indices]


class AnomalyExplainer:
    """
    Provide explanations for anomaly predictions.
    """
    
    def __init__(self):
        pass
    
    def get_feature_contributions(self, original_values, reconstructed_values, 
                                   feature_names=None):
        """
        Calculate feature contributions to anomaly score.
        
        Args:
            original_values: Original feature values
            reconstructed_values: Reconstructed/predicted feature values
            feature_names: Names of features
            
        Returns:
            DataFrame with feature contributions
        """
        contributions = np.abs(original_values - reconstructed_values)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(contributions))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'contribution': contributions,
            'original': original_values,
            'reconstructed': reconstructed_values
        })
        
        df = df.sort_values('contribution', ascending=False)
        return df


if __name__ == "__main__":
    # Example usage
    pass

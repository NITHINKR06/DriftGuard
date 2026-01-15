"""
Risk Scoring Module
Brain of the system - combines all models into one decision.
Normalizes scores, fuses model outputs, and assigns severity levels.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler


class RiskScorer:
    """
    Unified risk scoring system that combines multiple anomaly detection models.
    This is the core logic SOC teams care about.
    """
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 severity_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize risk scorer.
        
        Args:
            weights: Model weights {'iforest': 0.4, 'autoencoder': 0.3, 'lstm': 0.3}
            severity_thresholds: Thresholds for severity levels
        """
        # Default weights (can be tuned)
        self.weights = weights or {
            'iforest': 0.35,
            'autoencoder': 0.35, 
            'lstm': 0.30
        }
        
        # Default severity thresholds
        self.severity_thresholds = severity_thresholds or {
            'high': 0.7,
            'medium': 0.4
        }
        
        self.scalers = {}
        self.is_fitted = False
    
    def fit(self, model_scores: Dict[str, np.ndarray]) -> 'RiskScorer':
        """
        Fit the risk scorer on training scores to learn normalization.
        
        Args:
            model_scores: Dictionary with model names as keys and score arrays as values
                         e.g., {'iforest': scores, 'autoencoder': errors, 'lstm': errors}
        
        Returns:
            Self
        """
        for model_name, scores in model_scores.items():
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(scores.reshape(-1, 1))
            self.scalers[model_name] = scaler
        
        self.is_fitted = True
        print(f"Risk scorer fitted on {len(model_scores)} models")
        return self
    
    def normalize_scores(self, model_scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize all model scores to [0, 1] range.
        Higher normalized score = higher risk.
        
        Args:
            model_scores: Dictionary of raw model scores
            
        Returns:
            Dictionary of normalized scores
        """
        normalized = {}
        
        for model_name, scores in model_scores.items():
            if model_name in self.scalers:
                # Use fitted scaler
                norm_scores = self.scalers[model_name].transform(scores.reshape(-1, 1)).flatten()
            else:
                # Fallback: min-max normalize inline
                min_val = np.min(scores)
                max_val = np.max(scores)
                if max_val - min_val > 0:
                    norm_scores = (scores - min_val) / (max_val - min_val)
                else:
                    norm_scores = np.zeros_like(scores)
            
            # For Isolation Forest: lower scores = more anomalous
            # We need to invert so higher = more risky
            if model_name == 'iforest':
                norm_scores = 1.0 - norm_scores
            
            normalized[model_name] = norm_scores
        
        return normalized
    
    def compute_risk_scores(self, model_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute unified risk scores by fusing normalized model outputs.
        
        Args:
            model_scores: Dictionary of model scores/errors
            
        Returns:
            Array of final risk scores [0, 1]
        """
        # Normalize scores
        normalized_scores = self.normalize_scores(model_scores)
        
        # Get sample count
        n_samples = len(next(iter(normalized_scores.values())))
        
        # Initialize final scores
        final_scores = np.zeros(n_samples)
        total_weight = 0.0
        
        # Weighted fusion
        for model_name, scores in normalized_scores.items():
            weight = self.weights.get(model_name, 1.0 / len(normalized_scores))
            final_scores += weight * scores
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_scores /= total_weight
        
        # Clip to [0, 1] range
        final_scores = np.clip(final_scores, 0, 1)
        
        return final_scores
    
    def assign_severity(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Assign severity levels based on risk scores.
        
        Args:
            risk_scores: Array of risk scores
            
        Returns:
            Array of severity labels ('HIGH', 'MEDIUM', 'LOW')
        """
        severity = np.empty(len(risk_scores), dtype=object)
        
        high_thresh = self.severity_thresholds['high']
        medium_thresh = self.severity_thresholds['medium']
        
        severity[risk_scores >= high_thresh] = 'HIGH'
        severity[(risk_scores >= medium_thresh) & (risk_scores < high_thresh)] = 'MEDIUM'
        severity[risk_scores < medium_thresh] = 'LOW'
        
        return severity
    
    def score_and_classify(self, model_scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete scoring pipeline: compute risk scores and assign severity.
        
        Args:
            model_scores: Dictionary of model scores/errors
            
        Returns:
            Tuple of (risk_scores, severity_labels)
        """
        risk_scores = self.compute_risk_scores(model_scores)
        severity = self.assign_severity(risk_scores)
        
        return risk_scores, severity
    
    def get_top_risks(self, risk_scores: np.ndarray, 
                     severity: np.ndarray,
                     top_n: int = 20) -> np.ndarray:
        """
        Get indices of top risk alerts.
        
        Args:
            risk_scores: Array of risk scores
            severity: Array of severity labels
            top_n: Number of top alerts to return
            
        Returns:
            Array of indices sorted by risk score (descending)
        """
        top_indices = np.argsort(risk_scores)[::-1][:top_n]
        return top_indices
    
    def print_summary(self, risk_scores: np.ndarray, severity: np.ndarray):
        """Print summary statistics of risk scoring."""
        print("\n" + "="*60)
        print("RISK SCORING SUMMARY")
        print("="*60)
        print(f"Total Samples: {len(risk_scores)}")
        print(f"Mean Risk Score: {np.mean(risk_scores):.4f}")
        print(f"Std Risk Score: {np.std(risk_scores):.4f}")
        print(f"Max Risk Score: {np.max(risk_scores):.4f}")
        print(f"Min Risk Score: {np.min(risk_scores):.4f}")
        print("\nSeverity Distribution:")
        unique, counts = np.unique(severity, return_counts=True)
        for sev, count in zip(unique, counts):
            pct = (count / len(severity)) * 100
            print(f"  {sev}: {count} ({pct:.2f}%)")
        print("="*60 + "\n")


# ==================== Standalone Functions ====================

def severity_from_score(score: float, 
                       high_threshold: float = 0.7,
                       medium_threshold: float = 0.4) -> str:
    """
    Convert a single risk score to severity level.
    
    Args:
        score: Risk score [0, 1]
        high_threshold: Threshold for HIGH severity
        medium_threshold: Threshold for MEDIUM severity
        
    Returns:
        Severity label ('HIGH', 'MEDIUM', 'LOW')
    """
    if score >= high_threshold:
        return "HIGH"
    elif score >= medium_threshold:
        return "MEDIUM"
    else:
        return "LOW"


def normalize_iforest_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize Isolation Forest scores to [0, 1] where higher = more anomalous.
    
    Args:
        scores: Raw Isolation Forest anomaly scores (negative values)
        
    Returns:
        Normalized scores [0, 1]
    """
    # IForest scores are negative, lower = more anomalous
    # Invert and normalize
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
    # Invert so higher = more anomalous
    scores_normalized = 1.0 - scores_normalized
    return scores_normalized


def normalize_reconstruction_errors(errors: np.ndarray) -> np.ndarray:
    """
    Normalize reconstruction errors (Autoencoder, LSTM) to [0, 1].
    
    Args:
        errors: Raw reconstruction/prediction errors
        
    Returns:
        Normalized errors [0, 1]
    """
    # Higher error = more anomalous (already in correct direction)
    errors_normalized = (errors - errors.min()) / (errors.max() - errors.min())
    return errors_normalized


def fuse_model_scores(iforest_scores: np.ndarray,
                     autoencoder_errors: np.ndarray,
                     lstm_errors: np.ndarray,
                     weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Fuse scores from all three models into unified risk scores.
    
    Args:
        iforest_scores: Isolation Forest anomaly scores
        autoencoder_errors: Autoencoder reconstruction errors
        lstm_errors: LSTM prediction errors
        weights: Model weights (default: equal weighting)
        
    Returns:
        Fused risk scores [0, 1]
    """
    # Default weights
    if weights is None:
        weights = {'iforest': 1/3, 'autoencoder': 1/3, 'lstm': 1/3}
    
    # Normalize each model's scores
    iforest_norm = normalize_iforest_scores(iforest_scores)
    autoencoder_norm = normalize_reconstruction_errors(autoencoder_errors)
    lstm_norm = normalize_reconstruction_errors(lstm_errors)
    
    # Weighted fusion
    risk_scores = (
        weights['iforest'] * iforest_norm +
        weights['autoencoder'] * autoencoder_norm +
        weights['lstm'] * lstm_norm
    )
    
    # Ensure [0, 1] range
    risk_scores = np.clip(risk_scores, 0, 1)
    
    return risk_scores


def compute_severity_distribution(severity_labels: np.ndarray) -> Dict[str, int]:
    """
    Compute distribution of severity levels.
    
    Args:
        severity_labels: Array of severity labels
        
    Returns:
        Dictionary with counts for each severity level
    """
    unique, counts = np.unique(severity_labels, return_counts=True)
    return dict(zip(unique, counts))


def get_high_risk_indices(risk_scores: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Get indices of high-risk alerts.
    
    Args:
        risk_scores: Array of risk scores
        threshold: Threshold for high risk
        
    Returns:
        Array of indices where risk_scores >= threshold
    """
    return np.where(risk_scores >= threshold)[0]


def explain_risk_score(iforest_score: float,
                       autoencoder_error: float,
                       lstm_error: float,
                       weights: Optional[Dict[str, float]] = None) -> str:
    """
    Generate textual explanation for a risk score.
    
    Args:
        iforest_score: Normalized Isolation Forest score
        autoencoder_error: Normalized Autoencoder error
        lstm_error: Normalized LSTM error
        weights: Model weights
        
    Returns:
        Human-readable explanation string
    """
    if weights is None:
        weights = {'iforest': 1/3, 'autoencoder': 1/3, 'lstm': 1/3}
    
    final_score = (
        weights['iforest'] * iforest_score +
        weights['autoencoder'] * autoencoder_error +
        weights['lstm'] * lstm_error
    )
    
    severity = severity_from_score(final_score)
    
    explanation = f"""
Risk Score: {final_score:.4f} (Severity: {severity})

Contributing Factors:
- Isolation Forest (rare behavior): {iforest_score:.4f} (weight: {weights['iforest']:.2f})
- Autoencoder (behavioral deviation): {autoencoder_error:.4f} (weight: {weights['autoencoder']:.2f})
- LSTM (sequential anomaly): {lstm_error:.4f} (weight: {weights['lstm']:.2f})

Explanation:
This alert was flagged due to {"strong" if final_score > 0.7 else "moderate" if final_score > 0.4 else "weak"} 
deviations from learned baseline behavior across multiple anomaly detection models.
    """.strip()
    
    return explanation


if __name__ == "__main__":
    # Example usage
    print("Risk Scoring Module - Testing")
    print("="*60)
    
    # Simulate model scores
    n_samples = 1000
    iforest_scores = np.random.randn(n_samples) * 0.5 - 0.3
    autoencoder_errors = np.abs(np.random.randn(n_samples)) * 0.01
    lstm_errors = np.abs(np.random.randn(n_samples)) * 0.02
    
    # Create risk scorer
    scorer = RiskScorer()
    
    # Fit on training data
    model_scores = {
        'iforest': iforest_scores[:800],
        'autoencoder': autoencoder_errors[:800],
        'lstm': lstm_errors[:800]
    }
    scorer.fit(model_scores)
    
    # Score test data
    test_scores = {
        'iforest': iforest_scores[800:],
        'autoencoder': autoencoder_errors[800:],
        'lstm': lstm_errors[800:]
    }
    risk_scores, severity = scorer.score_and_classify(test_scores)
    
    # Print summary
    scorer.print_summary(risk_scores, severity)
    
    # Show top risks
    top_indices = scorer.get_top_risks(risk_scores, severity, top_n=5)
    print("Top 5 Risk Alerts:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. Index {idx}: Score = {risk_scores[idx]:.4f}, Severity = {severity[idx]}")

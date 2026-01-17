# Implementation Details

**[← Previous: Methodology](04_METHODOLOGY.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Models →](06_MODELS.md)**

---

## Complete Project Structure

```
DriftGuard/
│
├── notebooks/              # Jupyter notebooks for iterative development
├── research/               # Research experiments and analysis
├── src/                    # Production-ready source code
├── dashboard/              # Streamlit visualization dashboard
├── data/                   # Datasets (Git LFS tracked)
├── models/                 # Trained models (Git LFS for large files)
├── reports/                # Generated analysis reports
├── docs/                   # Modular documentation (15 files)
│
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## Source Code Modules

### 1. Data Loader (`src/data_loader.py`)

**Purpose**: Centralize all data loading operations to avoid redundancy

**Key Functions**:

```python
# Raw data loading
load_raw_cicids_data(file_name=None) → DataFrame
load_processed_data() → DataFrame

# Preprocessed features
load_phase2_features() → Tuple[ndarray, ndarray]  # (X, y)

# Model outputs
load_iforest_scores() → ndarray
load_autoencoder_errors() → ndarray
load_lstm_errors() → ndarray
load_all_model_scores() → Dict[str, ndarray]

# Final outputs
load_final_outputs() → Tuple[ndarray, ndarray]  # (risk_scores, labels)
load_severity_labels() → ndarray

# Trained models
load_iforest_model(path) → IsolationForest
load_autoencoder_model(path) → keras.Model
load_lstm_model(path) → keras.Model
load_all_models(dir) → Dict[str, Model]

# SHAP values
load_shap_values(path) → ndarray

# Utilities
check_data_availability() → Dict[str, bool]
get_data_summary() → None  # Prints summary
```

**Usage Example**:

```python
from src.data_loader import load_phase2_features, load_all_models

# Load data
X, y = load_phase2_features()
print(f"Loaded {len(X)} samples with {X.shape[1]} features")

# Load models
models = load_all_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")
```

---

### 2. Feature Engineering (`src/features.py`)

**Purpose**: Consistent preprocessing between training and inference

**FeaturePipeline Class**:

```python
class FeaturePipeline:
    def __init__(self, scaler_type='standard'):
        """Initialize pipeline with scaler type."""
        
    def fit(self, X, feature_names=None):
        """Fit pipeline on training data."""
        
    def transform(self, X):
        """Transform using fitted parameters."""
        
    def fit_transform(self, X, feature_names=None):
        """Fit and transform in one step."""
        
    def save(self, path):
        """Save fitted pipeline."""
        
    def load(self, path):
        """Load fitted pipeline."""
```

**Standalone Utility Functions**:

```python
scale_features(df, scaler_type) → Tuple[ndarray, Scaler]
apply_scaling(X, scaler) → ndarray
remove_low_variance_features(X, threshold) → Tuple[ndarray, Selector]
prepare_features_from_dataframe(df, label_column, drop_columns) → Tuple
handle_missing_values(X, strategy) → ndarray
handle_infinite_values(X) → ndarray
preprocess_features(X, handle_missing, handle_infinite, scale, scaler) → ndarray
create_sequence_data(X, sequence_length) → Tuple[ndarray, ndarray]
save_scaler(scaler, path) → None
load_scaler(path) → Scaler
```

**Production Usage**:

```python
from src.features import FeaturePipeline

# Training phase
pipeline = FeaturePipeline(scaler_type='standard')
X_train_transformed = pipeline.fit_transform(X_train)
pipeline.save('models/feature_pipeline.pkl')

# Inference phase
pipeline_prod = FeaturePipeline()
pipeline_prod.load('models/feature_pipeline.pkl')
X_new_transformed = pipeline_prod.transform(X_new)
```

---

### 3. Model Training Scripts

#### Isolation Forest (`src/train_iforest.py`)

```python
def train_isolation_forest(X_train, params=None):
    """
    Train Isolation Forest on benign traffic.
    
    Args:
        X_train: Training data (benign samples only)
        params: Hyperparameters dict
    
    Returns:
        Trained IsolationForest model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 256,
            'random_state': 42
        }
    
    model = IsolationForest(**params)
    model.fit(X_train)
    
    return model
```

#### Autoencoder (`src/train_autoencoder.py`)

```python
def build_autoencoder(input_dim=77):
    """Build and compile autoencoder."""
    model = keras.Sequential([
        keras.layers.Dense(50, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X_train, epochs=50, batch_size=256):
    """Train autoencoder on benign traffic."""
    model = build_autoencoder(input_dim=X_train.shape[1])
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)]
    )
    
    return model, history
```

#### LSTM (`src/train_lstm.py`)

```python
def build_lstm_autoencoder(timesteps=10, features=77):
    """Build LSTM autoencoder."""
    model = keras.Sequential([
        keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True),
            input_shape=(timesteps, features)
        ),
        keras.layers.LSTM(32),
        keras.layers.RepeatVector(timesteps),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

---

### 4. Risk Scoring (`src/risk_scoring.py`)

**RiskScorer Class**:

```python
class RiskScorer:
    def __init__(self, weights=None, severity_thresholds=None):
        """Initialize with model weights and thresholds."""
        
    def fit(self, model_scores):
        """Learn normalization parameters from training scores."""
        
    def normalize_scores(self, model_scores):
        """Normalize all scores to [0, 1] range."""
        
    def compute_risk_scores(self, model_scores):
        """Compute weighted ensemble risk scores."""
        
    def assign_severity(self, risk_scores):
        """Classify severity as HIGH/MEDIUM/LOW."""
        
    def score_and_classify(self, model_scores):
        """Complete pipeline: normalize, fuse, classify."""
        
    def get_top_risks(self, risk_scores, severity, top_n):
        """Get indices of highest risk alerts."""
        
    def print_summary(self, risk_scores, severity):
        """Print statistics about risk distribution."""
```

**Standalone Functions**:

```python
severity_from_score(score, high_threshold, medium_threshold) → str
normalize_iforest_scores(scores) → ndarray
normalize_reconstruction_errors(errors) → ndarray
fuse_model_scores(if_scores, ae_errors, lstm_errors, weights) → ndarray
compute_severity_distribution(severity_labels) → Dict
get_high_risk_indices(risk_scores, threshold) → ndarray
```

---

### 5. Inference Pipeline (`src/inference.py`)

**AnomalyDetector Class**:

```python
class AnomalyDetector:
    def __init__(self):
        """Initialize detector with empty model registry."""
        self.models = {}
        self.feature_engineer = None
        
    def load_model(self, model_name, model_path, model_type):
        """Load a trained model (sklearn or keras)."""
        
    def predict(self, X, model_name=None):
        """Make predictions using loaded models."""
        
    def detect_anomalies(self, X, threshold=None, percentile=95):
        """Binary anomaly detection with threshold."""
        
    def batch_predict(self, data_path, output_path=None, batch_size=1000):
        """Batch processing for large datasets."""
```

**Production Workflow**:

```python
from src.inference import AnomalyDetector

# 1. Initialize
detector = AnomalyDetector()

# 2. Load models
detector.load_model('iforest', 'models/isolation_forest.pkl', 'sklearn')
detector.load_model('autoencoder', 'models/autoencoder.keras', 'keras')
detector.load_model('lstm', 'models/lstm_autoencoder.keras', 'keras')

# 3. Predict on new data
predictions = detector.predict(X_new)

# 4. Get binary anomalies
anomalies = detector.detect_anomalies(X_new, threshold=0.7)

# 5. Batch processing
detector.batch_predict('new_traffic.csv', 'predictions.csv')
```

---

### 6. Visualization Utilities (`src/plot_utils.py`)

```python
def save_plot(filename, phase, dpi=300):
    """Save matplotlib plot with consistent formatting."""
    base_path = f"../reports/figures/{phase}"
    os.makedirs(base_path, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, filename), 
                dpi=dpi, bbox_inches="tight")
    plt.close()
```

---

## Design Patterns & Best Practices

### 1. Separation of Concerns

- **Training scripts** (`train_*.py`) ≠ **Inference** (`inference.py`)
- **Notebooks** for exploration ≠ **src/** for production
- **Data loading** centralized in `data_loader.py`

### 2. DRY Principle (Don't Repeat Yourself)

```python
# Bad: Repeat loading logic everywhere
df = pd.read_csv('data/processed_phase1.csv')  # Repeated 10 times

# Good: Centralize in data_loader
from src.data_loader import load_processed_data
df = load_processed_data()  # Used everywhere
```

### 3. Configuration Over Hardcoding

```python
# Good practice
CONFIG = {
    'iforest': {'n_estimators': 100, 'contamination': 0.1},
    'autoencoder': {'epochs': 50, 'batch_size': 256},
    'lstm': {'epochs': 50, 'batch_size': 128}
}

model = train_iforest(**CONFIG['iforest'])
```

### 4. Type Hints for Clarity

```python
from typing import Dict, Tuple, Optional
import numpy as np

def load_phase2_features() -> Tuple[np.ndarray, np.ndarray]:
    """Load processed features and labels."""
    ...
    return X, y
```

### 5. Docstrings for Documentation

```python
def compute_risk_scores(model_scores: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute unified risk scores by fusing model outputs.
    
    Args:
        model_scores: Dictionary with keys 'iforest', 'autoencoder', 'lstm'
                     and values as numpy arrays of scores/errors
    
    Returns:
        Array of final risk scores in range [0, 1]
        
    Example:
        >>> scores = load_all_model_scores()
        >>> risk_scores = compute_risk_scores(scores)
        >>> print(risk_scores.shape)
        (100000,)
    """
    ...
```

---

## Error Handling & Logging

### Robust Error Handling

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_safe(model_path):
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
```

---

## Testing (Future Work)

### Unit Test Example

```python
# tests/test_risk_scoring.py
import pytest
import numpy as np
from src.risk_scoring import normalize_iforest_scores

def test_normalize_iforest_scores():
    # Test data
    scores = np.array([-0.5, -0.3, -0.1])
    
    # Normalize
    normalized = normalize_iforest_scores(scores)
    
    # Assertions
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert len(normalized) == len(scores)
```

---

**Next: [Model Architectures →](06_MODELS.md)**

**[← Previous: Methodology](04_METHODOLOGY.md) | [Back to Index](00_INDEX.md)**

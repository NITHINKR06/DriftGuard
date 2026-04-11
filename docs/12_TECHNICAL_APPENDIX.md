# Technical Appendix

**[← Previous: Deployment](11_DEPLOYMENT.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Version Control →](13_VERSION_CONTROL.md)**

---

## Hardware & Software Environment

### Development Environment

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 11 |
| **Python** | 3.10+ |
| **CPU** | Intel i7-8750H (6 cores) |
| **RAM** | 16GB DDR4 |
| **Storage** | 512GB SSD |
| **GPU** | None (CPU-only training) |

### Production Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux Ubuntu 20.04+ | Linux Ubuntu 22.04+ |
| **Python** | 3.10 | 3.11 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8GB | 32GB |
| **Storage** | 50GB | 200GB SSD |
| **GPU** | Optional | NVIDIA (TensorFlow compatible) |

---

## Complete Hyperparameters

### Isolation Forest

```python
IFOREST_PARAMS = {
    'n_estimators': 100,          # Number of trees
    'contamination': 0.1,          # Expected outlier fraction
    'max_samples': 256,            # Subsample size per tree
    'max_features': 1.0,           # Features per split
    'bootstrap': False,            # No resampling
    'n_jobs': -1,                  # Use all CPU cores
    'random_state': 42,            # Reproducibility
    'verbose': 0                   # No training output
}
```

### Autoencoder

```python
AUTOENCODER_CONFIG = {
    'architecture': [77, 50, 25, 50, 77],
    'activation': {
        'encoder': 'relu',
        'decoder_hidden': 'relu',
        'output': 'linear'
    },
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'mse',
    'batch_size': 256,
    'epochs': 50,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'shuffle': True
}
```

### LSTM Autoencoder

```python
LSTM_CONFIG = {
    'window_size': 10,
    'features': 77,
    'architecture': {
        'encoder': [
            ('bidirectional_lstm', 64, True),
            ('lstm', 32, False)
        ],
        'decoder': [
            ('repeat_vector', 10),
            ('lstm', 32, True),
            ('lstm', 64, True),
            ('time_distributed_dense', 77)
        ]
    },
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'mse',
    'batch_size': 128,
    'epochs': 50,
    'validation_split': 0.2
}
```

### Ensemble Weights

```python
ENSEMBLE_WEIGHTS = {
    'iforest': 0.3,
    'autoencoder': 0.4,
    'lstm': 0.3
}

SEVERITY_THRESHOLDS = {
    'high': 0.7,
    'medium': 0.4
}
```

---

## Performance Benchmarks

### Training Time (CICIDS2017, ~2M samples)

| Model | Training Time | Hardware |
|-------|---------------|----------|
| Isolation Forest | ~5 minutes | CPU (6 cores) |
| Autoencoder | ~15 minutes | CPU (6 cores) |
| LSTM | ~45 minutes | CPU (6 cores) |
| **Total** | **~65 minutes** | - |

### Inference Time

| Operation | Samples | Time | Throughput |
|-----------|---------|------|------------|
| Isolation Forest | 1000 | 50ms | ~20,000/s |
| Autoencoder | 1000 | 120ms | ~8,333/s |
| LSTM | 1000 | 250ms | ~4,000/s |
| **Ensemble (sequential)** | **1000** | **420ms** | **~2,380/s** |
| SHAP Explanation | 1 | ~2s | - |

### Memory Usage

| Phase | RAM Usage | Notes |
|-------|-----------|-------|
| Data Loading | ~2GB | Loading CICIDS2017 CSV |
| Feature Engineering | ~4GB | Peak during scaling |
| Isolation Forest Training | ~3GB | Tree construction |
| Autoencoder Training | ~5GB | TensorFlow overhead |
| LSTM Training | ~6GB | Sequence creation |
| Inference (batch 1000) | ~1GB | Minimal memory |

---

## Troubleshooting Guide

### Issue 1: Memory Error During LSTM Training

**Symptom**:
```
MemoryError: Unable to allocate array
```

**Cause**: Sequence creation consumes too much RAM

**Solution**:
```python
# Option 1: Reduce batch size
model.fit(X_seq, X_seq, batch_size=64)  # Instead of 128

# Option 2: Use data generators
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

generator = TimeseriesGenerator(
    data=X_benign,
    targets=X_benign,
    length=10,
    batch_size=64
)

model.fit(generator, epochs=50)
```

### Issue 2: Model Loading Error

**Symptom**:
```
ValueError: File format not supported (got .h5, expected .keras)
```

**Cause**: Keras 3.0 changed default format

**Solution**:
```python
# Use .keras extension
model.save('models/autoencoder.keras')  # Not .h5

# Load with compile=False if needed
model = keras.models.load_model('models/autoencoder.keras', compile=False)
```

### Issue 3: Feature Dimension Mismatch

**Symptom**:
```
ValueError: Input shape (49,) does not match expected (77,)
```

**Cause**: UNSW-NB15 has different features

**Solution**:
```python
from src.features import align_features

# Align to CICIDS2017 feature set
X_aligned = align_features(
    X_unsw,
    reference_features=X_cicids.columns
)
```

### Issue 4: Infinite Values in Data

**Symptom**:
```
RuntimeWarning: overflow encountered in divide
```

**Cause**: Flow rate features (bytes/s, packets/s) contain inf

**Solution**:
```python
# Replace infinites before scaling
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)  # Or dropna() to remove
```

### Issue 5: SHAP Takes Too Long

**Symptom**: SHAP computation hangs for hours

**Cause**: Computing SHAP for large dataset

**Solution**:
```python
# Sample subset
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

# Or use approximate methods
import shap
explainer = shap.KernelExplainer(model.predict, X_train[:100])
```

---

## Code Optimization Tips

### Vectorization

```python
# Slow (loop)
errors = []
for i in range(len(X)):
    error = np.mean((X[i] - X_recon[i])**2)
    errors.append(error)

# Fast (vectorized)
errors = np.mean((X - X_recon)**2, axis=1)
```

### Batch Processing

```python
# Process in chunks to avoid memory issues
chunk_size = 10000
predictions = []

for i in range(0, len(X), chunk_size):
    chunk = X[i:i+chunk_size]
    chunk_pred = model.predict(chunk)
    predictions.append(chunk_pred)

predictions = np.concatenate(predictions)
```

### GPU Acceleration (Optional)

```python
# Enable GPU for TensorFlow
import tensorflow as tf

# Check GPU availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

## Data Statistics

### CICIDS2017 Statistics

| Metric | Value |
|--------|-------|
| Total Flows | ~2.8M |
| Benign | ~2.3M (82%) |
| Attack | ~500K (18%) |
| Features | 78 → 77 used |
| File Size (CSV) | ~2.5GB |
| Processed (NPY) | ~250MB |

### UNSW-NB15 Statistics

| Metric | Value |
|--------|-------|
| Total Flows | ~2.5M |
| Benign | ~2.2M (87%) |
| Attack | ~300K (13%) |
| Features | 49 |
| File Size (CSV) | ~2.0GB |

---

## File Format Specifications

### NumPy Arrays (.npy)

```python
# Save
np.save('data/X_phase2.npy', X)

# Load
X = np.load('data/X_phase2.npy')

# File info
import os
size_mb = os.path.getsize('data/X_phase2.npy') / (1024**2)
print(f"Size: {size_mb:.2f} MB")
```

### Keras Models (.keras)

```python
# Save (Keras 3.0+ format)
model.save('models/autoencoder.keras')

# Load
model = keras.models.load_model('models/autoencoder.keras')

# Model summary
model.summary()
```

### Scikit-learn Models (.pkl)

```python
# Save
import joblib
joblib.dump(model, 'models/isolation_forest.pkl')

# Load
model = joblib.load('models/isolation_forest.pkl')

# Check model type
print(type(model))  # sklearn.ensemble.IsolationForest
```

---

## API Reference Quick Links

### TensorFlow/Keras
- Docs: https://www.tensorflow.org/api_docs
- Keras Guide: https://keras.io/guides/

### Scikit-learn
- Docs: https://scikit-learn.org/stable/
- Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

### SHAP
- Docs: https://shap.readthedocs.io/
- GitHub: https://github.com/slundberg/shap

### Pandas
- Docs: https://pandas.pydata.org/docs/
- API Reference: https://pandas.pydata.org/docs/reference/

### NumPy
- Docs: https://numpy.org/doc/stable/
- User Guide: https://numpy.org/doc/stable/user/

---

## Common Commands Reference

### Environment Setup

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Verify
python -c "import tensorflow, sklearn, shap; print('OK')"
```

### Model Training

```bash
# Train all models
python src/train_iforest.py
python src/train_autoencoder.py
python src/train_lstm.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Convert notebook to script
jupyter nbconvert --to script notebook.ipynb

# Execute notebook
jupyter nbconvert --to notebook --execute notebook.ipynb
```

### Git LFS

```bash
# Install LFS
git lfs install

# Track large files
git lfs track "data/**/*.csv"
git lfs track "data/**/*.npy"

# Pull LFS files
git lfs pull

# Check LFS status
git lfs ls-files
```

---

**Next: [Version Control →](13_VERSION_CONTROL.md)**

**[← Previous: Deployment](11_DEPLOYMENT.md) | [Back to Index](00_INDEX.md)**

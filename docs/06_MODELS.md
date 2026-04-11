# Model Architectures

**[← Previous: Implementation](05_IMPLEMENTATION.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Experiments →](07_EXPERIMENTS.md)**

---

## Ensemble Overview

DriftGuard uses **three complementary models** that capture different types of anomalies:

| Model | Type | Strength | Training Data |
|-------|------|----------|---------------|
| **Isolation Forest** | Tree-based | Fast, interpretable, rare events | Benign traffic |
| **Autoencoder** | Neural network | Reconstruction errors | Benign traffic |
| **LSTM Autoencoder** | Recurrent NN | Temporal patterns | Benign sequences |

---

## Model 1: Isolation Forest

**File**: `src/train_iforest.py`, `notebooks/03_isolation_forest.ipynb`

### Algorithm Overview

Isolation Forest isolates anomalies by randomly partitioning the feature space. Anomalies are easier to isolate (shorter path length) than normal points.

### Architecture

```
Input: 77 features
│
├── Tree 1: Random feature splits
├── Tree 2: Random feature splits
├── ...
└── Tree 100: Random feature splits
│
Output: Average path length → Anomaly score
```

### Hyperparameters

```python
IsolationForest(
    n_estimators=100,        # Number of trees
    contamination=0.1,       # Expected proportion of anomalies
    max_samples=256,         # Samples to draw for each tree
    max_features=1.0,        # Features to consider for splits
    bootstrap=False,         # Don't resample
    random_state=42          # Reproducibility
)
```

### Training Process

```python
from sklearn.ensemble import IsolationForest

# 1. Train on benign traffic only
X_benign = X_train[y_train == 0]

# 2. Initialize model
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    max_samples=256,
    random_state=42
)

# 3. Fit
model.fit(X_benign)

# 4. Predict anomaly scores (negative values)
scores = model.decision_function(X_test)
# Lower score = more anomalous

# 5. Binary predictions
predictions = model.predict(X_test)
# -1 = anomaly, 1 = normal

# 6. Save
import joblib
joblib.dump(model, 'models/isolation_forest.pkl')
```

### Output Interpretation

- **Anomaly Score**: `decision_function(X)` returns negative values
- **Lower score** = More anomalous (e.g., -0.5 more anomalous than -0.1)
- **Threshold**: Typically use `contamination` parameter or percentile

### Advantages

✅ Fast training and inference  
✅ No assumptions about data distribution  
✅ Handles high-dimensional data well  
✅ Interpretable with SHAP TreeExplainer  

### Limitations

❌ Sensitive to contamination parameter  
❌ May miss complex, multi-modal anomalies  
❌ Performance degen rates with many irrelevant features  

---

## Model 2: Autoencoder

**File**: `src/train_autoencoder.py`, `notebooks/04_autoencoder.ipynb`

### Architecture

```
Input Layer:    77 features
      ↓
Encoder:        Dense(50, ReLU)
      ↓
Bottleneck:     Dense(25, ReLU)  ← Compressed representation
      ↓
Decoder:        Dense(50, ReLU)
      ↓
Output Layer:   Dense(77, Linear)

Loss: Mean Squared Error (MSE)
Optimizer: Adam (lr=0.001)
```

### Complete Implementation

```python
from tensorflow import keras

def build_autoencoder(input_dim=77):
    """Build dense autoencoder."""
    
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(50, activation='relu')(encoder_input)
    encoded = keras.layers.Dense(25, activation='relu')(encoded)
    
    # Decoder
    decoded = keras.layers.Dense(50, activation='relu')(encoded)
    decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
    
    # Full autoencoder
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoded)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return autoencoder

# Train
model = build_autoencoder()
history = model.fit(
    X_benign, X_benign,  # Input = Output (reconstruction)
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

# Save
model.save('models/autoencoder.keras')
```

### Anomaly Detection Logic

```python
# 1. Reconstruct test data
X_reconstructed = model.predict(X_test)

# 2. Compute reconstruction error (MSE per sample)
reconstruction_errors = np.mean(
    (X_test - X_reconstructed) ** 2,
    axis=1
)

# 3. Higher error = more anomalous
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold
```

### Training Curves

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.hist(reconstruction_errors[y_test == 0], bins=50, alpha=0.5, label='Benign')
plt.hist(reconstruction_errors[y_test == 1], bins=50, alpha=0.5, label='Attack')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.legend()
plt.title('Error Distribution')
```

### Advantages

✅ Learns complex, non-linear patterns  
✅ Captures correlations between features  
✅ Good for high-dimensional data  
✅ No assumption about anomaly distribution  

### Limitations

❌ Requires careful architecture design  
❌ Prone to overfitting on training anomalies  
❌ Slower training than Isolation Forest  
❌ Black-box (requires SHAP for interpretability)  

---

## Model 3: LSTM Autoencoder

**File**: `src/train_lstm.py`, `notebooks/05_lstm_sequence_model.ipynb`

### Architecture

```
Input: (10, 77)  # 10 timesteps, 77 features per timestep
      ↓
Encoder:
  Bidirectional LSTM(64, return_sequences=True)
  LSTM(32)
      ↓
Bottleneck: 32-d representation
      ↓
Decoder:
  RepeatVector(10)  # Repeat for 10 timesteps
  LSTM(32, return_sequences=True)
  LSTM(64, return_sequences=True)
  TimeDistributed Dense(77)
      ↓
Output: (10, 77)  # Reconstructed sequence

Loss: MSE
Optimizer: Adam (lr=0.001)
```

### Complete Implementation

```python
def build_lstm_autoencoder(timesteps=10, features=77):
    """Build LSTM autoencoder for temporal patterns."""
    
    # Input
    inputs = keras.layers.Input(shape=(timesteps, features))
    
    # Encoder
    encoded = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True)
    )(inputs)
    encoded = keras.layers.LSTM(32)(encoded)
    
    # Decoder
    decoded = keras.layers.RepeatVector(timesteps)(encoded)
    decoded = keras.layers.LSTM(32, return_sequences=True)(decoded)
    decoded = keras.layers.LSTM(64, return_sequences=True)(decoded)
    decoded = keras.layers.TimeDistributed(
        keras.layers.Dense(features)
    )(decoded)
    
    # Model
    lstm_ae = keras.Model(inputs=inputs, outputs=decoded)
    lstm_ae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return lstm_ae
```

### Sequence Creation

```python
def create_sequences(X, window_size=10):
    """Create sliding window sequences."""
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
    return np.array(X_seq)

# Create sequences
X_benign_seq = create_sequences(X_benign, window_size=10)
# Shape: (N_samples, 10, 77)

#Train
model = build_lstm_autoencoder(timesteps=10, features=77)
history = model.fit(
    X_benign_seq, X_benign_seq,
    epochs=50,
    batch_size=128,
    validation_split=0.2
)

model.save('models/lstm_autoencoder.keras')
```

### Anomaly Detection

```python
# 1. Create test sequences
X_test_seq = create_sequences(X_test, window_size=10)

# 2. Reconstruct
X_test_reconstructed = model.predict(X_test_seq)

# 3. Compute sequence errors
sequence_errors = np.mean(
    (X_test_seq - X_test_reconstructed) ** 2,
    axis=(1, 2)  # Average over timesteps and features
)

# 4. Threshold
threshold = np.percentile(sequence_errors, 95)
anomalies = sequence_errors > threshold
```

### Advantages

✅ Captures temporal dependencies  
✅ Detects sequential anomalies (e.g., port scanning)  
✅ Bidirectional LSTM sees past and future context  
✅ Effective for time-series attacks  

### Limitations

❌ Slowest training time (~45 minutes)  
❌ Requires sequence creation (windowing)  
❌ Less effective for instantaneous anomalies  
❌ Temporal patterns may not generalize cross-dataset  

---

## Hyperparameter Tuning Results

### Isolation Forest Grid Search

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'contamination': [0.05, 0.1, 0.15],
    'max_samples': [128, 256, 512]
}

# Best parameters (via cross-validation)
best_params = {
    'n_estimators': 100,
    'contamination': 0.1,
    'max_samples': 256
}
```

### Autoencoder Architecture Search

| Hidden Layers | Bottleneck | Val Loss | Selected |
|---------------|------------|----------|----------|
| [50, 25] | 25 | 0.0023 | ✅ Yes |
| [64, 32] | 32 | 0.0025 | No |
| [100, 50, 25] | 25 | 0.0028 | No |
| [128, 64, 32] | 32 | 0.0031 | No |

### LSTM Configuration

| Window Size | LSTM Units | Val Loss | Selected |
|-------------|------------|----------|----------|
| 5 | [64, 32] | 0.0045 | No |
| **10** | **[64, 32]** | **0.0039** | **✅ Yes** |
| 15 | [64, 32] | 0.0042 | No |
| 20 | [64, 32] | 0.0048 | No |

---

## Model Comparison Summary

| Aspect | Isolation Forest | Autoencoder | LSTM |
|--------|-----------------|-------------|------|
| **Training Time** | 5 min | 15 min | 45 min |
| **Inference (1000 samples)** | 50ms | 120ms | 250ms |
| **Model Size** | 1.8 MB | 200 KB | 936 KB |
| **AUC-ROC** | 0.88 | 0.91 | 0.89 |
| **Precision** | 0.69 | 0.52 | 0.57 |
| **Interpretability** | High (SHAP) | Medium (SHAP) | Low |
| **Best For** | Rare outliers | Complex patterns | Temporal attacks |

---

**Next: [Experiments & Results →](07_EXPERIMENTS.md)**

**[← Previous: Implementation](05_IMPLEMENTATION.md) | [Back to Index](00_INDEX.md)**

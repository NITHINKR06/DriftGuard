# Deployment Guide

**[← Previous: Visualization & Dashboard](10_VISUALIZATION_DASHBOARD.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Technical Appendix →](12_TECHNICAL_APPENDIX.md)**

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ for datasets and models

### Step 1: Clone Repository

```bash
git clone https://github.com/NITHINKR06/DriftGuard.git
cd DriftGuard
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac OS)
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; import sklearn; import shap; print('All packages installed successfully!')"
```

---

## Training Pipeline

### Quick Start: Train All Models

```bash
# Step 1: Data Exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Step 2: Feature Engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# Step 3: Train Isolation Forest
python src/train_iforest.py

# Step 4: Train Autoencoder
python src/train_autoencoder.py

# Step 5: Train LSTM
python src/train_lstm.py

# Step 6: Evaluate Models
jupyter notebook notebooks/06_model_evaluation.ipynb
```

### Trained Model Outputs

After training, models are saved to `models/`:
- `isolation_forest.pkl` (1.8 MB)
- `autoencoder.keras` (200 KB)
- `lstm_autoencoder.keras` (936 KB)

---

## Inference Usage

### Python API

```python
from src.inference import AnomalyDetector
from src.data_loader import load_phase2_features

# Initialize detector
detector = AnomalyDetector()

# Load trained models
detector.load_model('iforest', 'models/isolation_forest.pkl', 'sklearn')
detector.load_model('autoencoder', 'models/autoencoder.keras', 'keras')
detector.load_model('lstm', 'models/lstm_autoencoder.keras', 'keras')

# Load test data
X_test, y_test = load_phase2_features()

# Make predictions
predictions = detector.predict(X_test)

# Detect anomalies with threshold
anomalies = detector.detect_anomalies(X_test, threshold=0.7)

# Batch processing
results = detector.batch_predict(
    data_path='data/new_traffic.csv',
    output_path='results/predictions.csv'
)
```

### Risk Scoring

```python
from src.risk_scoring import RiskScorer
from src.data_loader import load_all_model_scores

# Load model outputs
scores = load_all_model_scores()

# Initialize risk scorer
scorer = RiskScorer(weights={
    'iforest': 0.3,
    'autoencoder': 0.4,
    'lstm': 0.3
})

# Compute risk scores and severity
risk_scores, severity = scorer.score_and_classify(scores)

# Get top 20 high-risk alerts
top_risks = scorer.get_top_risks(risk_scores, severity, top_n=20)

# Print summary
scorer.print_summary(risk_scores, severity)
```

---

## Dashboard Deployment

### Local Development

```bash
# Navigate to dashboard directory
cd dashboard

# Run Streamlit app
streamlit run app.py

# Dashboard opens at http://localhost:8501
```

### Production Deployment

#### Option 1: Streamlit Cloud (Free)

```bash
# Deploy to Streamlit Cloud
streamlit deploy dashboard/app.py
```

#### Option 2: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t driftguard-dashboard .
docker run -p 8501:8501 driftguard-dashboard
```

---

## Production Deployment Considerations

### 1. Scalability

**Batch Processing**
```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_traffic.csv', chunksize=chunk_size):
    predictions = detector.predict(chunk)
    # Save incrementally
```

**Model Serving** with TensorFlow Serving:
```bash
# Export models for serving
tensorflow_model_server --model_base_path=/models/autoencoder --rest_api_port=8501
```

**Distributed Processing** with Apache Spark:
```python
from pyspark.sql import SparkSession
# Distribute inference across cluster
```

### 2. Monitoring

**Track Performance Metrics**
```python
import mlflow

# Log metrics
mlflow.log_metric("auc", 0.93)
mlflow.log_metric("precision", 0.87)
mlflow.log_artifact("models/autoencoder.keras")
```

**Dataset Drift Detection**
```python
from src.risk_scoring import compute_psi

# Monitor for drift
psi = compute_psi(expected=X_train, actual=X_production)
if psi > 0.25:
    alert("Significant dataset drift detected!")
```

### 3. Model Updates

```python
# A/B Testing
if random.random() < 0.1:
    prediction = model_v2.predict(X)  # 10% traffic to new model
else:
    prediction = model_v1.predict(X)  # 90% to current model
```

### 4. Integration Points

**SIEM Integration** (Splunk, ELK):
```python
import json
import requests

# Send alerts to SIEM
alert = {
    "timestamp": timestamp,
    "risk_score": risk_score,
    "severity": severity,
    "src_ip": src_ip,
    "dst_ip": dst_ip
}
requests.post("http://siem-endpoint/api/alerts", json=alert)
```

**SOAR Integration**:
```python
# Trigger automated response
if severity == "HIGH":
    soar.create_incident(alert)
    soar.quarantine_ip(src_ip)
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory

**Symptoms**: `MemoryError` during LSTM training

**Solution**:
```python
# Reduce batch size
model.fit(X_train, X_train, batch_size=64)  # Instead of 128

# Use data generators
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
generator = Times eriesGenerator(X_train, X_train, length=10, batch_size=64)
model.fit(generator, epochs=50)
```

#### Issue 2: Model Loading Errors

**Symptoms**: `ValueError` when loading `.keras` files

**Solution**:
```python
# Use correct Keras version
pip install keras>=3.0.0

# Load with custom objects if needed
from tensorflow.keras.models import load_model
model = load_model('models/autoencoder.keras', compile=False)
```

#### Issue 3: Feature Dimension Mismatch

**Symptoms**: Shape errors during inference

**Solution**:
```python
from src.features import align_features

# Align features to training set
X_aligned = align_features(X_new, reference_features=X_train.columns)
```

---

## Performance Benchmarks

| Operation | Time | Throughput |
|-----------|------|------------|
| Load CICIDS2017 CSV | ~30s | - |
| Train Isolation Forest | ~5 min | - |
| Train Autoencoder | ~15 min | - |
| Train LSTM | ~45 min | - |
| Inference (1000 samples) | 420ms | ~2,380 samples/s |
| SHAP Explanation (1 sample) | ~2s | - |

**Hardware**: i7-8750H, 16GB RAM, No GPU

---

## Next Steps

✅ **Trained models** ready in `models/`  
✅ **Dashboard** running locally  
📋 **Next**: Deploy to production, integrate with Zeek  

---

**Next: [Technical Appendix →](12_TECHNICAL_APPENDIX.md)**

**[← Previous: Visualization & Dashboard](10_VISUALIZATION_DASHBOARD.md) | [Back to Index](00_INDEX.md)**

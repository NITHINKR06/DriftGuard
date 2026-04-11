# DriftGuard

DriftGuard is a network intrusion detection system (IDS) that uses an ensemble of anomaly detection models to identify malicious network behavior under dataset shift.

Unlike many IDS projects that only report strong results on one benchmark, DriftGuard is designed and validated for cross-environment reliability using:

- CICIDS2017 for training
- UNSW-NB15 for cross-dataset validation
- UGR16v2noIRC for real-world ISP-style validation

## Why This Project Exists

A common IDS failure pattern is:

1. High accuracy in the training environment
2. Significant degradation after deployment in a different network
3. Increased false positives and missed true attacks

DriftGuard addresses this by combining complementary detectors, quantifying distribution shift, and providing explanation artifacts for analyst trust.

## What DriftGuard Solves

- Detects anomalous network flows with an ensemble approach
- Reduces dependence on a single model's blind spots
- Provides calibrated risk scoring (0 to 1)
- Supports severity-based triage (LOW, MEDIUM, HIGH)
- Includes SHAP-based explainability for flagged traffic
- Validates generalization across heterogeneous datasets

## System Overview

### Ensemble Models

DriftGuard combines three anomaly engines:

1. Isolation Forest (weight: 0.3)
- Tree-based unsupervised outlier detection
- Fast inference for point anomalies

2. Autoencoder (weight: 0.4)
- Dense reconstruction network: 77 -> 50 -> 25 -> 50 -> 77
- Detects deviations via reconstruction error

3. LSTM Autoencoder (weight: 0.3)
- Bidirectional LSTM with sequence windows
- Captures temporal and sequence-level anomalies

### Risk Fusion

Model outputs are normalized to [0, 1] and fused as:

risk = 0.3 * IF + 0.4 * AE + 0.3 * LSTM

### Severity Bands

- HIGH: score >= 0.7
- MEDIUM: 0.4 <= score < 0.7
- LOW: score < 0.4

## End-to-End Workflow

1. Data ingestion and preprocessing
- Load raw network flow CSV files
- Clean missing/infinite values
- Build and align feature matrices
- Apply scaling and persist arrays

2. Model training (benign-first)
- Train all models on benign traffic patterns
- Save model artifacts for reuse

3. Inference and risk scoring
- Generate per-sample model scores
- Compute fused ensemble risk
- Convert to severity labels

4. Explainability
- Use SHAP to attribute high-risk outputs to features
- Produce analyst-friendly interpretation plots

5. Cross-dataset validation
- Evaluate transfer from CICIDS2017 to UNSW-NB15 and UGR16
- Measure shift with KS/PSI statistics

## Repository Structure

```text
.
├── dashboard/              # Streamlit SOC dashboard
├── data/                   # Datasets and intermediate .npy artifacts
├── docs/                   # Extended project documentation
├── models/                 # Trained model files (.pkl, .keras)
├── notebooks/              # Notebook-based research workflow
├── reports/                # Generated reports and figures
├── research/               # Experiment notebooks and notes
├── results/                # Output artifacts
├── src/                    # Core pipeline code
├── filestr.md
├── index.html
├── requirements.txt
└── README.md
```

### Important Source Modules

- src/data_loader.py: loading, cleaning, and final output utilities
- src/features.py: feature engineering transformations
- src/train_iforest.py: Isolation Forest training
- src/train_autoencoder.py: dense autoencoder training
- src/train_lstm.py: sequence model training
- src/risk_scoring.py: score normalization, fusion, and severity mapping
- src/inference.py: model loading and prediction API
- dashboard/app.py: Streamlit SOC dashboard for risk monitoring

## Setup

### Prerequisites

- Python 3.10+ recommended
- Sufficient RAM for dataset processing and deep learning training

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1) Train Models

```bash
python src/train_iforest.py
python src/train_autoencoder.py
python src/train_lstm.py
```

Expected artifacts in models/:

- iforest_model.pkl
- autoencoder.keras
- lstm_autoencoder.keras

### 2) Run Evaluation and Validation

```bash
jupyter notebook notebooks/06_model_evaluation.ipynb
jupyter notebook notebooks/09_cross_dataset_validation.ipynb
```

### 3) Run Explainability

```bash
jupyter notebook notebooks/07_explainability_shap.ipynb
```

### 4) Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Inference API Example

```python
from src.inference import AnomalyDetector

# Initialize detector
adet = AnomalyDetector()

# Load trained models
adet.load_model('iforest', 'models/iforest_model.pkl', 'sklearn')
adet.load_model('autoencoder', 'models/autoencoder.keras', 'keras')
adet.load_model('lstm', 'models/lstm_autoencoder.keras', 'keras')

# Predict and score
preds = adet.predict(X_test)
risk = adet.get_risk_scores(X_test)
```

## Dashboard Behavior

The Streamlit dashboard:

- loads final risk outputs and labels
- computes severity from risk scores
- displays key SOC metrics
- shows severity distribution
- lists top risk events for triage

## Datasets

### CICIDS2017 (training baseline)

- primary source for model fitting and scaler baseline
- modern labeled network traffic benchmark

### UNSW-NB15 (cross-dataset validation)

- used to test domain transfer from training distribution
- requires feature alignment into the common feature space

### UGR16v2noIRC (real-world ISP validation)

- operationally realistic flow data from ISP-like traffic
- used to stress-test transferability under stronger drift

## Outputs and Artifacts

Common outputs include:

- trained model files in models/
- processed arrays and scores in data/
- figures and evaluation artifacts in reports/ and research/results/
- dashboard-ready risk outputs consumed by dashboard/app.py

## Current Limitations

- Feature alignment across datasets can hide source-specific signals
- Severity thresholds may require environment-specific tuning
- Concept drift over time still requires periodic recalibration/retraining
- Real-time packet pipeline integration is planned, not default

## Roadmap

### Near-term

- strengthen cross-dataset calibration workflows
- improve automated threshold tuning for SOC contexts
- add robustness checks for evolving traffic profiles

### Planned

- Zeek-based real-time inference pipeline
- MITRE ATT&CK tactic/technique mapping for alerts

## Documentation

For detailed technical writeups, see docs/:

- PROJECT_OVERVIEW.md
- 02_ARCHITECTURE.md
- 05_IMPLEMENTATION.md
- 06_MODELS.md
- 08_CROSS_DATASET_VALIDATION.md
- 09_EXPLAINABILITY.md
- 10_VISUALIZATION_DASHBOARD.md

## License

MIT License

# DriftGuard

A network intrusion detection system that uses ensemble machine learning (Isolation Forest, Autoencoder, LSTM) to identify cyber attacks in network traffic. DriftGuard maintains strong performance across different network environments - validated on CICIDS2017, UNSW-NB15, and UGR16v2noIRC datasets.

## 📁 Project Structure

```
.
├── notebooks/              # Jupyter notebooks for model development
│   ├── 01_data_exploration.ipynb          # CICIDS2017 dataset exploration
│   ├── 02_feature_engineering.ipynb       # Feature engineering pipeline
│   ├── 03_isolation_forest.ipynb          # Isolation Forest training
│   ├── 04_autoencoder.ipynb               # Autoencoder training
│   ├── 05_lstm_sequence_model.ipynb       # LSTM autoencoder training
│   ├── 06_model_evaluation.ipynb          # Model evaluation metrics
│   ├── 07_explainability_shap.ipynb       # SHAP-based model explainability
│   └── 09_cross_dataset_validation.ipynb  # UNSW-NB15 cross-validation
│
├── research/               # Research experiments and analysis
│   ├── experiments/
│   │   ├── exp1_model_comparison.ipynb    # Comparative model analysis
│   │   ├── exp2_false_positive_analysis.ipynb # False positive investigation
│   │   ├── exp3_dataset_shift_analysis.ipynb  # Distribution shift analysis
│   │   └── exp4_ugr16_real_world_validation.ipynb # UGR16 validation
│   ├── results/            # Experiment results and figures
│   └── notes.md            # Research notes
│
├── src/                    # Source code modules
│   ├── data_loader.py      # Data loading and preprocessing utilities
│   ├── features.py         # Feature engineering transformations
│   ├── train_iforest.py    # Isolation Forest training module
│   ├── train_autoencoder.py # Autoencoder training module
│   ├── train_lstm.py       # LSTM autoencoder training module
│   ├── risk_scoring.py     # Ensemble risk scoring and fusion
│   ├── inference.py        # End-to-end model inference
│   ├── plot_utils.py       # Visualization utilities
│   └── fix_plot_saving.py  # Plot saving helper functions
│
├── dashboard/              # Dashboard application (future)
├── data/                   # Dataset storage (CICIDS2017, UNSW-NB15, UGR16)
├── models/                 # Saved trained models (.pkl, .keras)
├── reports/                # Generated analysis reports
├── results/                # Output results and visualizations
└── README.md
```

## 🎯 Project Overview

DriftGuard is a network intrusion detection system that combines three AI models to detect network attacks:

**What it does:**
- **Detects** network attacks using ensemble ML models (Isolation Forest, Autoencoder, LSTM)
- **Works** across different network environments without retraining
- **Validates** on three datasets: CICIDS2017 (training), UNSW-NB15, and UGR16v2noIRC (validation)
- **Explains** why traffic was flagged using SHAP analysis
- **Scores** anomalies using weighted ensemble fusion (0.3 IF + 0.4 AE + 0.3 LSTM)
- **Processes** ~2,380 network flows per second with low latency

**Key advantage:** Most intrusion detection systems fail when deployed in different networks than where they were trained. DriftGuard maintains robust performance across diverse environments - from academic networks to real-world ISP traffic.

## 🚀 Getting Started

### Prerequisites

```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow, keras
- **Visualization**: matplotlib, seaborn, plotly
- **Explainability**: shap
- **Model Persistence**: joblib
- **Development**: jupyter, pytest

## 🔧 Usage

### 1. Data Exploration

Explore the CICIDS2017 dataset:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Training Models

Train individual models using the provided scripts:

```bash
# Train Isolation Forest
python src/train_iforest.py

# Train Autoencoder
python src/train_autoencoder.py

# Train LSTM Autoencoder
python src/train_lstm.py
```

Models are saved to the `models/` directory:
- `iforest_model.pkl` - Isolation Forest
- `autoencoder.keras` - Autoencoder
- `lstm_autoencoder.keras` - LSTM Autoencoder

### 3. Cross-Dataset Validation

Validate models on UNSW-NB15 dataset:
```bash
jupyter notebook notebooks/09_cross_dataset_validation.ipynb
```

### 4. Making Predictions

Use the inference module for end-to-end detection:

```python
from src.inference import AnomalyDetector

# Initialize detector
detector = AnomalyDetector()

# Load trained models
detector.load_model('iforest', 'models/iforest_model.pkl', 'sklearn')
detector.load_model('autoencoder', 'models/autoencoder.keras', 'keras')
detector.load_model('lstm', 'models/lstm_autoencoder.keras', 'keras')

# Make predictions
predictions = detector.predict(X_test)
risk_scores = detector.get_risk_scores(X_test)
```

## 📊 Models

### Isolation Forest
- Unsupervised tree-based anomaly detection
- Fast training and inference
- Effective for high-dimensional network traffic data
- Contamination factor: 0.1

### Autoencoder
- Dense neural network-based reconstruction
- Architecture: 77 → 50 → 25 → 50 → 77
- Learns normal traffic patterns via reconstruction error
- Trained on benign traffic samples only

### LSTM Autoencoder
- Sequence-based temporal pattern learning
- Bidirectional LSTM layers for time-series modeling
- Window size: 10 timesteps
- Captures temporal dependencies in network flows

## 📈 Risk Scoring System

DriftGuard uses a weighted ensemble approach combining all three models:

```python
from src.risk_scoring import RiskScorer

# Initialize with custom weights
scorer = RiskScorer(weights={
    'iforest': 0.3, 
    'autoencoder': 0.4, 
    'lstm': 0.3
})

# Compute ensemble risk score
ensemble_score = scorer.compute_ensemble_score(scores_dict)
```

Risk scores are normalized to [0, 1] where higher values indicate greater anomaly likelihood.

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) provides feature-level interpretability:

```bash
jupyter notebook notebooks/07_explainability_shap.ipynb
```

Includes:
- Waterfall plots for individual predictions
- Feature importance ranking
- False positive analysis

## 🧪 Research Experiments

Located in `research/experiments/`:

1. **Model Comparison** - Performance benchmarking across models
2. **False Positive Analysis** - SHAP-based investigation of misclassifications
3. **Dataset Shift Analysis** - Distribution comparison between CICIDS2017 and UNSW-NB15
4. **UGR16 Real-World Validation** - Additional validation on UGR16 dataset

## 📂 Datasets

### CICIDS2017 (Training)
- Canadian Institute for Cybersecurity dataset
- Modern network traffic with labeled attacks
- Training set: benign traffic samples

### UNSW-NB15 (Cross-Dataset Validation)
- University of New South Wales dataset
- Cross-dataset validation for generalization testing
- Feature alignment required for compatibility
- Preprocessed data available in `data/unsw/`

### UGR16v2noIRC (Real-World ISP Validation)
- **Source:** University of Granada
- **Version Used:** UGR16v2noIRC (IRC traffic removed for cleaner validation)
- **Features:** 135 network flow features (truncated to 78, then aligned to 77 for CICIDS compatibility)
- **Training set:** 98,262 samples (for distribution analysis only - NOT used for training)
- **Test set:** 43,200 samples (100% attack traffic from real ISP network)
- **Attack Types:** Multi-label (DoS, Port Scans, Botnet, Blacklist, Anomaly Detection)
- **Experiment:** `research/experiments/exp4_ugr16_real_world_validation.ipynb`
- **Outputs:** Preprocessed predictions and scores in `data/ugr_*.npy`

## � Future Enhancements

The following features are planned for future development and will be implemented in a Linux environment:

### Zeek Real-Time Inference
- **Integration with Zeek** (formerly Bro) network security monitor
- Real-time network traffic analysis and anomaly detection
- Live packet processing and feature extraction
- Streaming prediction pipeline for production deployment
- Requires: Zeek installation on Linux systems

### MITRE ATT&CK Auto-Mapping
- **Automatic attack technique classification** based on detected anomalies
- Mapping of network intrusions to MITRE ATT&CK framework tactics and techniques
- Enhanced threat intelligence and incident response capabilities
- Integration with security orchestration workflows
- Support for ATT&CK Navigator visualization

> [!NOTE]
> These features are currently in planning phase and will require a Linux deployment environment for production use. Development will begin after completing current research experiments on cross-dataset validation.

## �📝 License

This project is licensed under the MIT License.

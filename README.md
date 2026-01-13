# Anomaly Detection Project

A comprehensive anomaly detection system using multiple machine learning approaches including Isolation Forest, Autoencoders, and LSTM networks.

## 📁 Project Structure

```
.
├── notebooks/              # Jupyter notebooks for exploration and experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_lstm_sequence_model.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_explainability_shap.ipynb
│
├── src/                    # Source code modules
│   ├── data_loader.py      # Data loading utilities
│   ├── features.py         # Feature engineering
│   ├── train_iforest.py    # Isolation Forest training
│   ├── train_autoencoder.py # Autoencoder training
│   ├── train_lstm.py       # LSTM training
│   ├── risk_scoring.py     # Risk scoring and ensemble methods
│   └── inference.py        # Model inference
│
├── dashboard/              # Dashboard application
├── data/                   # Data directory
├── models/                 # Trained models
├── reports/                # Generated reports and visualizations
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- shap
- joblib
- jupyter

## 🔧 Usage

### 1. Data Exploration

Start with the data exploration notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Training Models

Train individual models using the provided scripts:

```bash
# Train Isolation Forest
python src/train_iforest.py --data data/your_data.csv --output models/iforest.pkl

# Train Autoencoder
python src/train_autoencoder.py --data data/your_data.csv --output models/autoencoder.h5

# Train LSTM
python src/train_lstm.py --data data/your_data.csv --output models/lstm.h5
```

### 3. Making Predictions

Use the inference module to make predictions:

```python
from src.inference import AnomalyDetector

detector = AnomalyDetector()
detector.load_model('iforest', 'models/iforest.pkl', 'sklearn')
detector.load_model('autoencoder', 'models/autoencoder.h5', 'keras')

predictions = detector.predict(X_test)
```

## 📊 Models

### Isolation Forest
- Unsupervised anomaly detection
- Fast training and inference
- Good for high-dimensional data

### Autoencoder
- Neural network-based reconstruction
- Learns normal patterns
- Detects anomalies via reconstruction error

### LSTM
- Sequence-based anomaly detection
- Temporal pattern recognition
- Suitable for time-series data

## 📈 Risk Scoring

The project includes an ensemble risk scoring system that combines predictions from multiple models:

```python
from src.risk_scoring import RiskScorer

scorer = RiskScorer(weights={'iforest': 0.3, 'autoencoder': 0.4, 'lstm': 0.3})
ensemble_score = scorer.compute_ensemble_score(scores_dict)
```

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) is used for model interpretability:
```bash
jupyter notebook notebooks/07_explainability_shap.ipynb
```

## 📝 License

This project is licensed under the MIT License.

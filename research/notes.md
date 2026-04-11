# DriftGuard Research Notes

Last Updated: 2026-01-22

## Project Overview

DriftGuard is an ensemble-based network intrusion detection system designed to address dataset shift challenges in cybersecurity ML applications. The system combines three complementary anomaly detection models (Isolation Forest, Autoencoder, and LSTM Autoencoder) to maintain robust performance across different network environments.

## Current Status

### ✅ Completed Components

1. **Core Models** (All trained and saved in `models/`)
   - Isolation Forest (`isolation_forest.pkl`)
   - Autoencoder (`autoencoder.keras`)
   - LSTM Autoencoder (`lstm_autoencoder.keras`)

2. **Datasets Processed**
   - **CICIDS2017**: Primary training dataset, 77 features, ~2M benign samples
   - **UNSW-NB15**: Cross-dataset validation, 49 native features → mapped to 77
   - **UGR16**: Real-world validation dataset, recently integrated

3. **Research Experiments** (4 completed)
   - Exp 1: Model comparison and performance benchmarking
   - Exp 2: False positive analysis using SHAP
   - Exp 3: Dataset shift analysis (KS tests, PSI calculations)
   - Exp 4: UGR16 real-world validation

4. **Production Code** (`src/` modules)
   - Data loading and preprocessing pipeline
   - Feature engineering transformations
   - Individual model training scripts
   - Ensemble risk scoring system
   - End-to-end inference module
   - Visualization utilities

5. **Documentation** (`docs/` - 18 files)
   - Comprehensive modular documentation covering all aspects
   - Index, architecture, datasets, methodology, models, experiments
   - Cross-dataset validation, explainability, deployment guides

## Datasets

### 1. CICIDS2017 (Training)
- **Source**: Canadian Institute for Cybersecurity
- **Purpose**: Primary training dataset
- **Features**: 77 network flow features
- **Samples**: ~2.8M total, ~2M benign used for training
- **Location**: `data/CICIDS2017/`
- **Files**: 8 CSV files covering 5 days of network traffic
- **Attack Types**: Brute Force, DoS, DDoS, Web attacks, Infiltration, Botnet

### 2. UNSW-NB15 (Cross-Dataset Validation)
- **Source**: University of New South Wales
- **Purpose**: Test generalization to different network environment
- **Features**: 49 native features (mapped to 77 via zero-padding)
- **Location**: `data/UNSW_NB15/`
- **Files**: Training and testing CSV files
- **Attack Types**: Exploits, Reconnaissance, Shellcode, Worms
- **Preprocessed Data**: `data/unsw/` (X_unsw.npy, y_unsw.npy, predictions, scores)

### 3. UGR16 (Real-World Validation)
- **Source**: University of Granada
- **Purpose**: Additional real-world validation
- **Location**: `data/UGR16/`
- **Files**: Multiple versions (v1, v2, v3, v4) with training/testing splits
- **Preprocessed Data**: `data/ugr_*.npy` files for predictions and scores
- **Status**: Recently integrated in Experiment 4

## Experiment Results

### Experiment 1: Model Comparison
- **Notebook**: `research/experiments/exp1_model_comparison.ipynb`
- **Goal**: Benchmark individual models vs ensemble
- **Results**: `research/results/tables/experiment1_model_comparison.csv`
- **Figure**: `research/results/figures/experiment1_model_comparison.png`
- **Key Finding**: Ensemble approach maintains ~15-20% higher accuracy on cross-dataset validation

### Experiment 2: False Positive Analysis
- **Notebook**: `research/experiments/exp2_false_positive_analysis.ipynb`
- **Goal**: Use SHAP to understand misclassifications
- **Results**: KS test results, SHAP waterfall plots
- **Figure**: `research/results/figures/experiment2_fp_shap_waterfall.png`
- **Key Finding**: Protocol-level features (ports, flags) more interpretable than derivative features

### Experiment 3: Dataset Shift Analysis
- **Notebook**: `research/experiments/exp3_dataset_shift_analysis.ipynb`
- **Goal**: Quantify distribution differences between CICIDS2017 and UNSW-NB15
- **Methods**: Kolmogorov-Smirnov tests, Population Stability Index
- **Results**: 
  - `research/results/tables/experiment3_ks_dataset_shift.csv`
  - `research/results/tables/experiment3_psi_results.csv`
- **Figures**: 10 distribution shift plots in `research/results/figures/shifts_exp/`
- **Key Finding**: PSI and KS tests can predict which features will cause performance degradation

### Experiment 4: UGR16 Real-World Validation
- **Notebook**: `research/experiments/exp4_ugr16_real_world_validation.ipynb`
- **Goal**: Validate ensemble on additional real-world dataset
- **Status**: Completed with UGR16 dataset integration
- **Data Generated**: `data/ugr_*.npy` files (predictions, scores, errors for all models)

## Model Architecture

### Ensemble Weights
- **Isolation Forest**: 30% weight
  - Fast, works well with high-dimensional data
  - Detects point anomalies effectively
  
- **Autoencoder**: 40% weight (highest)
  - Most reliable for network traffic
  - Captures complex non-linear patterns
  
- **LSTM Autoencoder**: 30% weight
  - Temporal dependency modeling
  - Detects sequence-based attacks (port scans)

### Risk Scoring
- All model scores normalized to [0, 1] range
- Weighted fusion: `risk = 0.3×IF + 0.4×AE + 0.3×LSTM`
- Severity levels:
  - HIGH: score ≥ 0.7 (immediate investigation)
  - MEDIUM: 0.4 ≤ score < 0.7 (review when time permits)
  - LOW: score < 0.4 (likely benign, log only)

## Key Insights

1. **Cross-Dataset Generalization**: Ensemble methods significantly outperform individual models when deployed across different network environments

2. **Feature Stability**: Basic protocol features (ports, flags, packet sizes) are more stable across datasets than derived features (bytes/sec, packets/sec)

3. **Explainability Value**: SHAP integration helps build trust with security analysts by providing clear feature-level explanations

4. **Dataset Shift Quantification**: Statistical tests (KS, PSI) effectively predict performance degradation in new environments

5. **Three-Dataset Validation**: Validation on UNSW-NB15 and UGR16 demonstrates robustness across multiple network environments

## Next Steps / Future Work

1. **Zeek Integration** (Linux deployment)
   - Real-time network traffic analysis
   - Live packet processing pipeline
   - Production streaming inference

2. **MITRE ATT&CK Auto-Mapping**
   - Automatic attack technique classification
   - Map detected anomalies to ATT&CK framework
   - Enhanced threat intelligence

3. **Dashboard Enhancement**
   - Streamlit visualization improvements
   - Real-time monitoring capabilities
   - Interactive SHAP explanations

4. **Model Optimization**
   - Hyperparameter tuning for cross-dataset performance
   - Adaptive ensemble weights based on environment
   - Incremental learning for concept drift

## File Locations

### Trained Models
- `models/isolation_forest.pkl` (1.8 MB)
- `models/autoencoder.keras` (200 KB)
- `models/lstm_autoencoder.keras` (936 KB)

### Preprocessed Data
- CICIDS2017: `data/X_phase2.npy`, `data/y_phase2.npy`
- UNSW-NB15: `data/unsw/X_unsw.npy`, `data/unsw/y_unsw.npy`
- UGR16: `data/ugr_*.npy` files

### Model Outputs (CICIDS2017)
- `data/iforest_scores.npy`
- `data/autoencoder_errors.npy`
- `data/lstm_sequence_errors.npy`
- `data/final_risk_scores.npy`
- `data/final_severity_labels.npy`
- `data/shap_values_high_risk.npy`

### Experiment Results
- Tables: `research/results/tables/*.csv`
- Figures: `research/results/figures/*.png`
- Shift Analysis: `research/results/figures/shifts_exp/*.png`

## Performance Metrics

- **Inference Speed**: ~2,380 samples/second
- **Latency**: 420ms per 1000 samples
- **Total Model Size**: 3 MB (all 3 models)
- **Training Time**: ~65 minutes on 2M+ samples
- **Memory Usage**: ~4 GB RAM during training

## References

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- UGR16: https://nesg.ugr.es/nesg-ugr16/
- SHAP: https://github.com/slundberg/shap

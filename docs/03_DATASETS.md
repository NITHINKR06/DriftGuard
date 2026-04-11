# Datasets

**[← Previous: Architecture](02_ARCHITECTURE.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Methodology →](04_METHODOLOGY.md)**

---

## CICIDS2017 (Training Dataset)

**Source**: Canadian Institute for Cybersecurity  
**Purpose**: Primary training dataset for model development

### Dataset Characteristics

- **Size**: ~2.8 million network flows
- **Collection Period**: Monday to Friday (5 days)
- **Features**: 78+ flow-based features
- **Class Distribution**: Highly imbalanced (majority benign traffic)

### Attack Types Included

| Attack Category | Examples | Description |
|----------------|----------|-------------|
| **DoS/DDoS** | Slowloris, GoldenEye, Hulk | Denial of Service attacks |
| **Brute Force** | FTP-Patator, SSH-Patator | Password guessing attacks |
| **Web Attacks** | XSS, SQL Injection | Application-layer exploits |
| **Infiltration** | Cool disk, Dropbox download | Insider threats |
| **Botnet** | Ares | Command and control traffic |
| **Port Scan** | Nmap, Nikto | Network reconnaissance |

### Feature Categories

1. **Basic Flow Features**
   - Flow Duration
   - Total Fwd Packets, Total Bwd Packets
   - Total Length of Fwd Packets, Total Length of Bwd Packets

2. **Packet Statistics**
   - Min, Max, Mean, Std of packet lengths
   - Fwd/Bwd packet length statistics

3. **Inter-Arrival Times**
   - Flow IAT (Inter-Arrival Time)
   - Fwd/Bwd IAT statistics

4. **TCP Flag Counts**
   - FIN, SYN, RST, PSH, ACK, URG flag counts
   - PSH Flag Count, URG Flag Count

5. **Flow Rates**
   - Flow Bytes/s, Flow Packets/s
   - Data rates in both directions

6. **Subflow Features**
   - Init_Win_bytes_forward
   - Init_Win_bytes_backward
   - Active Mean, Active Std, Active Min, Active Max
   - Idle Mean, Idle Std, Idle Min, Idle Max

### Preprocessing Steps

```python
# 1. Load raw CICIDS2017 CSV
df = pd.read_csv('CICIDS2017/traffic.csv')

# 2. Remove infinite values and NaNs
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 3. Handle class imbalance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train on benign traffic only (unsupervised)
X_benign = X_train_scaled[y_train == 0]
```

### Data Quality Issues

**Challenges**:
- ⚠️ Infinite values in flow rate features
- ⚠️ Missing values in some traffic samples
- ⚠️ Extreme outliers in packet counts
- ⚠️ Class imbalance (95% benign, 5% attack)

**Solutions**:
- Replace infinites with NaN, then drop
- Robust scaling for outliers
- Train unsupervised (benign only)
- Stratified sampling for evaluation

---

## UNSW-NB15 (Validation Dataset)

**Source**: University of New South Wales  
**Purpose**: Cross-dataset validation to test generalization

### Dataset Characteristics

- **Size**: ~2.5 million network flows
- **Collection Period**: Different network environment than CICIDS2017
- **Features**: 49 features (subset overlap with CICIDS2017)
- **Purpose**: Test zero-shot transfer learning

### Attack Types Included

| Attack Category | Description | Severity |
|----------------|-------------|----------|
| **Fuzzers** | Automated vulnerability scanners | Medium |
| **Analysis** | Port scanning, OS fingerprinting | Low |
| **Backdoors** | Unauthorized access mechanisms | Critical |
| **DoS** | Service disruption attacks | High |
| **Exploits** | Software vulnerability exploitation | High |
| **Generic** | Polymorphic attacks | Medium |
| **Reconnaissance** | Information gathering | Low |
| **Shellcode** | Code injection attacks | Critical |
| **Worms** | Self-replicating malware | Critical |

### Key Differences from CICIDS2017

| Aspect | CI CIDS2017 | UNSW-NB15 |
|--------|-------------|-----------|
| **Network Type** | University/Academic | Research testbed |
| **Traffic Patterns** | 8am-5pm, student behavior | 24/7 simulated enterprise |
| **Feature Count** | 78 features | 49 features |
| **Attack Taxonomy** | Web-focused (XSS, SQLi) | Broader (Fuzzers, Worms) |
| **Capture Tool** | CICFlowMeter | Argus, Bro |
| **Distribution** | Different mean/std | Different distribution shape |

### Feature Alignment Challenge

**Problem**: CICIDS2017 has 77 features, UNSW-NB15 has 49 features

**Solution**: Feature Mapping

```python
# CICIDS2017 canonical feature set (77 features)
cicids_features = X_cicids.columns

# UNSW-NB15 features (49 features)
unsw_features = X_unsw.columns

# Find common features
common_features = set(cicids_features) & set(unsw_features)
# Result: ~40 features overlap

# Create aligned feature matrix
X_unsw_aligned = pd.DataFrame(
    np.zeros((len(X_unsw), len(cicids_features))),
    columns=cicids_features
)

# Map available features
for feature in common_features:
    X_unsw_aligned[feature] = X_unsw[feature]

# Missing features remain as zeros (zero-padding)
# Extra UNSW features are ignored
```

### Normalization Strategy

**Critical**: Use CICIDS2017 statistics for normalization

```python
# Load scaler fitted on CICIDS2017
scaler = joblib.load('models/scaler_cicids.pkl')

# Transform UNSW-NB15 using CICIDS2017 mean/std
X_unsw_scaled = scaler.transform(X_unsw_aligned)
```

**Rationale**: Prevents data leakage, simulates production deployment

---

## UGR16 (Real-World Validation Dataset)

**Source**: University of Granada  
**Purpose**: Additional real-world validation for testing ensemble robustness

### Dataset Characteristics

- **Size**: Real ISP network traffic
- **Collection Period**: Long-term ISP network monitoring
- **Features**: Network flow features similar to CICIDS2017
- **Purpose**: Validate cross-dataset generalization on third independent dataset

### Dataset Versions

UGR16 consists of multiple versions with different configurations:

| Version | Description | Purpose |
|---------|-------------|----------|
| **UGR16v1** | Base version | Initial validation |
| **UGR16v2** | Enhanced version | Extended attacks |
| **UGR16v2noIRC** | No IRC traffic | Clean subset |
| **UGR16v3** | Additional attacks | Robustness testing |
| **UGR16v3v4** | Combined versions | Comprehensive validation |
| **UGR16v4** | Latest version | Most recent attacks |

### Integration with DriftGuard

**Experiment 4**: `research/experiments/exp4_ugr16_real_world_validation.ipynb`

```python
# Load UGR16 data
X_ugr = pd.read_csv('data/UGR16/UGR16v1.Xtest.csv')
y_ugr = pd.read_csv('data/UGR16/UGR16v1.Ytest.csv')

# Feature alignment (similar process to UNSW-NB15)
X_ugr_aligned = align_features(X_ugr, reference=X_cicids.columns)

# Apply CICIDS scaler
X_ugr_scaled = scaler.transform(X_ugr_aligned)

# Make predictions with all three models
ugr_predictions = ensemble_predict(X_ugr_scaled)
```

### Preprocessed Data Outputs

Stored in `data/`:
- `ugr_predictions.npy` - Ensemble predictions
- `ugr_risk_scores.npy` - Final risk scores
- `ugr_iforest_scores.npy` - Isolation Forest anomaly scores
- `ugr_ae_errors.npy` - Autoencoder reconstruction errors
- `ugr_lstm_errors.npy` - LSTM sequence prediction errors

### Research Value

**Three-Dataset Validation**:
1. **CICIDS2017** → Training
2. **UNSW-NB15** → Cross-dataset validation
3. **UGR16** → Real-world validation

This comprehensive validation strategy demonstrates that the ensemble approach generalizes well across:
- Different network environments
- Different attack taxonomies
- Real-world ISP production traffic

---

## Data Storage Structure

### Directory Organization

```
data/
├── CICIDS2017/              # Raw CICIDS2017 CSV files
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-workingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│
├── UNSW_NB15/               # Raw UNSW-NB15 CSV files
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   └── UNSW-NB15_4.csv
│
├── UGR16/                   # Raw UGR16 CSV files
│   ├── UGR16v1.Xtest.csv / UGR16v1.Ytest.csv
│   ├── UGR16v2.Xtest.csv / UGR16v2.Ytest.csv
│   ├── UGR16v3.Xtest.csv / UGR16v3.Ytest.csv
│   └── UGR16v4.Xtest.csv / UGR16v4.Ytest.csv
│
├── unsw/                    # UNSW preprocessed data
│
├── processed_phase1.csv     # Preprocessed CICIDS2017
├── X_phase2.npy             # Feature matrix (77 features)
├── y_phase2.npy             # Labels (0=benign, 1=attack)
│
├── iforest_scores.npy       # Isolation Forest outputs
├── autoencoder_errors.npy   # Autoencoder reconstruction errors
├── lstm_sequence_errors.npy # LSTM prediction errors
├── ugr_predictions.npy      # UGR16 ensemble predictions
├── ugr_risk_scores.npy      # UGR16 risk scores
├── ugr_iforest_scores.npy   # UGR16 Isolation Forest outputs
├── ugr_ae_errors.npy        # UGR16 Autoencoder errors
├── ugr_lstm_errors.npy      # UGR16 LSTM errors
├── final_risk_scores.npy    # Ensemble risk scores
└── final_severity_labels.npy # HIGH/MEDIUM/LOW classifications
```

### File Sizes

| File | Size | Format | Description |
|------|------|--------|-------------|
| CICIDS2017 CSVs | ~2.5 GB | CSV | Raw traffic (Git LFS) |
| UNSW-NB15 CSVs | ~2.0 GB | CSV | Raw traffic (Git LFS) |
| X_phase2.npy | ~250 MB | NumPy | Preprocessed features |
| y_phase2.npy | ~8 MB | NumPy | Labels |
| Model outputs | ~50 MB each | NumPy | Predictions/scores |

**Note**: Large files tracked with Git LFS

---

## Data Access & Loading

### Quick Data Loading

```python
from src.data_loader import (
    load_raw_cicids_data,
    load_phase2_features,
    load_all_model_scores,
    load_final_outputs
)

# Load raw CICIDS2017
df_raw = load_raw_cicids_data('Monday-WorkingHours.pcap_ISCX.csv')

# Load preprocessed features
X, y = load_phase2_features()

# Load all model outputs
scores = load_all_model_scores()  
# Returns: {'iforest': ..., 'autoencoder': ..., 'lstm': ...}

# Load final risk scores
risk_scores, labels = load_final_outputs()
```

---

## Dataset Citations

### CICIDS2017

```
Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).
Toward generating a new intrusion detection dataset and intrusion traffic characterization.
ICISSp, 108-116.
```

**Download**: https://www.unb.ca/cic/datasets/ids-2017.html

### UNSW-NB15

```
Moustafa, N., & Slay, J. (2015).
UNSW-NB15: a comprehensive data set for network intrusion detection systems.
Military Communications and Information Systems Conference (MilCIS), 1-6.
```

**Download**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

### UGR16

```
Macia-Fernandez, G., Camacho, J., Magán-Carrión, R., García-Teodoro, P., & Theron, R. (2018).
UGR '16: A new dataset for the evaluation of cyclostationarity-based network IDSs.
Computers & Security, 73, 411-424.
```

**Download**: https://nesg.ugr.es/nesg-ugr16/

---

**Next: [Methodology →](04_METHODOLOGY.md)**

**[← Previous: Architecture](02_ARCHITECTURE.md) | [Back to Index](00_INDEX.md)**

# Future Work & References

**[← Previous: Version Control](13_VERSION_CONTROL.md)** | **[Back to Index](00_INDEX.md)**

---

## Planned Enhancements

##

 1. Real-Time Inference with Zeek

> [!NOTE]
> **Status**: Planned for Linux deployment

**Objective**: Integrate with Zeek network security monitor for live traffic analysis

### Architecture

```
Network Traffic → Zeek → Feature Extraction → DriftGuard Models → Risk Scoring → SIEM
```

### Components

- **Zeek Log Parser**: Parse Zeek conn.log, dns.log, http.log
- **Real-Time Feature Engineering**: Extract 77 features from live Zeek logs
- **Streaming Prediction Pipeline**: Continuous inference on incoming traffic
- **Alert Generation**: Create alerts for SIEM integration

### Benefits

✅ **Immediate Threat Detection**: Sub-second detection latency  
✅ **Production-Ready**: Zeek widely deployed in enterprises  
✅ **Integration**: Compatible with existing security infrastructure  
✅ **Scalability**: Handles high-throughput networks  

### Implementation Plan

```python
# Zeek integration pseudocode
from watchdog import Observer

class ZeekLogHandler:
    def on_modified(self, event):
        # Parse new Zeek log entries
        logs = parse_zeek_log(event.src_path)
        
        # Extract features
        features = extract_features(logs)
        
        # Run DriftGuard inference
        risk_scores = detector.predict(features)
        
        # Generate alerts
        for score, log in zip(risk_scores, logs):
            if score > 0.7:
                send_alert_to_siem(log, score)
```

---

## 2. MITRE ATT&CK Auto-Mapping

> [!NOTE]
> **Status**: Research phase

**Objective**: Automatically map detected anomalies to MITRE ATT&CK framework

### Methodology

1. **Cluster Anomalies**: Group similar attacks based on feature patterns
2. **Map to Tactics**: Reconnaissance, Execution, Persistence, etc.
3. **Identify Techniques**: T1001 (Data Obfuscation), T1046 (Network Service Discovery), etc.
4. **Generate Threat Intel**: Create ATT&CK Navigator visualizations

### Use Cases

- **Incident Response Prioritization**: Understand attack kill chain
- **Threat Hunting**: Search for specific TTPs
- **Security Posture Assessment**: Identify gaps in defenses
- **Compliance Reporting**: Map to compliance frameworks (NIST, etc.)

### Example Mapping

```python
# Pseudocode for ATT&CK mapping
attack_signature = {
    "high_port_scan_rate": "T1046",  # Network Service Discovery
    "unusual_outbound_traffic": "T1041",  # Exfiltration Over C2 Channel
    "lateral_movement_patterns": "T1021"  # Remote Services
}

def map_to_attack(features, risk_score):
    techniques = []
    if features['Flow Packets/s'] > threshold:
        techniques.append("T1046")  # Port Scanning
    if features['Bwd Packet Length Mean'] > threshold:
        techniques.append("T1041")  # Data Exfiltration
    return techniques
```

---

## 3. Advanced Explainability

### Counterfactual Explanations

**Question**: *"What would need to change for this traffic to be classified as benign?"*

```python
# Example counterfactual
original_sample = X[high_risk_idx]
counterfactual = find_nearest_benign(original_sample)

changes = counterfactual - original_sample
print(f"To be benign, reduce Flow Packets/s by {changes['Flow Packets/s']}")
```

### Global Surrogate Models

Train simple decision tree to mimic ensemble:

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Train surrogate
surrogate = DecisionTreeClassifier(max_depth=5)
surrogate.fit(X_test, ensemble_predictions)

# Explain with simple rules
rules = export_text(surrogate, feature_names=features)
print(rules)
```

###Concept Activation Vectors (CAVs)

Understand high-level attack concepts learned by models

---

## 4. Federated Learning

**Objective**: Train models across multiple organizations without sharing raw data

### Benefits

✅ **Privacy-Preserving**: Organizations keep data local  
✅ **Improved Generalization**: Learn from diverse networksenvironments  
✅ **Diverse Attack Coverage**: Encounter more attack types  

### Architecture

```
Organization A → Local Training → Model Updates ↘
Organization B → Local Training → Model Updates → Aggregation Server → Global Model
Organization C → Local Training → Model Updates ↗
```

---

## 5. Adversarial Robustness

### Research Questions

1. **How robust are models to adversarial attacks?**
2. **Can attackers craft traffic to evade detection?**
3. **What defense mechanisms work against adversarial examples?**

### Adversarial Attack Examples

```python
# Example: FGSM attack on network traffic
import numpy as np

def fgsm_attack(model, X, epsilon=0.1):
    # Compute gradients
    gradients = model.gradients(X)
    
    # Perturb features
    X_adv = X + epsilon * np.sign(gradients)
    
    return X_adv
```

---

## Research Extensions

### 1. Multi-Dataset Training

**Idea**: Train on CICIDS2017 + UNSW-NB15 jointly

**Benefits**:
- Better generalization
- Learn dataset-invariant features
- Improved robustness

### 2. Transfer Learning

**Idea**: Fine-tune models for new environments with minimal data

```python
# Load pre-trained model
base_model = load_model('models/autoencoder.keras')

# Freeze early layers
for layer in base_model.layers[:3]:
    layer.trainable = False

# Fine-tune on new data
base_model.fit(X_new_environment, epochs=5)
```

### 3. Active Learning

**Idea**: Iteratively improve models with analyst feedback

```python
# Analyst labels uncertain samples
uncertain_idx = np.where((risk_scores > 0.4) & (risk_scores < 0.6))
analyst_labels = get_analyst_feedback(X[uncertain_idx])

# Retrain with new labels
model.fit(X[uncertain_idx], analyst_labels)
```

### 4. Graph Neural Networks

**Idea**: Model network topology for context-aware detection

```python
# Represent network as graph
import networkx as nx

G = nx.Graph()
G.add_edges_from([(src_ip, dst_ip, {'features': flow_features})])

# GNN for anomaly detection
gnn_model.predict(G)
```

### 5. Time-Series Forecasting

**Idea**: Predict future attacks based on trends

```python
# LSTM for attack forecasting
forecast_model = LSTMFor ecaster()
future_attacks = forecast_model.predict(historical_data)
```

---

## Academic References

### Datasets

1. **CICIDS2017**  
   Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSp*, 108-116.  
   URL: https://www.unb.ca/cic/datasets/ids-2017.html

2. **UNSW-NB15**  
   Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*, 1-6.  
   URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset

### Algorithms & Methods

3. **Isolation Forest**  
   Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*, 413-422.

4. **Autoencoders for Anomaly Detection**  
   Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction. *MLSDA Workshop*, 4-11.

5. **LSTM Networks**  
   Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

6. **SHAP**  
   Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4765-4774.

### Related Work

7. **Dataset Shift in ML**  
   Quionero-Candela, J., et al. (2009). Dataset shift in machine learning. *MIT Press*.

8. **Ensemble Methods for IDS**  
   Folino, G., et al. (2016). An ensemble-based evolutionary framework for coping with distributed intrusion detection. *Genetic Programming and Evolvable Machines*, 17(1), 61-88.

9. **Cross-Dataset Validation**  
   Ring, M., et al. (2019). A survey of network-based intrusion detection data sets. *Computers & Security*, 86, 147-167.

### Frameworks & Libraries

10. **Scikit-learn**  
    Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.

11. **TensorFlow**  
    Abadi, M., et al. (2016). Tensorflow: A system for large-scale machine learning. *OSDI*, 16, 265-283.

12. **SHAP Library**  
    https://github.com/slundberg/shap

---

## Acknowledgments

This research was conducted as part of a cybersecurity machine learning project exploring practical challenges in deploying intrusion detection systems across diverse network environments.

**Special Thanks**:
- Canadian Institute for Cybersecurity for CICIDS2017 dataset
- UNSW Canberra for UNSW-NB15 dataset
- Open-source community for tools and libraries

---

## Contact & Collaboration

**Repository**: https://github.com/NITHINKR06/DriftGuard  
**Issues**: Report bugs and feature requests via GitHub Issues  
**Contributions**: Pull requests welcome!

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Document Metadata

- **Documentation Version**: 2.0 (Modular Structure)
- **Last Updated**: January 2026
- **Authors**: Research Team, DriftGuard Project
- **Total Documentation**: 14 comprehensive modular files

---

**[← Previous: Version Control](13_VERSION_CONTROL.md) | [Back to Index](00_INDEX.md)**

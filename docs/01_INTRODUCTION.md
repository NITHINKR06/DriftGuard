# Introduction & Problem Statement

**[← Back to Index](00_INDEX.md)** | **[Next: Architecture →](02_ARCHITECTURE.md)**

---

## Executive Summary

**DriftGuard** is an advanced ensemble-based network intrusion detection system (NIDS) designed to address the critical challenge of **dataset shift** in cybersecurity machine learning applications. The system combines three complementary anomaly detection approaches:

- **Isolation Forest** - Tree-based anomaly detection
- **Autoencoder** - Neural network reconstruction-based detection
- **LSTM Autoencoder** - Temporal sequence-based detection

### Key Contributions

1. **Cross-Dataset Generalization**: Trained on CICIDS2017, validated on UNSW-NB15
2. **Ensemble Risk Scoring**: Weighted fusion of multiple model outputs
3. **Explainable AI**: SHAP-based interpretability for security analysts
4. **Distribution Shift Analysis**: Quantitative analysis of dataset drift using KS tests and PSI
5. **Production-Ready Architecture**: Modular, scalable, and maintainable codebase

### Research Impact

This research demonstrates that ensemble methods can achieve robust cross-dataset performance despite significant distribution shifts between training and deployment environments - a critical requirement for real-world cybersecurity applications.

---

## Research Problem & Motivation

### The Challenge

Network intrusion detection systems face a fundamental problem: **models trained on one dataset often fail when deployed in different network environments**. This happens because:

1. **Dataset Shift**: Network traffic characteristics vary across organizations, time periods, and attack landscapes
2. **Evolving Threats**: New attack patterns emerge that weren't present in training data
3. **Configuration Differences**: Different network architectures produce different traffic patterns
4. **Temporal Drift**: Network behavior changes over time

### Why This Matters

- **False Negatives**: Undetected attacks can lead to security breaches
- **False Positives**: Alert fatigue reduces analyst effectiveness
- **Resource Waste**: Retraining models for each new environment is expensive
- **Trust Issues**: Security teams need to understand *why* a model flags traffic as malicious

### The Dataset Shift Problem in Cybersecurity

Traditional machine learning assumes:
- Training data and deployment data come from the same distribution
- Model performance on test set predicts real-world performance

**Reality in cybersecurity**:
- Network environments are heterogeneous
- Attack techniques constantly evolve
- Each organization has unique traffic patterns

**Consequences**:
- Models with 95% accuracy in lab → 60% accuracy in production
- High false positive rates (thousands per day)
- Security teams lose trust in automated systems
- Manual investigation of every alert is infeasible

### Real-World Example

A model trained on **University Network Traffic** (CICIDS2017):
- Learns patterns: 8am-5pm peak usage, academic applications, student behavior
- Deploys to **Corporate Network** (UNSW-NB15)
- Encounters: 24/7 operations, different applications, varied user profiles
- **Result**: Flags normal corporate behavior as anomalous

---

## Research Questions

This research investigates four critical questions:

### 1. Cross-Dataset Generalization
**How well do anomaly detection models generalize across different network datasets?**

- Can a model trained on CICIDS2017 detect attacks in UNSW-NB15?
- What performance degradation occurs during cross-dataset transfer?
- Which model types (tree-based, neural network, sequence) generalize best?

### 2. Ensemble Robustness
**Can ensemble methods improve robustness to dataset shift?**

- Does combining multiple detection approaches reduce brittleness?
- How should individual model outputs be weighted for fusion?
- Can ensemble diversity compensate for distribution changes?

### 3. Distribution Analysis
**What types of distribution changes cause the most significant performance degradation?**

- Which network traffic features show the largest shifts?
- How can we quantify dataset shift (KS tests, PSI)?
- Can we predict model performance based on shift magnitude?

### 4. Interpretability
**How can we make intrusion detection models interpretable for security analysts?**

- Why did the model flag this specific traffic as malicious?
- Which features contribute most to anomaly detection?
- Can explanations help analysts trust automated systems?

---

## Research Hypotheses

### H1: Ensemble Superiority
**Hypothesis**: Ensemble methods will outperform individual models in cross-dataset scenarios due to complementary detection mechanisms.

**Reasoning**: Different models capture different anomaly patterns:
- Isolation Forest → Isolation-based outliers
- Autoencoder → Reconstruction errors
- LSTM → Temporal sequence anomalies

### H2: Gradual Performance Degradation
**Hypothesis**: Performance degradation will correlate with the magnitude of distribution shifts in key features.

**Testing**: Compare KS/PSI statistics with model accuracy drops

### H3: Feature Stability
**Hypothesis**: Protocol-level features (port numbers, flags) will be more stable across datasets than derived features (bytes/s, packets/s).

**Testing**: Measure PSI for different feature categories

---

## Contributions to the Field

### Academic Contributions

1. **Quantitative Cross-Dataset Analysis**
   - First comprehensive KS/PSI analysis between CICIDS2017 and UNSW-NB15
   - Identifies which features are stable vs. shifting

2. **Ensemble Architecture for Dataset Shift**
   - Novel weighted fusion approach
   - Demonstrates robustness compared to individual models

3. **Explainability Framework**
   - SHAP integration for network intrusion detection
   - Bridges gap between ML models and security analysts

### Practical Contributions

1. **Production-Ready System**
   - Modular codebase (data_loader, inference, risk_scoring)
   - Scalable architecture

2. **Interactive Dashboard**
   - Streamlit-based visualization
   - Real-time monitoring for SOC teams

3. **Open-Source Implementation**
   - Complete code on GitHub
   - Reproducible experiments

---

## Scope & Limitations

### In Scope

✅ Anomaly-based detection (training on benign traffic)  
✅ Binary classification (normal vs. attack)  
✅ Offline batch processing  
✅ Cross-dataset validation  
✅ Model interpretability  

### Out of Scope

❌ Real-time packet processing  
❌ Multi-class attack categorization  
❌ Adversarial robustness testing  
❌ Zero-day attack detection  
❌ Network-wide deployment architecture  

### Limitations

1. **Datasets**: Only CICIDS2017 and UNSW-NB15 tested
2. **Attacks**: Limited to attacks present in datasets
3. **Deployment**: Not tested in live production network
4. **Scalability**: Benchmarked on single machine, not distributed systems

---

## Document Roadmap

**What You'll Learn in This Documentation**:

1. **[Architecture](02_ARCHITECTURE.md)**: How DriftGuard is designed
2. **[Datasets](03_DATASETS.md)**: What data we used and how it's processed
3. **[Methodology](04_METHODOLOGY.md)**: Step-by-step approach to building the system
4. **[Implementation](05_IMPLEMENTATION.md)**: Code structure and key modules
5. **[Models](06_MODELS.md)**: Detailed model architectures and training
6. **[Experiments](07_EXPERIMENTS.md)**: Actual experimental results with real metrics
7. **[Cross-Dataset Validation](08_CROSS_DATASET_VALIDATION.md)**: How models perform on UNSW-NB15
8. **[Explainability](09_EXPLAINABILITY.md)**: Understanding model decisions with SHAP
9. **[Visualization](10_VISUALIZATION_DASHBOARD.md)**: Dashboard and plotting tools
10. **[Deployment](11_DEPLOYMENT.md)**: How to use the system

---

**Next: [System Architecture →](02_ARCHITECTURE.md)**

**[← Back to Index](00_INDEX.md)**

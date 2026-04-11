# DriftGuard Research Documentation Index

**A Comprehensive Guide to Cross-Dataset Network Intrusion Detection with Ensemble Machine Learning**

---

## 📚 Documentation Structure

This research documentation is organized into modular files for easy navigation:

### Core Research Documents

1. **[Introduction & Problem Statement](01_INTRODUCTION.md)**
   - Executive Summary
   - Research Problem & Motivation
   - Research Questions
   - Key Contributions
   - Research Impact

2. **[System Architecture](02_ARCHITECTURE.md)**
   - Architecture Overview
   - Component Breakdown
   - System Flow Diagrams
   - Technology Stack

3. **[Datasets](03_DATASETS.md)**
   - CICIDS2017 (Training Dataset)
   - UNSW-NB15 (Validation Dataset)
   - Data Preprocessing
   - Feature Descriptions
   - Data Storage Structure

4. **[Methodology](04_METHODOLOGY.md)**
   - Phase 1: Data Exploration & Feature Engineering
   - Phase 2: Model Training
   - Phase 3: Ensemble Risk Scoring
   - Phase 4: Model Evaluation
   - Training Strategies

5. **[Implementation Details](05_IMPLEMENTATION.md)**
   - Project Structure
   - Source Code Modules
   - Data Loader
   - Risk Scoring System
   - Inference Pipeline
   - Feature Engineering (FeaturePipeline)

6. **[Model Architectures](06_MODELS.md)**
   - Isolation Forest
   - Autoencoder Architecture
   - LSTM Autoencoder Architecture
   - Hyperparameter Tuning
   - Training Code Examples

### Experimental Results

7. **[Experiments & Results](07_EXPERIMENTS.md)**
   - Experiment 1: Model Comparison (Actual Metrics)
   - Experiment 2: False Positive Analysis
   - Experiment 3: Dataset Shift Analysis (KS & PSI)
   - Performance Benchmarks

8. **[Cross-Dataset Validation](08_CROSS_DATASET_VALIDATION.md)**
   - Feature Alignment Strategy
   - Performance Degradation Analysis
   - Attack Type Detection Rates
   - Generalization Findings

9. **[Model Explainability](09_EXPLAINABILITY.md)**
   - SHAP Analysis
   - Feature Importance
   - Waterfall Plots
   - Top Features for Detection
   - Interpretability for Security Analysts

### Visualization & Deployment

10. **[Visualization & Dashboard](10_VISUALIZATION_DASHBOARD.md)**
    - Streamlit Dashboard
    - Visualization Utilities
    - Plot Saving Tools
    - Generated Figures

11. **[Deployment Guide](11_DEPLOYMENT.md)**
    - Installation Instructions
    - Training Pipeline
    - Inference Usage
    - Production Deployment
    - Integration Points

### Technical References

12. **[Technical Appendix](12_TECHNICAL_APPENDIX.md)**
    - Hardware & Software Requirements
    - Complete Hyperparameters
    - Code Architectures
    - Performance Benchmarks
    - Troubleshooting Common Issues

13. **[Version Control & Configuration](13_VERSION_CONTROL.md)**
    - Git LFS Setup
    - Git Ignore Configuration
    - Project Dependencies
    - Model File Sizes
    - Environment Setup

14. **[Future Work & References](14_FUTURE_REFERENCES.md)**
    - Planned Enhancements
    - Research Extensions
    - Academic References
    - Citations

---

## 🚀 Quick Navigation

### For Research Papers
- [Introduction](01_INTRODUCTION.md) → [Methodology](04_METHODOLOGY.md) → [Experiments](07_EXPERIMENTS.md) → [Results Discussion](08_CROSS_DATASET_VALIDATION.md)

### For Implementation
- [Architecture](02_ARCHITECTURE.md) → [Implementation](05_IMPLEMENTATION.md) → [Models](06_MODELS.md) → [Deployment](11_DEPLOYMENT.md)

### For Understanding Results
- [Experiments](07_EXPERIMENTS.md) → [Cross-Dataset Validation](08_CROSS_DATASET_VALIDATION.md) → [Explainability](09_EXPLAINABILITY.md)

### For Deployment
- [Deployment Guide](11_DEPLOYMENT.md) → [Version Control](13_VERSION_CONTROL.md) → [Technical Appendix](12_TECHNICAL_APPENDIX.md)

---

## 📊 Project Overview

**DriftGuard** is an advanced ensemble-based network intrusion detection system that:
- Trains on CICIDS2017 dataset
- Validates on UNSW-NB15 dataset (cross-dataset)
- Combines 3 models: Isolation Forest, Autoencoder, LSTM
- Uses ensemble risk scoring with interpretable SHAP analysis
- Analyzes dataset shift using KS tests and PSI metrics

---

## 📂 Project Repository

- **GitHub**: https://github.com/NITHINKR06/DriftGuard
- **Project Path**: `e:\3rd_year\6th-sem\await`
- **Documentation**: `docs/` directory

---

## 📝 Document Metadata

- **Version**: 2.0 (Modular Structure)
- **Last Updated**: January 2026
- **Authors**: Research Team, DriftGuard Project
- **Total Documentation Pages**: 14 comprehensive documents

---

## 💡 How to Use This Documentation

1. **Start with [Introduction](01_INTRODUCTION.md)** to understand the research problem
2. **Review [Architecture](02_ARCHITECTURE.md)** for system design overview
3. **Read [Methodology](04_METHODOLOGY.md)** for detailed approach
4. **Explore [Experiments](07_EXPERIMENTS.md)** for actual results
5. **Check [Deployment](11_DEPLOYMENT.md)** for implementation guide

Each document is self-contained but cross-references related topics.

---

**Ready to dive in? Start with [01_INTRODUCTION.md](01_INTRODUCTION.md) →**

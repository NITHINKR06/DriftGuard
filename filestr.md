# DriftGuard - Project File Structure

## Root Level Files
```
.gitattributes
.gitignore
README.md
filestr.md
index.html
requirements.txt
```

## Dashboard
```
dashboard/
├── app.py
```

## Data Directory
```
data/
├── CICIDS2017/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-workingHours.pcap_ISCX.csv
│
├── UGR16/
│   ├── UGR16v1.Xtest.csv
│   ├── UGR16v1.Xtrain.csv
│   ├── UGR16v1.Ytest.csv
│   ├── UGR16v1.Ytrain.csv
│   ├── UGR16v2.Xtest.csv
│   ├── UGR16v2.Xtrain.csv
│   ├── UGR16v2.Ytest.csv
│   ├── UGR16v2.Ytrain.csv
│   ├── UGR16v2noIRC.Xtest.csv
│   ├── UGR16v2noIRC.Xtrain.csv
│   ├── UGR16v2noIRC.Ytest.csv
│   ├── UGR16v2noIRC.Ytrain.csv
│   ├── UGR16v3.Xtest.csv
│   ├── UGR16v3.Xtrain.csv
│   ├── UGR16v3.Ytest.csv
│   ├── UGR16v3.Ytrain.csv
│   ├── UGR16v3v4.Xtest.csv
│   ├── UGR16v3v4.Xtrain.csv
│   ├── UGR16v3v4.Ytest.csv
│   ├── UGR16v3v4.Ytrain.csv
│   ├── UGR16v4.Xtest.csv
│   ├── UGR16v4.Xtrain.csv
│   ├── UGR16v4.Ytest.csv
│   └── UGR16v4.Ytrain.csv
│
├── UNSW_NB15/
│   ├── UNSW_NB15_testing-set.csv
│   └── UNSW_NB15_training-set.csv
│
├── unsw/
│   ├── unsw_ae_errors.npy
│   ├── unsw_iforest_scores.npy
│   ├── unsw_labels.npy
│   ├── unsw_lstm_errors.npy
│   ├── unsw_predictions.npy
│   ├── unsw_risk_scores.npy
│   ├── X_unsw.npy
│   └── y_unsw.npy
│
├── autoencoder_errors.npy
├── final_risk_scores.npy
├── final_severity_labels.npy
├── iforest_scores.npy
├── lstm_sequence_errors.npy
├── processed_phase1.csv
├── shap_values_high_risk.npy
├── ugr_ae_errors.npy
├── ugr_iforest_scores.npy
├── ugr_lstm_errors.npy
├── ugr_predictions.npy
├── ugr_risk_scores.npy
├── X_phase2.npy
└── y_phase2.npy
```

## Documentation
```
docs/
├── 00_INDEX.md
├── 01_INTRODUCTION.md
├── 02_ARCHITECTURE.md
├── 03_DATASETS.md
├── 04_METHODOLOGY.md
├── 05_IMPLEMENTATION.md
├── 06_MODELS.md
├── 07_EXPERIMENTS.md
├── 08_CROSS_DATASET_VALIDATION.md
├── 09_EXPLAINABILITY.md
├── 10_VISUALIZATION_DASHBOARD.md
├── 11_DEPLOYMENT.md
├── 12_TECHNICAL_APPENDIX.md
├── 13_VERSION_CONTROL.md
├── 14_FUTURE_REFERENCES.md
├── PROJECT_OVERVIEW.md
├── README.md
└── RESEARCH_DOCUMENTATION.md
```

## Models
```
models/
├── autoencoder.keras
├── isolation_forest.pkl
└── lstm_autoencoder.keras
```

## Notebooks
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_engineering.ipynb
├── 03_isolation_forest.ipynb
├── 04_autoencoder.ipynb
├── 05_lstm_sequence_model.ipynb
├── 06_model_evaluation.ipynb
├── 07_explainability_shap.ipynb
└── 09_cross_dataset_validation.ipynb
```

## Reports
```
reports/
└── figures/
    ├── phase1/
    │   └── label_distribution.png
    ├── phase2/
    ├── phase3/
    │   └── iforest_anomaly_scores.png
    ├── phase4/
    │   └── autoencoder_errors.png
    ├── phase5/
    │   └── lstm_sequence_errors.png
    ├── phase6/
    │   ├── final_risk_score.png
    │   └── risk_score_distribution.png
    └── phase7/
        ├── global_explanation.png
        ├── shap_summary.png
        └── shap_waterfall_high_risk.png
```

## Research
```
research/
├── notes.md
│
├── experiments/
│   ├── exp1_model_comparison.ipynb
│   ├── exp2_false_positive_analysis.ipynb
│   ├── exp3_dataset_shift_analysis.ipynb
│   └── exp4_ugr16_real_world_validation.ipynb
│
└── results/
    ├── figures/
    │   ├── experiment1_model_comparison.png
    │   ├── experiment2_fp_shap_waterfall.png
    │   └── shifts_exp/
    │       ├── experiment3_feature_shift_0.png
    │       ├── experiment3_feature_shift_5.png
    │       ├── experiment3_feature_shift_19.png
    │       ├── experiment3_feature_shift_28.png
    │       ├── experiment3_feature_shift_29.png
    │       ├── experiment3_feature_shift_34.png
    │       ├── experiment3_feature_shift_37.png
    │       ├── experiment3_feature_shift_43.png
    │       ├── experiment3_feature_shift_45.png
    │       └── experiment3_feature_shift_50.png
    │
    └── tables/
        ├── experiment1_model_comparison.csv
        ├── experiment2_ks_test_results.csv
        ├── experiment3_ks_dataset_shift.csv
        └── experiment3_psi_results.csv
```

## Source Code
```
src/
├── data_loader.py
├── features.py
├── fix_plot_saving.py
├── inference.py
├── plot_utils.py
├── risk_scoring.py
├── temporal_split.py
├── train_autoencoder.py
├── train_iforest.py
└── train_lstm.py
```

## Results Directory
```
results/
└── (empty or additional results)
```

---

**Note:** 
- Excluded directories: `.venv`, `__pycache__`, `.git`
- Excluded files: `*.pyc`
- Last updated: 2026-01-22
- Now includes UGR16 dataset and experiment 4 validation data

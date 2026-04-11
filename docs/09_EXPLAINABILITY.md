# Model Explainability

**[← Previous: Cross-Dataset Validation](08_CROSS_DATASET_VALIDATION.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Visualization & Dashboard →](10_VISUALIZATION_DASHBOARD.md)**

---

## Why Explainability Matters in Cybersecurity

### The Trust Gap

Security analysts face a critical challenge: **How can we trust a black-box ML model to make security decisions?**

**Problems with Opaque Models**:
- ❌ Analysts can't verify if detections make sense
- ❌ False positives waste time without understanding why
- ❌ Unable to learn new attack patterns from model behavior
- ❌ Regulatory compliance requires explainable decisions

**Benefits of Explainability**:
- ✅ **Trust**: Analysts understand why traffic was flagged
- ✅ **Debugging**: Identify model weaknesses and biases
- ✅ **Learning**: Discover new attack signatures
- ✅ **Compliance**: Meet GDPR/regulatory transparency requirements

---

## SHAP (SHapley Additive exPlanations)

### What is SHAP?

SHAP is a **game theory-based approach** to explain machine learning predictions by computing the contribution of each feature to a specific prediction.

**Key Concepts**:
- **Shapley Values**: Fair attribution of prediction to each feature
- **Additive**: Sum of all SHAP values = (prediction - base value)
- **Consistent**: If a feature contributes more, its SHAP value is higher

### Mathematical Foundation

```
φ_i = Σ over all subsets S not containing i of:
    |S|! × (|F| - |S| - 1)! / |F|! × [f(S ∪ {i}) - f(S)]

Where:
- φ_i = SHAP value for feature i
- S = subset of features
- F = full feature set
- f(S) = model prediction using only features in S
```

**Simplified Interpretation**: SHAP value = average marginal contribution of a feature across all possible feature combinations

---

## Implementation in DriftGuard

### File Location

**Notebook**: `notebooks/07_explainability_shap.ipynb`

### SHAP for Isolation Forest

```python
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load trained Isolation Forest model
from src.data_loader import load_iforest_model, load_phase2_features

model = load_iforest_model()
X_test, y_test = load_phase2_features()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# SHAP values shape: (n_samples, n_features)
print(f"SHAP values shape: {shap_values.shape}")
```

### Visualization Types

#### 1. Waterfall Plot (Individual Prediction)

Shows how each feature contributes to a single prediction.

```python
# Explain a high-risk sample
high_risk_idx = np.argmax(risk_scores)

# Create waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test[high_risk_idx],
        feature_names=feature_names
    )
)
plt.savefig('shap_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
```

**Example Interpretation**:
```
Base value (expected): 0.45
+ Flow Duration (high) → +0.15
+ Total Fwd Packets (high) → +0.12
+ Flow Bytes/s (high) → +0.10
- Avg Packet Size (normal) → -0.05
= Final prediction: 0.77 (HIGH RISK)
```

#### 2. Summary Plot (Global Feature Importance)

Shows the most important features across all predictions.

```python
# Summary plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar"
)
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')

# Detailed summary with value distribution
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="dot"
)
plt.savefig('shap_summary_dot.png', dpi=300, bbox_inches='tight')
```

**Interpretation**:
- **Red dots**: High feature values
- **Blue dots**: Low feature values
- **X-axis**: SHAP value (impact on prediction)
- **Features ordered by importance** (top to bottom)

#### 3. Force Plot (Interactive)

Interactive HTML visualization for exploring multiple predictions.

```python
# Force plot for a single sample
shap.force_plot(
    explainer.expected_value,
    shap_values[high_risk_idx],
    X_test[high_risk_idx],
    feature_names=feature_names
)

# Force plot for multiple samples (interactive)
shap.force_plot(
    explainer.expected_value,
    shap_values[:100],
    X_test[:100],
    feature_names=feature_names
)
```

---

## Top Features for Anomaly Detection

Based on SHAP analysis across all test samples:

| Rank | Feature | Avg SHAP | Impact | Description |
|------|---------|----------|--------|-------------|
| 1 | Flow Duration | 0.18 | +/- | Total duration of network flow |
| 2 | Total Fwd Packets | 0.15 | + | Number of forward packets |
| 3 | Flow Bytes/s | 0.14 | + | Bytes per second throughput |
| 4 | Flow Packets/s | 0.12 | + | Packets per second rate |
| 5 | Avg Packet Length | 0.11 | +/- | Average size of packets |
| 6 | PSH Flag Count | 0.10 | + | Number of PUSH flags (urgent data) |
| 7 | SYN Flag Count | 0.09 | +/- | TCP SYN flags (connection attempts) |
| 8 | Idle Mean | 0.08 | - | Average idle time between packets |
| 9 | Fwd Packet Length Std | 0.07 | + | Variance in forward packet sizes |
| 10 | Bwd Packet Length Mean | 0.06 | +/- | Average backward packet size |

### Impact Interpretation

- **+ (Positive)**: Higher values → Higher anomaly score
- **- (Negative)**: Higher values → Lower anomaly score
- **+/- (Mixed)**: Depends on context

---

## Use Cases for Security Analysts

### 1. Investigating High-Risk Alerts

**Scenario**: Alert flagged with risk score 0.85

```python
# Get SHAP explanation
idx = alert_index
explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test[idx],
    feature_names=feature_names
)

# Show waterfall plot
shap.waterfall_plot(explanation)

# Analyst sees:
# - Flow Duration: +0.20 (very long connection)
# - Flow Bytes/s: +0.15 (high data transfer rate)
# - PSH Flag Count: +0.12 (urgent data transmission)
# → Conclusion: Likely data exfiltration attempt
```

### 2. Understanding False Positives

**Scenario**: Benign traffic flagged as malicious

```python
# Find false positives
false_positive_idx = np.where((predictions == 1) & (y_test == 0))[0]

# Analyze common patterns
fp_shap = shap_values[false_positive_idx]
mean_fp_shap = np.mean(np.abs(fp_shap), axis=0)
top_fp_features = np.argsort(mean_fp_shap)[-5:]

print("Features causing false positives:")
for feat_idx in top_fp_features[::-1]:
    print(f"{feature_names[feat_idx]}: {mean_fp_shap[feat_idx]:.3f}")

# Result:
# Flow Duration: 0.180 (long-running legitimate transfers)
# Total Fwd Packets: 0.145 (bulk data transfers)
# Flow Bytes/s: 0.132 (high-throughput applications)
```

**Actionable Insight**: Whitelist known backup jobs or large file transfers

### 3. Feature Engineering Guidance

**Question**: Which features should we focus on improving?

```python
# Global feature importance
global_importance = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(global_importance)[::-1]

print("Top 10 most important features for detection:")
for i, idx in enumerate(sorted_idx[:10], 1):
    print(f"{i}. {feature_names[idx]}: {global_importance[idx]:.4f}")
```

---

## SHAP Limitations & Considerations

### Computational Cost

⚠️ **SHAP is computationally expensive** for large datasets

```python
# For 10,000 samples, 77 features:
# TreeExplainer: ~30 seconds
# KernelExplainer: ~10 minutes
```

**Solution**: Sample subset for SHAP analysis

```python
# Analyze random sample
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
shap_values_sample = explainer.shap_values(X_test[sample_idx])
```

### Model-Specific Explainers

Different models require different SHAP explainers:

| Model | SHAP Explainer | Speed |
|-------|----------------|-------|
| Isolation Forest | TreeExplainer | Fast ⚡ |
| Autoencoder | DeepExplainer | Medium 🔄 |
| LSTM | DeepExplainer | Slow 🐌 |
| Ensemble | KernelExplainer | Very Slow 🐌🐌 |

### Interpretatbility vs. Accuracy Trade-off

- **Complex models** (deep neural networks) → Higher accuracy, harder to explain
- **Simple models** (decision trees) → Lower accuracy, easier to understand

---

## Practical Workflow

### Step-by-Step SHAP Analysis

```python
# 1. Train model
model = IsolationForest()
model.fit(X_train)

# 2. Make predictions
predictions = model.predict(X_test)
scores = model.decision_function(X_test)

# 3. Initialize SHAP
explainer = shap.TreeExplainer(model)

# 4. Compute SHAP values
shap_values = explainer.shap_values(X_test)

# 5. Save SHAP values for later use
np.save('data/shap_values.npy', shap_values)

# 6. Visualize global importance
shap.summary_plot(shap_values, X_test, feature_names=features)

# 7. Investigate high-risk samples
high_risk = np.where(scores < -0.5)[0]
for idx in high_risk[:10]:
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test[idx]
    ))
```

---

## Integration with Dashboard

```python
# In dashboard/app.py
import shap
import streamlit as st

# Load SHAP values
shap_values = np.load('data/shap_values.npy')

# Display explanation for selected alert
selected_idx = st.selectbox("Select alert:", range(len(alerts)))

# Show SHAP waterfall
fig = shap.waterfall_plot(
    shap.Explanation(values=shap_values[selected_idx], ...),
    show=False
)
st.pyplot(fig)

# Show top contributing features
top_features = get_top_shap_features(shap_values[selected_idx])
st.write("Top contributing features:")
for feat, value in top_features:
    st.write(f"- {feat}: {value:.3f}")
```

---

## References

- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- SHAP Documentation: https://shap.readthedocs.io/
- SHAP GitHub: https://github.com/slundberg/shap

---

**Next: [Visualization & Dashboard →](10_VISUALIZATION_DASHBOARD.md)**

**[← Previous: Cross-Dataset Validation](08_CROSS_DATASET_VALIDATION.md) | [Back to Index](00_INDEX.md)**

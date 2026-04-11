# Visualization & Dashboard

**[← Previous: Explainability](09_EXPLAINABILITY.md)** | **[Back to Index](00_INDEX.md)** | **[Next: Deployment →](11_DEPLOYMENT.md)**

---

## Streamlit Dashboard

**File**: `dashboard/app.py`

DriftGuard includes an interactive **Streamlit dashboard** for real-time monitoring and visualization of detection results.

### Dashboard Features

#### 1. Key Performance Indicators (KPIs)
- **Total Events**: Number of network flows processed
- **High Risk Alerts**: Count of HIGH severity alerts
- **Detected Attacks**: Ground truth attack count

#### 2. Alert Severity Distribution
- Bar chart showing HIGH/MEDIUM/LOW distribution
- Visual breakdown of risk levels
- Real-time updates as new data arrives

#### 3. Top Risk Alerts Table
- Top 20 highest risk scores
- Sortable by risk score, severity, ground truth
- Filterable table for analyst investigation
- Shows risk score, severity label, and actual label

####4. Alert Explanation
- Detailed breakdown of why specific alerts were flagged
- Shows contribution from each model:
  - **Isolation Forest**: Rare behavior detection
  - **Autoencoder**: Behavioral deviation
  - **LSTM**: Sequential anomalies
- Educational for security analysts

### Running the Dashboard

```bash
# Navigate to dashboard directory
cd dashboard

# Run Streamlit app
streamlit run app.py

# Dashboard opens at http://localhost:8501
```

### Dashboard Code Overview

```python
import streamlit as st
import pandas as pd
from src.data_loader import load_final_outputs
from src.risk_scoring import severity_from_score

# Page configuration
st.set_page_config(
    page_title="DriftGuard SOC Dashboard",
    layout="wide"
)

st.title("🛡️ DriftGuard — SOC Dashboard")
st.caption("Adaptive ML-based Cyber Threat Detection System")

# Load risk scores and labels
risk_scores, labels = load_final_outputs()

# Create DataFrame
df = pd.DataFrame({
    "Risk Score": risk_scores,
    "Ground Truth": labels
})
df["Severity"] = df["Risk Score"].apply(severity_from_score)

# Display KPIs in three columns
col1, col2, col3 = st.columns(3)
col1.metric("Total Events", len(df))
col2.metric("High Risk Alerts", (df["Severity"] == "HIGH").sum())
col3.metric("Detected Attacks", (df["Ground Truth"] == 1).sum())

# Severity distribution bar chart
st.subheader("📊 Alert Severity Distribution")
st.bar_chart(df["Severity"].value_counts())

# Top alerts table
st.subheader("🚨 Top Risk Alerts")
top_alerts = df.sort_values("Risk Score", ascending=False).head(20)
st.dataframe(top_alerts, use_container_width=True)

# Alert explanation for top alert
st.subheader("🔍 Alert Explanation")
top = top_alerts.iloc[0]
st.markdown(f"""
**Risk Score:** `{top['Risk Score']:.2f}`  
**Severity:** `{top['Severity']}`  

This alert was flagged due to strong deviations from learned
baseline behavior across multiple anomaly detection models.
The final score is a fusion of:
- Isolation Forest (rare behavior)
- Autoencoder (behavioral deviation)
- LSTM (sequential anomalies)
""")
```

### Dashboard Benefits

✅ **Interactive Exploration**: Analysts can filter and sort alerts  
✅ **Real-Time Monitoring**: Updates with latest predictions  
✅ **Educational**: Helps understand model decision-making  
✅ **Deployment Ready**: Can be deployed as web application  
✅ **Low Barrier**: No ML expertise required to use  

### Production Deployment

```bash
# Deploy with Streamlit Cloud (free)
streamlit deploy dashboard/app.py

# Or deploy with Docker
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py"]
```

---

## Visualization Utilities

### plot_utils.py

**Purpose**: Standardized plotting functions for consistent visualization across all notebooks and scripts.

#### save_plot() Function

```python
import os
import matplotlib.pyplot as plt

def save_plot(filename, phase, dpi=300):
    """
    Save matplotlib plots with consistent formatting.
    
    Args:
        filename: Name of the output file (e.g., 'model_comparison.png')
        phase: Project phase directory (e.g., 'phase2', 'experiments')
        dpi: Resolution (default 300 for publication quality)
    
    Returns:
        None (saves plot to disk)
    """
    base_path = f"../reports/figures/{phase}"
    os.makedirs(base_path, exist_ok=True)
    
    full_path = os.path.join(base_path, filename)
    
    plt.tight_layout()
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved → {full_path}")
```

#### Usage Example

```python
from src.plot_utils import save_plot
import matplotlib.pyplot as plt

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")

# Save with standardized settings
save_plot("training_loss.png", "experiments")
# Saved to: reports/figures/experiments/training_loss.png
```

#### Benefits

- **Consistency**: All plots use same DPI, bbox settings
- **Organization**: Automatic directory structure
- **Publication Ready**: 300 DPI default
- **No Memory Leaks**: `plt.close()` after save

---

### fix_plot_saving.py

**Purpose**: Automated script to modify Jupyter notebooks for batch plot saving

#### Problem Solved

In interactive notebooks, we use `plt.show()` to display plots. For documentation and reproducibility, we need to save these plots as PNG files. Manually editing every cell is tedious.

#### Solution

This script automatically:
1. Reads Jupyter notebook JSON structure
2. Finds code cells with `plt.show()`
3. Replaces with `plt.savefig()` and `plt.close()`
4. Saves modified notebook
5. Re-run notebook to generate all PNG files

#### Code

```python
import os
import json

# Read notebook
notebook_path = r'research/experiments/exp3_dataset_shift_analysis.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and modify cells
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'plt.show()' in source and 'for feature in top_features' in source:
            # Add directory creation
            new_source = []
            new_source.append('import os\\n')
            new_source.append('os.makedirs("../../results/figures", exist_ok=True)\\n')
            
            # Replace plt.show() with plt.savefig()
            for line in cell['source']:
                if 'plt.show()' in line:
                    new_source.append(f'    plt.savefig(f"../../results/figures/experiment3_feature_shift_{{feature}}.png", dpi=300, bbox_inches="tight")\\n')
                    new_source.append(f'    plt.close()\\n')
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            break

# Save modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
```

#### Use Case

```bash
# Run the modification script
python src/fix_plot_saving.py

# Re-execute the modified notebook
jupyter nbconvert --to notebook --execute \
    research/experiments/exp3_dataset_shift_analysis.ipynb

# Results:10 PNG files saved to research/results/figures/shifts_exp/
```

---

## Generated Visualizations

### Experiment Figures

#### 1. Model Comparison
**File**: `research/results/figures/experiment1_model_comparison.png`

Bar chart comparing:
- Precision
- Recall
- F1-Score
- False Positive Rate

Across all four models (Isolation Forest, Autoencoder, LSTM, Ensemble)

#### 2. False Positive SHAP Waterfall
**File**: `research/results/figures/experiment2_fp_shap_waterfall.png`

SHAP waterfall plot showing:
- Base value (expected output)
- Feature contributions (positive/negative)
- Final prediction value
- Color-coded impact

#### 3. Distribution Shift Plots
**Directory**: `research/results/figures/shifts_exp/`

10 histogram plots comparing CICIDS2017 vs UNSW-NB15 distributions:

| File | Feature ID | Description |
|------|------------|-------------|
| `experiment3_feature_shift_0.png` | 0 | Feature 0 distribution |
| `experiment3_feature_shift_5.png` | 5 | Feature 5 distribution |
| `experiment3_feature_shift_19.png` | 19 | Feature 19 distribution |
| `experiment3_feature_shift_28.png` | 28 | Feature 28 distribution |
| `experiment3_feature_shift_29.png` | 29 | Feature 29 distribution |
| `experiment3_feature_shift_34.png` | 34 | Feature 34 distribution |
| `experiment3_feature_shift_37.png` | 37 | Feature 37 distribution |
| `experiment3_feature_shift_43.png` | 43 | Feature 43 distribution |
| `experiment3_feature_shift_45.png` | 45 | Feature 45 distribution |
| `experiment3_feature_shift_50.png` | 50 | Feature 50 distribution |

Each plot shows:
- Blue histogram: CICIDS2017 distribution
- Orange histogram: UNSW-NB15 distribution
- Visual confirmation of KS/PSI statistics

---

## Visualization Best Practices

### Publication-Quality Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot(x, y, linewidth=2, label="Model Performance")

# Formatting
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=14)
ax.set_title("Model Training Progress", fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3)

# Save
plt.tight_layout()
plt.savefig("results.png", dpi=300, bbox_inches='tight')
plt.close()
```

### Interactive Plots with Plotly

```python
import plotly.graph_objects as go

# Create interactive scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=risk_scores,
    y=ground_truth,
    mode='markers',
    marker=dict(
        size=8,
        color=risk_scores,
        colorscale='Reds',
        showscale=True
    ),
    text=[f"Risk: {score:.2f}" for score in risk_scores],
    hovertemplate='%{text}<br>Label: %{y}<extra></extra>'
))

fig.update_layout(
    title="Risk Score Distribution",
    xaxis_title="Risk Score",
    yaxis_title="Ground Truth",
    hovermode='closest'
)

# Save as HTML (interactive)
fig.write_html("risk_distribution_interactive.html")
```

---

## Dashboard Extensions (Future)

### Planned Features

1. **Real-Time Streaming**
   - Live connection to Zeek logs
   - Continuous risk score updates
   - Rolling statistics

2. **Alert Management**
   - Acknowledge/dismiss alerts
   - Add analyst notes
   - Export to SIEM

3. **Model Performance Monitoring**
   - Track accuracy over time
   - Detect drift in production
   - Alert on performance degradation

4. **MITRE ATT&CK Integration**
   - Auto-map alerts to tactics/techniques
   - Attack chain visualization
   - Threat intelligence enrichment

---

**Next: [Deployment Guide →](11_DEPLOYMENT.md)**

**[← Previous: Explainability](09_EXPLAINABILITY.md) | [Back to Index](00_INDEX.md)**

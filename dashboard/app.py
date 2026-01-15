import sys
import os

# Allow importing from src/
sys.path.append(os.path.abspath(".."))

import streamlit as st
import pandas as pd

from src.data_loader import load_final_outputs
from src.risk_scoring import severity_from_score

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="ADAPT-SEC SOC Dashboard",
    layout="wide"
)

st.title("🛡️ ADAPT-SEC — SOC Dashboard")
st.caption("Adaptive ML-based Cyber Threat Detection System")

# -----------------------
# Load data
# -----------------------
risk_scores, labels = load_final_outputs()

df = pd.DataFrame({
    "Risk Score": risk_scores,
    "Ground Truth": labels
})

df["Severity"] = df["Risk Score"].apply(severity_from_score)

# -----------------------
# KPIs
# -----------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Events", len(df))
col2.metric("High Risk Alerts", (df["Severity"] == "HIGH").sum())
col3.metric("Detected Attacks", (df["Ground Truth"] == 1).sum())

st.divider()

# -----------------------
# Severity Distribution
# -----------------------
st.subheader("📊 Alert Severity Distribution")
st.bar_chart(df["Severity"].value_counts())

# -----------------------
# Top Alerts
# -----------------------
st.subheader("🚨 Top Risk Alerts")

top_alerts = df.sort_values("Risk Score", ascending=False).head(20)

st.dataframe(
    top_alerts,
    use_container_width=True
)

# -----------------------
# Alert Explanation (Textual)
# -----------------------
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

import sys
import os

# Allow importing from src/
sys.path.append(os.path.abspath(".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.data_loader import load_final_outputs, load_all_model_scores
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


def trigger_auto_refresh(interval_seconds: int) -> None:
    interval_ms = max(5, int(interval_seconds)) * 1000
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {interval_ms});
        </script>
        """,
        height=0,
    )

@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    risk_scores, labels = load_final_outputs()
    model_scores = load_all_model_scores()

    valid_lengths = [
        len(risk_scores),
        len(labels),
        len(model_scores["iforest"]),
        len(model_scores["autoencoder"]),
        len(model_scores["lstm"]),
    ]
    min_len = min(valid_lengths)

    risk_scores = risk_scores[:min_len]
    labels = labels[:min_len]
    iforest = model_scores["iforest"][:min_len]
    autoencoder = model_scores["autoencoder"][:min_len]
    lstm = model_scores["lstm"][:min_len]

    def minmax(values: np.ndarray, invert: bool = False) -> np.ndarray:
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if max_val - min_val == 0:
            norm = np.zeros_like(values, dtype=float)
        else:
            norm = (values - min_val) / (max_val - min_val)
        if invert:
            norm = 1.0 - norm
        return np.clip(norm, 0.0, 1.0)

    iforest_norm = minmax(iforest, invert=True)
    ae_norm = minmax(autoencoder)
    lstm_norm = minmax(lstm)

    df = pd.DataFrame({
        "Event ID": np.arange(min_len),
        "Risk Score": risk_scores,
        "Ground Truth": labels,
        "IForest Raw": iforest,
        "Autoencoder Raw": autoencoder,
        "LSTM Raw": lstm,
        "IForest Norm": iforest_norm,
        "Autoencoder Norm": ae_norm,
        "LSTM Norm": lstm_norm,
    })

    df["Severity"] = df["Risk Score"].apply(severity_from_score)
    df["Ground Truth"] = pd.to_numeric(df["Ground Truth"], errors="coerce").fillna(0).astype(int)
    df["Attack"] = np.where(df["Ground Truth"] == 1, "Attack", "Benign")

    end_time = pd.Timestamp.now().floor("min")
    df["Event Time"] = pd.date_range(end=end_time, periods=min_len, freq="min")

    return df


def severity_sort_order(series: pd.Series) -> pd.Series:
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return series.map(order).fillna(99)


def add_behavior_clusters(dataframe: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    features = dataframe[["Risk Score", "IForest Norm", "Autoencoder Norm", "LSTM Norm"]].copy()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    dataframe = dataframe.copy()
    dataframe["Behavior Cluster"] = model.fit_predict(features)

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)
    dataframe["Cluster X"] = reduced[:, 0]
    dataframe["Cluster Y"] = reduced[:, 1]
    return dataframe


df = load_dashboard_data()

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("SOC Controls")

severity_options = ["HIGH", "MEDIUM", "LOW"]
selected_severity = st.sidebar.multiselect(
    "Severity",
    options=severity_options,
    default=severity_options,
)

attack_filter = st.sidebar.selectbox(
    "Traffic Type",
    options=["All", "Attack", "Benign"],
    index=0,
)

risk_min = float(df["Risk Score"].min())
risk_max = float(df["Risk Score"].max())
selected_risk = st.sidebar.slider(
    "Risk Score Range",
    min_value=float(round(risk_min, 4)),
    max_value=float(round(risk_max, 4)),
    value=(float(round(risk_min, 4)), float(round(risk_max, 4))),
)

top_n = st.sidebar.slider("Top Alerts to Show", min_value=10, max_value=100, value=25, step=5)

threshold = st.sidebar.slider("Operational Alert Threshold", 0.0, 1.0, 0.7, 0.01)

st.sidebar.subheader("Live Monitoring")
enable_auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=5, max_value=120, value=20, step=5)
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()

cluster_count = st.sidebar.slider("Behavior Clusters", min_value=2, max_value=8, value=4, step=1)

if enable_auto_refresh:
    st.cache_data.clear()
    trigger_auto_refresh(refresh_interval)

filtered = df[df["Severity"].isin(selected_severity)].copy()
filtered = filtered[(filtered["Risk Score"] >= selected_risk[0]) & (filtered["Risk Score"] <= selected_risk[1])]

if attack_filter != "All":
    filtered = filtered[filtered["Attack"] == attack_filter]

filtered = filtered.sort_values("Risk Score", ascending=False)

if filtered.empty:
    st.warning("No events match the current filters. Relax one or more filters in the sidebar.")
    st.stop()

# -----------------------
# KPI Row
# -----------------------
total_events = len(filtered)
high_risk = int((filtered["Severity"] == "HIGH").sum())
detected_attacks = int((filtered["Ground Truth"] == 1).sum())
above_threshold = int((filtered["Risk Score"] >= threshold).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Events", total_events)
col2.metric("High Risk Alerts", high_risk)
col3.metric("Detected Attacks", detected_attacks)
col4.metric("Above Threshold", above_threshold)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Behavior Clusters", "Alert Explorer"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Severity Distribution")
        sev_counts = (
            filtered["Severity"]
            .value_counts()
            .rename_axis("Severity")
            .reset_index(name="Count")
            .sort_values("Severity", key=severity_sort_order)
            .set_index("Severity")
        )
        st.bar_chart(sev_counts)

    with right:
        st.subheader("Risk Score Histogram")
        bins = np.linspace(0, 1, 21)
        hist = pd.cut(filtered["Risk Score"], bins=bins, include_lowest=True).value_counts().sort_index()
        hist_df = hist.rename_axis("Risk Bin").reset_index(name="Count")
        hist_df["Risk Bin"] = hist_df["Risk Bin"].astype(str)
        st.bar_chart(hist_df.set_index("Risk Bin"))

    st.subheader("Top Risk Alerts")
    top_alerts = filtered.head(top_n).copy()
    st.dataframe(
        top_alerts[
            [
                "Event ID",
                "Event Time",
                "Risk Score",
                "Severity",
                "Attack",
                "Ground Truth",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = top_alerts.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Filtered Alerts (CSV)",
        data=csv_data,
        file_name="filtered_alerts.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("Risk Trend Over Time")
    trend_df = (
        filtered.set_index("Event Time")
        .resample("30min")
        .agg(
            Mean_Risk=("Risk Score", "mean"),
            Alert_Count=("Risk Score", "count"),
            High_Risk_Count=("Severity", lambda x: int((x == "HIGH").sum())),
        )
        .reset_index()
    )

    st.line_chart(trend_df.set_index("Event Time")[["Mean_Risk", "High_Risk_Count"]])

    st.subheader("Model Contribution Profile")
    model_profile = pd.DataFrame({
        "Model": ["Isolation Forest", "Autoencoder", "LSTM"],
        "Average Normalized Contribution": [
            filtered["IForest Norm"].mean(),
            filtered["Autoencoder Norm"].mean(),
            filtered["LSTM Norm"].mean(),
        ],
    }).set_index("Model")
    st.bar_chart(model_profile)

with tab3:
    st.subheader("Attack Behavior Clustering")
    cluster_df = add_behavior_clusters(filtered, n_clusters=cluster_count)

    cluster_summary = (
        cluster_df.groupby("Behavior Cluster", as_index=False)
        .agg(
            Events=("Event ID", "count"),
            Mean_Risk=("Risk Score", "mean"),
            Attack_Rate=("Ground Truth", "mean"),
            High_Risk_Count=("Severity", lambda x: int((x == "HIGH").sum())),
        )
        .sort_values("Mean_Risk", ascending=False)
    )
    cluster_summary["Attack_Rate"] = cluster_summary["Attack_Rate"] * 100

    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    fig = px.scatter(
        cluster_df,
        x="Cluster X",
        y="Cluster Y",
        color="Behavior Cluster",
        size="Risk Score",
        hover_data=["Event ID", "Severity", "Attack", "Risk Score"],
        title="Behavior Cluster Map (PCA projection)",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    cluster_choice = st.selectbox(
        "Inspect Cluster",
        options=sorted(cluster_df["Behavior Cluster"].unique().tolist()),
        index=0,
        key="cluster_inspect",
    )
    cluster_slice = cluster_df[cluster_df["Behavior Cluster"] == cluster_choice].sort_values("Risk Score", ascending=False)
    st.dataframe(
        cluster_slice[["Event ID", "Event Time", "Risk Score", "Severity", "Attack", "IForest Norm", "Autoencoder Norm", "LSTM Norm"]].head(30),
        use_container_width=True,
        hide_index=True,
    )

with tab4:
    st.subheader("Investigate Specific Alert")

    event_ids = filtered["Event ID"].tolist()
    selected_event = st.selectbox("Select Event ID", options=event_ids, index=0)

    alert_row = filtered[filtered["Event ID"] == selected_event].iloc[0]

    a1, a2, a3 = st.columns(3)
    a1.metric("Risk Score", f"{alert_row['Risk Score']:.4f}")
    a2.metric("Severity", alert_row["Severity"])
    a3.metric("Traffic", alert_row["Attack"])

    st.write("Model-level normalized anomaly signals")
    contrib = pd.DataFrame(
        {
            "Signal": ["IForest Norm", "Autoencoder Norm", "LSTM Norm"],
            "Score": [
                float(alert_row["IForest Norm"]),
                float(alert_row["Autoencoder Norm"]),
                float(alert_row["LSTM Norm"]),
            ],
        }
    ).set_index("Signal")
    st.bar_chart(contrib)

    st.markdown(
        f"""
**Event Time:** `{alert_row['Event Time']}`  
**Ground Truth Label:** `{int(alert_row['Ground Truth'])}`  
**Operational Threshold Breach:** `{bool(alert_row['Risk Score'] >= threshold)}`

This alert is prioritized because the fused risk score indicates multi-model deviation from baseline behavior.
The decomposition chart above helps SOC analysts quickly see whether this event is dominated by distributional outliers (IForest), reconstruction anomalies (Autoencoder), or sequence irregularities (LSTM).
"""
    )

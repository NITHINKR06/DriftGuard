# 🧠 DriftGuard — Upgrade Plan

## 🌟 Unique Feature to Add: MITRE ATT&CK Mapping
Map every detected anomaly to a MITRE ATT&CK tactic and technique ID. This is what real SOC tools do — Splunk, CrowdStrike, Elastic SIEM all do this. It bridges your ML output to actionable threat intelligence and is something no other student IDS project does.

---

## Phase 1 — Credibility Fixes (1 week)

- [ ] **Add results table to README** — the numbers exist in your notebooks, just surface them:

| Dataset | Detection Rate | FPR | AUC |
|---|---|---|---|
| CICIDS2017 (train) | xx% | xx% | x.xx |
| UNSW-NB15 (validate) | xx% | xx% | x.xx |
| UGR16 (stress test) | xx% | xx% | x.xx |

- [ ] **Embed one ROC curve image** in README — export from your notebook as PNG, add one line in markdown. Single most important ML proof artifact.
- [ ] **Document threshold rationale** — one paragraph in README explaining why 0.4 / 0.7 were chosen (ROC elbow, F1 tradeoff, cost of FP vs FN in IDS context)
- [ ] **Add per-attack-class F1 table** — separate scores for DoS, port scan, brute force, web attacks, infiltration

---

## Phase 2 — New Unique Features (3–4 weeks)

### 🗺️ MITRE ATT&CK Mapping (The Standout Feature)
Build a mapping layer between anomaly patterns and ATT&CK Tactic + Technique IDs.

**Mapping table (starter set):**
```python
ATTACK_MAPPING = {
    "syn_flood":         {"tactic": "Impact",        "technique": "T1498", "name": "Network Denial of Service"},
    "port_scan":         {"tactic": "Discovery",     "technique": "T1046", "name": "Network Service Discovery"},
    "brute_force_ssh":   {"tactic": "Credential Access", "technique": "T1110", "name": "Brute Force"},
    "data_exfil":        {"tactic": "Exfiltration",  "technique": "T1041", "name": "Exfiltration Over C2 Channel"},
    "lateral_movement":  {"tactic": "Lateral Movement", "technique": "T1021", "name": "Remote Services"},
    "c2_beacon":         {"tactic": "Command & Control", "technique": "T1071", "name": "Application Layer Protocol"},
}
```

**In Streamlit dashboard — alert card shows:**
```
Alert: CRITICAL (score: 0.83)
Flow: 192.168.1.45 → 10.0.0.1:22

MITRE ATT&CK:
  Tactic:    Credential Access
  Technique: T1110 — Brute Force
  Sub-tech:  T1110.001 — Password Guessing
  [View on ATT&CK →]

Top SHAP features: flow_duration (+0.41), flag_SYN_ratio (+0.38)
```

### 🔌 REST Prediction API (FastAPI)
```
POST /predict
Body: { "features": { "flow_duration": 12, "fwd_packet_length_max": 1480, ... } }

Response: {
  "anomaly_score": 0.83,
  "label": "critical",
  "attack_mapping": { "tactic": "Credential Access", "technique": "T1110" },
  "shap_values": { "flow_duration": 0.41, "flag_SYN_ratio": 0.38 },
  "threshold_used": 0.7
}
```
This makes DriftGuard integrable into any SOC pipeline or SIEM tool — huge for interviews.

### 📡 Live PCAP Capture Mode (Beta)
```bash
python src/pipeline/serve.py --interface eth0 --threshold 0.7
```
- Uses `dpkt` or `scapy` to capture live packets
- Computes CICFlowMeter-style features per flow in real time
- Scores and alerts in the Streamlit dashboard as flows complete
- Start with file-based PCAP replay (`--pcap capture.pcap`) before live capture

---

## Phase 3 — Research Depth (2 weeks)

- [ ] **Incremental autoencoder fine-tuning** — `POST /api/retrain` accepts new labeled flows (JSON), fine-tunes the AE for 5 epochs without full retraining; useful for adapting to new network environments
- [ ] **Attack class breakdown in dashboard** — separate detection rate + F1 chart per attack category, not just aggregate accuracy
- [ ] **Suricata alert ingestor** — parse `eve.json` from Suricata and feed flagged flows into DriftGuard as a second-stage ML classifier (reduces Suricata false positives)
- [ ] **Drift alert email/webhook** — when PSI > 0.2 on any feature, POST a drift warning to a Slack webhook or email via smtplib

---

## Resume Line After Upgrades

> "Implemented MITRE ATT&CK technique mapping for ML-detected anomalies. Exposed FastAPI REST prediction endpoint for SOC/SIEM integration. Cross-dataset validation across CICIDS2017, UNSW-NB15, and UGR16 with documented per-class F1 scores."

---

## Quick Wins (Do These First — Under 1 Day Each)

| Task | Time | Impact |
|---|---|---|
| Add results table + ROC curve to README | 1 hr | Instant credibility — #1 missing thing |
| Document threshold selection rationale | 30 min | Removes the obvious interview question |
| Add per-attack-class F1 table | 1 hr | Shows ML depth beyond overall accuracy |

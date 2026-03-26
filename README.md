# HealthShield
### Privacy-Enhancing AI for Maternal Health · On-Device · Offline · No Data Leaves the Phone

> *"We did not make this responsible by adding a governance layer. We made it responsible by designing a system where the harmful option does not exist. The data cannot leave the device because we never built the pipe."*

---

## What This Is

HealthShield is a Streamlit prototype for a maternal health decision support application designed for community health workers (CHWs) in low-connectivity settings across Sub-Saharan Africa and beyond.

It combines two AI components — a small risk classifier and a small language model — to give a CHW visiting a pregnant woman at home something that has not existed before in this context: a clinical thinking partner that works offline, in her language, at the patient's door.

The classifier tells her what the risk level is. The language model tells her what it means, what to do, what to say, and what to write down.

No patient data leaves the device. Ever.

---

## The Problem

Of the 287,000 maternal deaths that occur globally each year, the overwhelming majority happen in low- and middle-income countries — in contexts where:

- Community health workers have no real-time clinical decision support at the point of care
- Health data cannot be shared across institutions because of legitimate re-identification risk
- AI tools fail because they assume internet connectivity and hardware that does not exist in the communities they are meant to serve
- Deploying a model in a new region requires a local dataset that does not yet exist — a cold-start paradox that has blocked clinical AI deployment for years

HealthShield is designed to address all three of these problems in one architecture.

---

## Core Capabilities

### 1. On-Device Risk Prediction
A trained `GradientBoostingClassifier` takes 8 clinical measurements and returns a maternal risk level (LOW / MID / HIGH) with a confidence score in under one second. No internet required. Runs on any machine.

**Inputs:** Age · Systolic BP · Diastolic BP · Blood Glucose · Body Temperature · Heart Rate · Haemoglobin · Gestational Week

### 2. Small Language Model — Five Integration Points
A local Gemma-2B model (via Ollama or HuggingFace transformers) is integrated across five distinct moments in the CHW workflow:

| Integration Point | What It Does |
|---|---|
| **Risk Explanation** | Explains the risk result in plain English and Swahili |
| **Conversational Q&A** | Answers CHW follow-up questions in clinical context |
| **Referral Letter** | Generates a structured referral note for the receiving facility |
| **Trajectory Interpretation** | Explains rising/falling vital trends in plain language |
| **CHW Training** | Answers clinical questions between patient visits |

### 3. Longitudinal Trajectory Tracking
Tracks patient measurements across visits stored in a local SQLite database. Computes rolling deltas and flags patients as DETERIORATING / STABLE / IMPROVING. A sparkline visualisation shows directional change across visits.

### 4. Synthetic-First Model Bootstrapping
A CTGAN-based synthetic data generator conditioned on demographic inputs (region, facility access distance, HIV prevalence) produces a locally-calibrated training dataset without requiring any real patient records. Solves the cold-start problem: deploy anywhere, day one, zero real records needed.

### 5. Differentially Private Federated Learning (Simulated)
Gaussian noise calibrated to a per-update epsilon value is applied to model weight updates before aggregation. Provides a formal mathematical guarantee that individual records cannot be reconstructed from the update stream. Epsilon fixed at 0.1 in the current prototype.

### 6. Privacy Passport
Before any synthetic export, an on-device auditor computes re-identification risk using nearest-neighbour distance on quasi-identifiers. If the dataset passes, a cryptographic certificate is generated: SHA-256 hash, epsilon value, re-ID risk score, record count. Enables clinics to share data with researchers safely for the first time.

---

## Architecture

```
healthshield/
├── app.py                     # Main entry point, sidebar navigation
├── pages/
│   ├── 1_patients.py          # Patient list, add/view patients
│   ├── 2_assessment.py        # Risk assessment + SLM explanation + Q&A
│   ├── 3_trajectory.py        # Longitudinal tracking + SLM interpretation
│   ├── 4_model_setup.py       # Synthetic bootstrap + model retraining
│   └── 5_privacy_passport.py  # Privacy audit + certified export
├── models/
│   ├── train.py               # Train + cache GradientBoostingClassifier
│   └── classifier.joblib      # Cached model
├── data/
│   └── uci_maternal_health.csv
├── db/
│   └── healthshield.db        # SQLite — patients, assessments, exports
├── utils/
│   ├── database.py            # All SQLite read/write
│   ├── privacy.py             # Re-ID risk, DP noise, certificate generation
│   ├── synthetic.py           # CTGAN wrapper
│   └── llm.py                 # Gemma-2B loader + inference + streaming
└── style/
    └── theme.css              # Custom CSS injected via st.markdown
```

---

## Database Schema

```sql
-- Patients
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    village TEXT,
    gestational_week INTEGER,
    gravidity INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessments (one row per CHW visit)
CREATE TABLE assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER REFERENCES patients(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    age INTEGER,
    systolic_bp REAL,
    diastolic_bp REAL,
    blood_glucose REAL,
    body_temp REAL,
    heart_rate REAL,
    haemoglobin REAL,
    gestational_week INTEGER,
    risk_level TEXT,
    confidence_score REAL
);

-- Synthetic exports
CREATE TABLE synthetic_exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    record_count INTEGER,
    epsilon REAL,
    reid_risk REAL,
    sha256_hash TEXT,
    certificate_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running (for local SLM)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/healthshield.git
cd healthshield
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the language model
```bash
ollama pull gemma:2b
```

### 4. Run the app
```bash
streamlit run app.py
```

The classifier trains automatically on first run using the UCI Maternal Health Risk dataset and caches to `models/classifier.joblib`.

### 5. (Optional) Use API mode during development
If you don't want to run Ollama locally, set your Gemma API key and switch the flag in `utils/llm.py`:
```python
USE_LOCAL = False  # uses Google AI Studio API
```
```bash
export GEMMA_API_KEY=your_key_here
```

---

## Dependencies

```
streamlit
scikit-learn
joblib
pandas
numpy
plotly
sdv
transformers
torch
bitsandbytes
ollama
google-generativeai
hashlib
sqlite3
```

---

## Hardware Requirements

| Component | Minimum Hardware |
|---|---|
| Risk classifier + privacy layer + trajectory | Any machine, 4GB RAM |
| Gemma-2B via Ollama (CPU) | 8GB RAM — slow but functional |
| Gemma-2B via Ollama (Apple Silicon) | 8GB RAM — usable speed |
| Gemma-2B via HuggingFace 4-bit | GPU recommended, 8GB VRAM |

---

## Data

The prototype trains on the [UCI Maternal Health Risk dataset](https://archive.ics.uci.edu/dataset/863/maternal+health+risk). This dataset was collected in rural Bangladesh. **It is used for prototype demonstration only.** The model must be validated against a locally-representative dataset before any clinical deployment.

---

## Current Status

This is a working prototype built for internal concept validation at the World Bank ITS Technology and Innovation Unit. It is not a clinically validated tool and must not be used for real patient decisions without Phase 1 clinical validation.

**Prototype demonstrates:** All six core capabilities individually and as an integrated workflow.

**Not yet done:** Clinical validation on a Tanzanian dataset · Swahili SLM output quality evaluation · Android packaging · Federated learning across multiple devices · Production security hardening.

---

## Honest Limits

- The risk classifier is trained on Bangladeshi data. Population-level differences mean accuracy figures from this dataset do not generalise to Tanzania without validation.
- The SLM generates plausible clinical language. It is not a substitute for clinical training, supervision, or a validated diagnostic tool.
- Differential privacy in this prototype is simulated — it demonstrates the mechanism and the privacy accounting, not a production federated learning deployment.
- The binding constraints on maternal mortality in target contexts — facility access, supply chains, skilled birth attendants — are not addressable by this application.

---

## Pilot Design

| Phase | Timeline | Goal |
|---|---|---|
| Phase 1 — Clinical Validation | Months 1–6 | Validate synthetic-bootstrapped classifier against clinician assessments on real Tanzanian data. CHW user study (n=20). |
| Phase 2 — Data Infrastructure | Months 7–12 | Open-source privacy pipeline toolkit. GIL research pilot. |
| Phase 3 — Federation + Publication | Months 13–18 | Multi-site federated learning. Methodology paper submission. |

---

## Contact

Sunidhi Goyal

---

*Built with the conviction that privacy-by-architecture is not a constraint on what AI can do in global health — it is the condition that makes deployment possible at all.*

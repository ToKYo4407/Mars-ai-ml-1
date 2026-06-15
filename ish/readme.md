 # Support Integrity Auditor (SIA)

Detects priority mismatches in support tickets using ML — flags tickets labelled too low (**Hidden Crisis**) or too high (**False Alarm**).


DEPLOYED LINK - https://ktnmvdgdndhmexajbrumec.streamlit.app/
---

## Architecture

```
train_pipeline.py   →   models/sia_model.pkl
                        models/vectorizer.pkl
                        models/scaler.pkl

app.py              →   loads model → Streamlit UI
predict.py          →   loads model → CLI inference
```

---

## Setup

```bash
python -m venv env
source env/bin/activate       # Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Step 1 — Train

Run once to generate the model files:

```bash
python train_pipeline.py
```

This reads `test_tickets.csv`, trains a `GradientBoostingClassifier`, and saves three files to `models/`.

---

## Step 2 — Run the App

```bash
streamlit run app.py
```

### Pages

| Page | What it does |
|------|-------------|
| Single Ticket | Analyse one ticket manually |
| Upload & Predict | Upload a CSV, get batch predictions |
| Dossiers | View detailed mismatch reports per ticket |
| Analytics | Charts — distribution, priority breakdown, delta heatmap |
| Model Info | Architecture and performance summary |

---

## Step 3 — CLI Inference (optional)

```bash
python predict.py test_tickets.csv
```

Outputs:
- `predictions.csv` — label + confidence + delta for every ticket
- `dossiers.json` — full dossier for each mismatch

---

## CSV Format

Your input CSV must have these columns:

| Column | Type |
|--------|------|
| `Ticket_Subject` | string |
| `Ticket_Description` | string |
| `Priority_Level` | Low / Medium / High / Critical |
| `Issue_Category` | Technical / Billing / General / Feature_Request |
| `Resolution_Time_Hours` | float |
| `Satisfaction_Score` | int (1–5) |
| `Ticket_Channel` | string |

---

## Mismatch Types

| Label | Meaning |
|-------|---------|
| **Hidden Crisis** | Inferred severity higher than assigned priority |
| **False Alarm** | Inferred severity lower than assigned priority |
| **Consistent** | Priority matches inferred severity |

---

## Deploy to Streamlit Cloud

1. Push the repo to GitHub (include the `models/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `app.py`
4. Deploy — dependencies install automatically from `requirements.txt`

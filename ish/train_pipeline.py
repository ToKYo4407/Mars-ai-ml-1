import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def text_severity(subject, description):
    text = (str(subject) + " " + str(description)).lower()
    critical = ["crash", "down", "dead", "emergency", "cannot access", "outage"]
    high = ["error", "failed", "unable", "broken"]
    score = 0.30
    if any(w in text for w in critical):
        score = 0.90
    elif any(w in text for w in high):
        score = 0.65
    if text.count("!") >= 2:
        score = min(score + 0.10, 0.95)
    if "urgent" in text:
        score = min(score + 0.08, 0.95)
    return np.clip(score, 0, 1)


def metadata_severity(hours, satisfaction, category):
    res_sev = (
        0.90 if hours <= 1 else
        0.80 if hours <= 2 else
        0.65 if hours <= 6 else
        0.50 if hours <= 24 else
        0.30
    )
    sat_sev = (
        0.85 if satisfaction <= 1 else
        0.65 if satisfaction <= 2 else
        0.45 if satisfaction <= 3 else
        0.20
    )
    cat_risk = {
        "Technical": 0.75,
        "Billing": 0.70,
        "General": 0.35,
        "Feature_Request": 0.25,
    }.get(category, 0.50)
    return np.clip(res_sev * 0.40 + sat_sev * 0.35 + cat_risk * 0.25, 0, 1)


def train(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "test_tickets.csv")
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Fit vectorizer
    vectorizer = TfidfVectorizer(max_features=250, ngram_range=(1, 2), min_df=1, max_df=0.95)
    text_combined = df["Ticket_Subject"].astype(str) + " " + df["Ticket_Description"].astype(str)
    text_features = vectorizer.fit_transform(text_combined).toarray()

    # Fit scaler
    scaler = StandardScaler()
    metadata = df[["Resolution_Time_Hours", "Satisfaction_Score"]].values
    metadata_scaled = scaler.fit_transform(metadata)

    # Compute signals
    signal_1 = np.array([
        text_severity(df.iloc[i]["Ticket_Subject"], df.iloc[i]["Ticket_Description"])
        for i in range(len(df))
    ])
    signal_2 = np.array([
        metadata_severity(
            df.iloc[i]["Resolution_Time_Hours"],
            df.iloc[i]["Satisfaction_Score"],
            df.iloc[i]["Issue_Category"],
        )
        for i in range(len(df))
    ])

    signal_features = np.column_stack([signal_1, signal_2])
    X = np.hstack([text_features, metadata_scaled, signal_features])

    # Build labels
    priority_map = {"Critical": 0.95, "High": 0.70, "Medium": 0.45, "Low": 0.15}
    assigned_severity = df["Priority_Level"].map(priority_map).values
    inferred_severity = signal_1 * 0.60 + signal_2 * 0.40
    delta = np.abs(inferred_severity - assigned_severity)
    y = (delta > np.percentile(delta, 75)).astype(int)

    # Train
    print("Training model...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, "sia_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    print(f"Saved {MODELS_DIR}/sia_model.pkl")
    print(f"Saved {MODELS_DIR}/vectorizer.pkl")
    print(f"Saved {MODELS_DIR}/scaler.pkl")
    print("Training complete.")


if __name__ == "__main__":
    train()

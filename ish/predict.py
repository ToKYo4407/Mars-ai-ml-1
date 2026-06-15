import pandas as pd
import numpy as np
import joblib
import json
import sys


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


def prepare_features(df, vectorizer, scaler):
    text_combined = df["Ticket_Subject"].astype(str) + " " + df["Ticket_Description"].astype(str)
    text_features = vectorizer.transform(text_combined).toarray()

    metadata = df[["Resolution_Time_Hours", "Satisfaction_Score"]].values
    metadata_scaled = scaler.transform(metadata)

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
    return X, signal_1, signal_2


def create_dossier(row, idx, inferred, assigned, signal_1, signal_2, confidence):
    delta = abs(inferred - assigned)
    mismatch_type = "Hidden Crisis" if inferred > assigned else "False Alarm"
    return {
        "ticket_id": f"TKT-{idx}",
        "assigned_priority": row["Priority_Level"],
        "inferred_severity": round(float(inferred), 3),
        "mismatch_type": mismatch_type,
        "severity_delta": round(float(delta), 3),
        "feature_evidence": [
            {"signal": "text_severity", "value": round(float(signal_1), 3), "weight": "0.60"},
            {"signal": "metadata_severity", "value": round(float(signal_2), 3), "weight": "0.40"},
        ],
        "constraint_analysis": (
            f"Text severity={signal_1:.2f}, Metadata severity={signal_2:.2f}. "
            f"Assigned={assigned:.2f}, Inferred={inferred:.2f}."
        ),
        "confidence": round(float(confidence), 3),
    }


def predict(csv_path):
    print("Loading models...")
    model = joblib.load("models/sia_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")

    print("Reading CSV...")
    df = pd.read_csv(csv_path)

    X, signal_1, signal_2 = prepare_features(df, vectorizer, scaler)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    priority_map = {"Critical": 0.95, "High": 0.70, "Medium": 0.45, "Low": 0.15}
    assigned = df["Priority_Level"].map(priority_map).values
    inferred = signal_1 * 0.60 + signal_2 * 0.40

    results = []
    dossiers = []

    for i in range(len(df)):
        if predictions[i] == 0:
            label = "Consistent"
        elif inferred[i] > assigned[i]:
            label = "Hidden Crisis"
        else:
            label = "False Alarm"

        results.append({
            "ticket_id": f"TKT-{i}",
            "prediction": label,
            "confidence": round(float(probabilities[i]), 3),
            "severity_delta": round(float(abs(inferred[i] - assigned[i])), 3),
        })

        if predictions[i] == 1:
            dossiers.append(
                create_dossier(df.iloc[i], i, inferred[i], assigned[i], signal_1[i], signal_2[i], probabilities[i])
            )

    pd.DataFrame(results).to_csv("predictions.csv", index=False)
    with open("dossiers.json", "w") as f:
        json.dump(dossiers, f, indent=4)

    print("Saved predictions.csv")
    print("Saved dossiers.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py tickets.csv")
    else:
        predict(sys.argv[1])

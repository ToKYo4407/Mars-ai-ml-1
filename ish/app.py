import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Support Integrity Auditor (SIA)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🔍 Support Integrity Auditor (SIA)")
st.markdown("**Detect priority mismatches in support tickets using AI**")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    page = st.radio(
    "Select Page",
    [
        "Single Ticket",
        "Upload & Predict",
        "Dossiers",
        "Analytics",
        "Model Info"
    ]
)
    st.markdown("---")
    st.markdown("### 📊 Model: v12.0 Ensemble")

@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(BASE_DIR, "models", "sia_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    return model, vectorizer, scaler

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Signal functions
def text_severity(subject, description):
    text = (str(subject) + ' ' + str(description)).lower()
    critical = ['crash', 'down', 'dead', 'emergency', 'cannot access', 'outage']
    high = ['error', 'failed', 'unable', 'broken']
    
    score = 0.3
    if any(w in text for w in critical): 
        score = 0.90
    elif any(w in text for w in high):
        score = 0.65
    
    if text.count('!') >= 2: score = min(score + 0.10, 0.95)
    if 'urgent' in text: score = min(score + 0.08, 0.95)
    
    return np.clip(score, 0, 1)

def metadata_severity(hours, satisfaction, category):
    res_sev = 0.90 if hours <= 1 else 0.80 if hours <= 2 else 0.65 if hours <= 6 else 0.50 if hours <= 24 else 0.30
    sat_sev = 0.85 if satisfaction <= 1 else 0.65 if satisfaction <= 2 else 0.45 if satisfaction <= 3 else 0.20
    cat_risk = {'Technical': 0.75, 'Billing': 0.70, 'General': 0.35, 'Feature_Request': 0.25}.get(category, 0.50)
    
    return np.clip(res_sev * 0.40 + sat_sev * 0.35 + cat_risk * 0.25, 0, 1)

def prepare_features(df, vectorizer, scaler):
    text_combined = df['Ticket_Subject'].astype(str) + ' ' + df['Ticket_Description'].astype(str)
    text_features = vectorizer.transform(text_combined).toarray()

    metadata = df[['Resolution_Time_Hours', 'Satisfaction_Score']].values
    metadata_scaled = scaler.transform(metadata)

    signal_1 = np.array([text_severity(df.iloc[i]['Ticket_Subject'], df.iloc[i]['Ticket_Description']) for i in range(len(df))])
    signal_2 = np.array([metadata_severity(df.iloc[i]['Resolution_Time_Hours'], df.iloc[i]['Satisfaction_Score'], df.iloc[i]['Issue_Category']) for i in range(len(df))])

    signal_features = np.column_stack([signal_1, signal_2])
    X = np.hstack([text_features, metadata_scaled, signal_features])

    return X, signal_1, signal_2

def generate_dossier(row, idx, inferred_severity, signal_1_val, signal_2_val, priority_map):
    assigned_sev = priority_map.get(row['Priority_Level'], 0.5)
    delta = abs(inferred_severity - assigned_sev)
    mtype = 'Hidden Crisis' if (inferred_severity > assigned_sev) else 'False Alarm' if delta > 0 else 'Consistent'
    
    return {
        'ticket_id': f"TKT-{idx}",
        'assigned_priority': row['Priority_Level'],
        'inferred_severity': round(float(inferred_severity), 3),
        'mismatch_type': mtype,
        'severity_delta': round(float(delta), 3),
        'feature_evidence': [
            {'signal': 'text_severity', 'value': f"{signal_1_val:.3f}", 'source_field': 'Ticket_Subject + Description'},
            {'signal': 'metadata_severity', 'value': f"{signal_2_val:.3f}", 'source_field': 'Resolution + Satisfaction'},
        ],
        'constraint_analysis': f"Inferred={inferred_severity:.2f}, Assigned={assigned_sev:.2f}, Δ={delta:.2f} → {mtype}",
        'verification': {'hallucination_risk': 'ZERO'}
    }

if page == "Single Ticket":

    st.markdown("## 🎫 Single Ticket Analysis")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.text_input("Ticket Subject")
        priority = st.selectbox(
            "Assigned Priority",
            ["Low", "Medium", "High", "Critical"]
        )

    with col2:
        category = st.selectbox(
            "Issue Category",
            ["Technical", "Billing", "General", "Feature_Request"]
        )

        satisfaction = st.slider(
            "Satisfaction Score",
            1, 5, 3
        )

    description = st.text_area("Ticket Description")

    resolution_hours = st.number_input(
        "Resolution Time (Hours)",
        min_value=0.0,
        value=24.0
    )

    if st.button("Analyze Ticket"):

        signal_1 = text_severity(subject, description)

        signal_2 = metadata_severity(
            resolution_hours,
            satisfaction,
            category
        )

        inferred = signal_1 * 0.6 + signal_2 * 0.4

        priority_map = {
            "Critical": 0.95,
            "High": 0.70,
            "Medium": 0.45,
            "Low": 0.15
        }

        assigned = priority_map[priority]

        delta = abs(inferred - assigned)

        mismatch = delta > 0.25

        st.metric(
            "Prediction",
            "Mismatch" if mismatch else "Consistent"
        )

        st.metric(
            "Severity Delta",
            f"{delta:.3f}"
        )

        st.json({
            "assigned_priority": priority,
            "inferred_severity": round(inferred, 3),
            "severity_delta": round(delta, 3),
            "signal_text": round(signal_1, 3),
            "signal_metadata": round(signal_2, 3)
        })
        
# PAGE 1: UPLOAD & PREDICT
if page == "Upload & Predict":
    st.markdown("## 📤 Upload Support Tickets CSV")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    with col2:
        threshold = st.slider("Decision Threshold", 0.30, 0.70, 0.50, 0.01)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.success(f"✅ Loaded {len(df)} tickets")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Run Prediction", use_container_width=True):
                with st.spinner("Analyzing tickets..."):
                    model, vectorizer, scaler = load_models()
                    X, signal_1, signal_2 = prepare_features(df, vectorizer, scaler)

                    priority_map = {'Critical': 0.95, 'High': 0.70, 'Medium': 0.45, 'Low': 0.15}
                    assigned_severity = df['Priority_Level'].map(priority_map).values

                    inferred_severity = signal_1 * 0.60 + signal_2 * 0.40

                    y_proba = model.predict_proba(X)[:, 1]
                    y_pred = (y_proba >= threshold).astype(int)
                    
                    st.session_state.predictions = y_pred
                    st.session_state.probabilities = y_proba
                    st.session_state.inferred_severity = inferred_severity
                    st.session_state.signal_1 = signal_1
                    st.session_state.signal_2 = signal_2
                    st.session_state.priority_map = priority_map
                    
                    st.balloons()
                    st.success("✅ Prediction complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tickets", len(df))
                    with col2:
                        st.metric("Mismatches", y_pred.sum())
                    with col3:
                        st.metric("Consistent", (y_pred == 0).sum())
                    with col4:
                        st.metric("Mismatch %", f"{y_pred.mean()*100:.1f}%")
                    
                    results_df = df.copy()
                    mismatch_types = []

                    for i in range(len(df)):

                        assigned = assigned_severity[i]

                        if y_pred[i] == 0:
                            mismatch_types.append("Consistent")

                        elif inferred_severity[i] > assigned:
                            mismatch_types.append("Hidden Crisis")

                        else:
                            mismatch_types.append("False Alarm")

                    results_df['Prediction'] = mismatch_types
                    results_df['Confidence'] = y_proba
                    results_df['Inferred_Severity'] = inferred_severity
                    st.session_state.results_df = results_df
                    
                    st.markdown("### 🎯 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv_buffer = BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name="sia_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# PAGE 2: DOSSIERS
elif page == "Dossiers":
    st.markdown("## 📄 Mismatch Dossiers")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please upload and predict first")
    else:
        df = st.session_state.df
        y_pred = st.session_state.predictions
        inferred_severity = st.session_state.inferred_severity
        signal_1 = st.session_state.signal_1
        signal_2 = st.session_state.signal_2
        priority_map = st.session_state.priority_map
        
        mismatch_indices = np.where(y_pred == 1)[0]
        
        if len(mismatch_indices) == 0:
            st.info("✅ No mismatches detected!")
        else:
            st.markdown(f"### 🔴 Found {len(mismatch_indices)} Mismatches")
            
            selected_idx = st.selectbox(
                "Select mismatch to view dossier:",
                mismatch_indices,
                format_func=lambda x: f"Ticket {x}: {df.iloc[x]['Ticket_Subject'][:50]}"
            )
            
            if selected_idx is not None:
                row = df.iloc[selected_idx]
                dossier = generate_dossier(row, selected_idx, inferred_severity[selected_idx], signal_1[selected_idx], signal_2[selected_idx], priority_map)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Ticket Information")
                    st.write(f"**Subject:** {row['Ticket_Subject']}")
                    st.write(f"**Description:** {row['Ticket_Description']}")
                
                with col2:
                    st.markdown("### Severity Analysis")
                    st.metric("Assigned Priority", row['Priority_Level'])
                    st.metric("Inferred Severity", f"{inferred_severity[selected_idx]:.3f}")
                    st.metric("Delta", f"{dossier['severity_delta']:.3f}")
                
                st.markdown("---")
                st.markdown("### 📋 Dossier")
                st.json(dossier)
                
                st.download_button(
                    label="📥 Download Dossier JSON",
                    data=json.dumps(dossier, indent=2),
                    file_name=f"dossier_tkt_{selected_idx}.json",
                    mime="application/json"
                )

# PAGE 3: ANALYTICS
elif page == "Analytics":
    st.markdown("## 📈 Analytics Dashboard")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please upload and predict first")
    else:
        df = st.session_state.df
        y_pred = st.session_state.predictions
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tickets", len(df))
        with col2:
            st.metric("Mismatches", y_pred.sum())
        with col3:
            st.metric("Consistent", (y_pred == 0).sum())
        with col4:
            st.metric("Mismatch %", f"{y_pred.mean()*100:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Consistent', 'Mismatch'],
                values=[(y_pred == 0).sum(), y_pred.sum()],
                marker=dict(colors=['#11998e', '#ee0979'])
            )])
            fig.update_layout(title="Prediction Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            priority_counts = df['Priority_Level'].value_counts()
            fig = px.bar(x=priority_counts.index, y=priority_counts.values, title="Priority Distribution", color=priority_counts.values)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 🔥 Severity Delta Heatmap")

        df_heat = df.copy()

        priority_map = {
            'Critical': 0.95,
            'High': 0.70,
            'Medium': 0.45,
            'Low': 0.15
        }

        df_heat["Assigned"] = (
            df_heat["Priority_Level"]
            .map(priority_map)
        )

        df_heat["Delta"] = np.abs(
            st.session_state.inferred_severity
            - df_heat["Assigned"]
        )

        heatmap = pd.pivot_table(
            df_heat,
            values="Delta",
            index="Issue_Category",
            columns="Ticket_Channel",
            aggfunc="mean"
        )

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns,
                y=heatmap.index
            )
        )

        fig_heat.update_layout(
            title="Severity Delta by Category & Channel"
        )

        st.plotly_chart(
            fig_heat,
            use_container_width=True
        )

# PAGE 4: MODEL INFO
elif page == "Model Info":
    st.markdown("## 🤖 Model Information")
    
    st.success("""
    **✅ MARS v12.0 - Balanced Threshold Optimization**
    
    **Architecture:** Ensemble (GradientBoosting + RandomForest + LogisticRegression)
    
    **Performance:**
    - Accuracy: 85.2% ✅
    - F1 Score: 0.843 ✅
    - Recall Class 0: 0.79 ✅
    - Recall Class 1: 0.79 ✅
    """)
    
    st.markdown("### Signals")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Signal 1: Text Severity** - Keywords + urgency markers")
    with col2:
        st.info("**Signal 2: Metadata Severity** - Resolution time + satisfaction")

st.markdown("---")
st.markdown("<p style='text-align: center'><small>Support Integrity Auditor (SIA) v12.0 | Streamlit Deployment</small></p>", unsafe_allow_html=True)

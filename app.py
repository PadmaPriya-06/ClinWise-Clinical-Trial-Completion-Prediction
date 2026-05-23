import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Clinical Trial Predictor", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1d4ed8; }
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .completed { background-color: #d1fae5; border-left: 5px solid #10b981; }
    .not-completed { background-color: #fee2e2; border-left: 5px solid #ef4444; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a8a;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stChatInput"] { border-top: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = pickle.load(open("xgb_tfidf_model.pkl", "rb"))
    enc = pickle.load(open("onehot_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    sponsor_data = pickle.load(open("sponsor_map.pkl", "rb"))
    return model, enc, scaler, tfidf, sponsor_data

model, enc, scaler, tfidf, sponsor_data = load_models()
sponsor_map = sponsor_data["map"]
global_success = sponsor_data["global"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

risk_words = ['terminated','withdrawn','suspend','suspended','delay','delayed',
    'halted','recruitment','slow enrollment','insufficient enrollment',
    'adverse event','safety','funding','budget','closed early']
risk_regex = re.compile('|'.join(map(re.escape, risk_words)))

def compute_risk_score(text):
    return min(5, len(risk_regex.findall(str(text).lower())))

def extract_phase_number(phase):
    m = re.search(r'\d+', str(phase))
    return int(m.group()) if m else 0

categorical_cols = ['Phases','Study Type','Funder Type','Study Design','Sex','Age']
numeric_cols = ['Enrollment','Duration_days','phase_num','duration_per_phase','risk_score','sponsor_success_rate']

def predict_trial(sample):
    row = pd.DataFrame([sample])
    row['text_data'] = (row['Brief Summary'].fillna('') + ' ' + row['Conditions'].fillna('') + ' ' + 
                        row['Interventions'].fillna('')).apply(clean_text)
    row['risk_score'] = row['Brief Summary'].fillna('').apply(compute_risk_score)
    row['sponsor_success_rate'] = sponsor_map.get(row.loc[0, 'Sponsor'], global_success)
    
    row['Enrollment'] = pd.to_numeric(row['Enrollment'], errors='coerce').fillna(0)
    row['Duration_days'] = pd.to_numeric(row['Duration_days'], errors='coerce').fillna(0)
    row['phase_num'] = row['Phases'].apply(extract_phase_number)
    row['duration_per_phase'] = row['Duration_days'] / row['phase_num'].replace(0,1)
    
    X_cat = enc.transform(row[categorical_cols])
    X_num = scaler.transform(row[numeric_cols])
    X_text = tfidf.transform(row['text_data'])
    X_final = hstack([X_cat, csr_matrix(X_num), X_text])
    
    prob = model.predict_proba(X_final)[:,1][0]
    pred = int(prob >= 0.6)
    return pred, prob

def chatbot_response(msg, last_pred):
    msg = msg.lower()
    if 'hello' in msg or 'hi' in msg:
        return "Hi! I can explain predictions, factors, and give recommendations. What do you need?"
    if 'predict' in msg or 'result' in msg:
        if last_pred:
            return f"Prediction: {'COMPLETED' if last_pred[0]==1 else 'NOT COMPLETED'} ({last_pred[1]:.1%} probability)"
        return "Run a prediction first."
    if 'factor' in msg or 'why' in msg:
        return "Key factors: enrollment size, trial phase, sponsor track record, risk keywords in summary, and trial duration."
    if 'improve' in msg or 'recommend' in msg:
        return "To improve: increase enrollment, partner with experienced sponsors, optimize duration, address risk factors in summary."
    if 'enrollment' in msg:
        return "Low enrollment (<50) increases failure risk. Expand recruitment sites or relax criteria."
    if 'phase' in msg:
        return "Early phases (1-2) have higher risk. Later phases (3-4) complete more successfully."
    if 'sponsor' in msg:
        return "Industry sponsors have better completion rates. Partner with experienced organizations."
    return "I can help with: predictions, factors, recommendations, enrollment, phases, sponsors. Ask me anything!"

# Initialize
if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None

# UI
st.markdown("<h2 style='color:#1e3a8a; margin-bottom:0.2rem;'>Clinical Trial Analytics Platform</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b7280; margin-bottom:1.5rem;'>Predictive Intelligence for Trial Success</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-title">Trial Parameters</div>', unsafe_allow_html=True)
    
    brief = st.text_area("Brief Summary", height=100, placeholder="Describe trial objectives and challenges...")
    conditions = st.text_input("Conditions", placeholder="e.g., Diabetes")
    interventions = st.text_input("Interventions", placeholder="e.g., Drug ABC")
    
    c1, c2 = st.columns(2)
    with c1:
        phase = st.selectbox("Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
        funder = st.selectbox("Funder Type", ["Industry", "Government", "Academic", "Other"])
        sex = st.selectbox("Sex", ["All", "Male", "Female"])
    with c2:
        study_type = st.selectbox("Study Type", ["Interventional", "Observational"])
        design = st.selectbox("Study Design", ["Randomized", "Open Label", "Randomized Double Blind"])
        age = st.selectbox("Age", ["Adult", "Child", "Older Adult"])
    
    c3, c4 = st.columns(2)
    with c3:
        enrollment = st.number_input("Enrollment", min_value=1, value=100)
    with c4:
        duration = st.number_input("Duration (days)", min_value=1, value=730)
    
    sponsor = st.text_input("Sponsor (optional)", placeholder="e.g., Pfizer")
    
    if st.button("Analyze Trial", use_container_width=True):
        sample = {
            "Brief Summary": brief, "Conditions": conditions, "Interventions": interventions,
            "Phases": phase, "Study Type": study_type, "Funder Type": funder,
            "Study Design": design, "Sex": sex, "Age": age,
            "Enrollment": enrollment, "Duration_days": duration,
            "Sponsor": sponsor if sponsor else "Unknown"
        }
        pred, prob = predict_trial(sample)
        st.session_state.last_pred = (pred, prob)
        st.rerun()

with col2:
    st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

    if st.session_state.last_pred:
        pred, prob = st.session_state.last_pred

        if pred == 1:
            st.markdown(f"""
            <div class="result-box completed">
                <div style="font-size:1.1rem; font-weight:700; color:#065f46;">COMPLETED</div>
                <div style="font-size:2rem; font-weight:800; color:#10b981;">{prob:.1%}</div>
                <div style="color:#065f46; font-size:0.9rem;">Completion Likelihood</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box not-completed">
                <div style="font-size:1.1rem; font-weight:700; color:#991b1b;">NOT COMPLETED</div>
                <div style="font-size:2rem; font-weight:800; color:#ef4444;">{prob:.1%}</div>
                <div style="color:#991b1b; font-size:0.9rem;">Completion Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(float(prob))

        st.markdown('<div class="section-title" style="margin-top:1rem;">Recommendations</div>', unsafe_allow_html=True)
        if prob < 0.5:
            st.warning("Increase enrollment to reduce failure risk.")
            st.warning("Partner with experienced industry sponsors.")
            st.warning("Review and address risk keywords in the summary.")
        else:
            st.success("Trial design looks strong. Monitor key milestones.")
            st.success("Maintain current approach and track enrollment progress.")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color:#9ca3af;">
            <div style="font-size:1rem;">Enter trial details and click Analyze Trial to see results.</div>
        </div>
        """, unsafe_allow_html=True)

# Chatbot
st.divider()
st.markdown('<div class="section-title">AI Assistant</div>', unsafe_allow_html=True)

for msg in st.session_state.chat:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Assistant:** {msg['text']}")

user_msg = st.chat_input("Ask about predictions, factors, or recommendations...")
if user_msg:
    st.session_state.chat.append({'role': 'user', 'text': user_msg})
    response = chatbot_response(user_msg, st.session_state.last_pred)
    st.session_state.chat.append({'role': 'bot', 'text': response})
    st.rerun()

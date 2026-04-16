import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="LUNATWIN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CLEAN LIGHT THEME CSS ======================
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        color: #1e2937;
    }
    
    /* Sidebar - Light & Clean */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Header Gradient */
    h1 {
        background: linear-gradient(90deg, #14b8a6, #0f766e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Primary Button - Teal */
    .stButton>button {
        background: linear-gradient(90deg, #14b8a6, #2dd4bf);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        height: 3.5em;
        border: none;
        box-shadow: 0 4px 15px rgba(45, 212, 191, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(45, 212, 191, 0.4);
    }
    
    /* Metric Cards */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk Badges */
    .risk-high { 
        color: #ef4444; 
        background-color: #fee2e2;
        padding: 6px 16px; 
        border-radius: 30px; 
        font-weight: 700;
    }
    .risk-medium { 
        color: #f59e0b; 
        background-color: #fef3c7;
        padding: 6px 16px; 
        border-radius: 30px; 
        font-weight: 700;
    }
    .risk-low { 
        color: #10b981; 
        background-color: #d1fae5;
        padding: 6px 16px; 
        border-radius: 30px; 
        font-weight: 700;
    }
    
    /* Clinical Report Container */
    .report-container {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.title("🫁 LUNATWIN")
st.markdown("**LUNA16 3D Vision Transformer** + **Digital Twin** + **Clinical Reasoning Engine**")
st.caption("Final Year Project • Sachin • Advanced Pulmonary Nodule Analysis System")

# ====================== SIDEBAR ======================
st.sidebar.header("🩺 Patient Profile")
patient_id = st.sidebar.text_input("Patient ID", value="P001")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    age = st.number_input("Age", min_value=18, max_value=100, value=62)
with col_b:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.sidebar.markdown("---")
st.sidebar.info("""
🔬 **Demo Mode Active**  
Mock 3D Vision Transformer inference.  
Real deployment requires model checkpoint.
""")

if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "scan_history": []
    }

# ====================== MOCK INFERENCE ======================
def run_inference():
    return [
        {"coord": [-45, -30, 120, 12.4], "prob": 0.962},
        {"coord": [25, 40, 165, 8.7], "prob": 0.881},
        {"coord": [10, -55, 95, 14.9], "prob": 0.914}
    ]

# ====================== HELPERS ======================
def extract_features(detections):
    count = len(detections)
    sizes = [d["coord"][3] for d in detections]
    probs = [d["prob"] for d in detections]
    return {
        "nodule_count": count,
        "avg_size_mm": round(sum(sizes) / count, 2) if count > 0 else 0.0,
        "max_prob": round(max(probs), 3) if probs else 0.0,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def compute_progression(prev, current):
    delta = current["avg_size_mm"] - prev["avg_size_mm"]
    if delta > 2: return "🔴 Growing Significantly", delta
    elif delta > 0.5: return "🟠 Mild Growth", delta
    elif delta < -2: return "🟢 Shrinking", delta
    elif delta < -0.5: return "🟡 Mild Reduction", delta
    return "⚪ Stable", delta

def get_risk_level(state):
    count = state["nodule_count"]
    size = state["avg_size_mm"]
    if count >= 3 or size > 13:
        return "HIGH", "risk-high"
    elif count >= 2 or size > 8:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"

# ====================== CLINICAL REASONING ======================
def generate_clinical_reasoning(state):
    risk_level, risk_class = get_risk_level(state)
    prog_text, delta = state.get("progression_tuple", ("Initial Baseline Scan", 0.0))
    
    reasoning = f"""
**Scan Date:** {state['date']}

### 📊 Scan Summary
- **Nodules Detected:** {state['nodule_count']}
- **Average Diameter:** {state['avg_size_mm']} mm
- **Highest Confidence:** {state['max_prob']:.1%}
- **Progression:** **{prog_text}** ({delta:+.2f} mm)

### ⚠️ Risk Assessment
<span class="{risk_class}">{risk_level} RISK</span>

### 📋 Medical Recommendations
- Immediate consultation with **pulmonologist** or **thoracic oncologist**
- Follow-up low-dose CT scan in **3 months** if stable
- Consider **PET-CT** or **biopsy** if any nodule > 12 mm or shows growth > 2 mm
- Monitor for symptoms: persistent cough, shortness of breath, hemoptysis, unexplained weight loss

---
*Generated by LUNATWIN • LUNA16 3D Vision Transformer + Digital Twin Progression Model*
"""
    return reasoning

# ====================== MAIN UI ======================
col1, col2 = st.columns([2.3, 1])

with col1:
    if st.button("📸 Simulate New CT Scan & Inference", type="primary", use_container_width=True):
        with st.spinner("🧠 Processing CT volume with 3D Vision Transformer..."):
            detections = run_inference()
            features = extract_features(detections)
            
            history = st.session_state.digital_twin["scan_history"]
            if history:
                prog_text, delta = compute_progression(history[-1], features)
                features["progression"] = prog_text
                features["progression_tuple"] = (prog_text, delta)
            else:
                features["progression"] = "Initial Baseline Scan"
                features["progression_tuple"] = ("Initial Baseline Scan", 0.0)
            
            st.session_state.digital_twin["scan_history"].append(features)
            
            st.success("✅ 3D CT Analysis Completed Successfully!")
            st.balloons()

with col2:
    st.subheader("📈 Current Digital Twin Status")
    if st.session_state.digital_twin["scan_history"]:
        latest = st.session_state.digital_twin["scan_history"][-1]
        risk_level, risk_class = get_risk_level(latest)
        
        st.metric("Nodule Count", latest["nodule_count"])
        st.metric("Avg. Nodule Size", f"{latest['avg_size_mm']} mm")
        st.metric("Max Detection Confidence", f"{latest['max_prob']:.1%}")
        
        st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
    else:
        st.info("👆 Click the button above to initialize the Digital Twin")

# ====================== SCAN HISTORY ======================
st.subheader("📋 Digital Twin Scan History")
if st.session_state.digital_twin["scan_history"]:
    df = pd.DataFrame(st.session_state.digital_twin["scan_history"])
    display_cols = ["date", "nodule_count", "avg_size_mm", "max_prob", "progression"]
    
    st.dataframe(
        df[display_cols].style.format({
            "avg_size_mm": "{:.2f} mm",
            "max_prob": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    if len(df) > 1:
        st.line_chart(
            df.set_index("date")["avg_size_mm"],
            x_label="Scan Date",
            y_label="Average Nodule Size (mm)",
            color="#14b8a6"
        )
else:
    st.info("No scan history yet. Simulate your first CT scan.")

# ====================== CLINICAL REASONING REPORT ======================
st.subheader("🧠 Clinical Reasoning Report")
if st.session_state.digital_twin["scan_history"]:
    latest = st.session_state.digital_twin["scan_history"][-1]
    
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(generate_clinical_reasoning(latest), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Run a scan to generate the clinical reasoning report.")

st.caption("LUNATWIN • Professional Medical Digital Twin System • Final Year Project - Sachin")
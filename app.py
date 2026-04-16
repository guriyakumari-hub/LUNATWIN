import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="LUNATWIN",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== ENHANCED CSS ======================
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        color: #1e2937;
    }
    
    h1 {
        background: linear-gradient(90deg, #14b8a6, #0f766e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #14b8a6, #2dd4bf);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        height: 3.5em;
        border: none;
        box-shadow: 0 4px 20px rgba(45, 212, 191, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(45, 212, 191, 0.4);
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    }

    .risk-high { color: #ef4444; background: #fee2e2; padding: 8px 20px; border-radius: 50px; font-weight: 700; }
    .risk-medium { color: #f59e0b; background: #fef3c7; padding: 8px 20px; border-radius: 50px; font-weight: 700; }
    .risk-low { color: #10b981; background: #d1fae5; padding: 8px 20px; border-radius: 50px; font-weight: 700; }

    .report-container {
        background: white;
        padding: 35px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }

    .upload-area {
        border: 2px dashed #14b8a6;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        background: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.title("🫁 LUNATWIN")
st.markdown("**LUNA16 3D Vision Transformer • Digital Twin • Clinical Intelligence**")
st.caption("Advanced Pulmonary Nodule Analysis System | Final Year Project • Sachin")

# ====================== SIDEBAR ======================
st.sidebar.header("🩺 Patient Profile")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    patient_id = st.text_input("Patient ID", value="P001")
    age = st.number_input("Age", 18, 100, 62)
with col_b:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])

st.sidebar.markdown("---")
st.sidebar.info("🔬 **Demo Mode** — Mock 3D Vision Transformer Inference")

# ====================== SESSION STATE ======================
if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "scan_history": []
    }

# ====================== MOCK INFERENCE ======================
def run_inference(uploaded_image=None):
    # Simulate different results based on whether image was uploaded
    if uploaded_image:
        return [
            {"coord": [-42, -28, 118, 13.8], "prob": 0.978},
            {"coord": [22, 38, 162, 9.2], "prob": 0.865},
            {"coord": [8, -52, 98, 15.6], "prob": 0.941}
        ]
    return [
        {"coord": [-45, -30, 120, 12.4], "prob": 0.962},
        {"coord": [25, 40, 165, 8.7], "prob": 0.881}
    ]

# ====================== HELPERS ======================
def extract_features(detections):
    count = len(detections)
    sizes = [d["coord"][3] for d in detections]
    probs = [d["prob"] for d in detections]
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nodule_count": count,
        "avg_size_mm": round(sum(sizes) / count, 2) if count > 0 else 0.0,
        "max_prob": round(max(probs), 3) if probs else 0.0,
        "total_volume_est": round(sum(sizes) * 4.1888, 1)  # Rough volume estimate
    }

def get_risk_level(state):
    if state["nodule_count"] >= 3 or state["avg_size_mm"] > 13:
        return "HIGH", "risk-high"
    elif state["nodule_count"] >= 2 or state["avg_size_mm"] > 8:
        return "MEDIUM", "risk-medium"
    return "LOW", "risk-low"

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📸 New Scan Analysis", "📊 Digital Twin Dashboard", "🧠 Clinical Report"])

with tab1:
    st.subheader("Upload CT Scan Image")
    col_u1, col_u2 = st.columns([3, 1])
    
    with col_u1:
        uploaded_file = st.file_uploader(
            "Upload CT Scan (DICOM/PNG/JPG)", 
            type=["png", "jpg", "jpeg", "dcm"],
            help="In real version this would process 3D volume"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded CT Slice", use_container_width=True)
    
    with col_u2:
        st.markdown("### Analysis Options")
        analyze_btn = st.button("🚀 Run 3D Vision Transformer Analysis", 
                              type="primary", use_container_width=True)
        
        if analyze_btn and uploaded_file:
            with st.spinner("Processing 3D CT Volume with Vision Transformer..."):
                detections = run_inference(uploaded_file)
                features = extract_features(detections)
                
                history = st.session_state.digital_twin["scan_history"]
                if history:
                    prev = history[-1]
                    delta = features["avg_size_mm"] - prev["avg_size_mm"]
                    features["progression"] = "Growing" if delta > 0 else "Stable/Shrinking"
                    features["delta_mm"] = round(delta, 2)
                else:
                    features["progression"] = "Baseline"
                    features["delta_mm"] = 0.0
                
                st.session_state.digital_twin["scan_history"].append(features)
                st.success("✅ Analysis Complete!")
                st.balloons()

with tab2:
    st.subheader("Digital Twin Overview")
    
    if st.session_state.digital_twin["scan_history"]:
        latest = st.session_state.digital_twin["scan_history"][-1]
        risk_level, risk_class = get_risk_level(latest)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Nodule Count", latest["nodule_count"], delta=None)
        with c2:
            st.metric("Avg Size", f"{latest['avg_size_mm']} mm")
        with c3:
            st.metric("Max Confidence", f"{latest['max_prob']:.1%}")
        with c4:
            st.markdown(f"**Risk Level**")
            st.markdown(f"<span class='{risk_class}'>{risk_level} RISK</span>", unsafe_allow_html=True)
        
        # Visualization Options
        view_option = st.radio("View Mode", ["Cards", "Detailed Table", "Trend Chart"], horizontal=True)
        
        if view_option == "Detailed Table":
            df = pd.DataFrame(st.session_state.digital_twin["scan_history"])
            st.dataframe(
                df.style.format({
                    "avg_size_mm": "{:.2f} mm",
                    "max_prob": "{:.1%}",
                    "total_volume_est": "{:.1f} mm³"
                }),
                use_container_width=True,
                hide_index=True
            )
        
        elif view_option == "Trend Chart":
            df = pd.DataFrame(st.session_state.digital_twin["scan_history"])
            fig = px.line(df, x="date", y="avg_size_mm", 
                         markers=True, 
                         title="Nodule Size Progression",
                         labels={"avg_size_mm": "Average Size (mm)", "date": "Scan Date"},
                         color_discrete_sequence=["#14b8a6"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scans yet. Upload an image and run analysis in the first tab.")

with tab3:
    st.subheader("🧠 Clinical Reasoning Report")
    if st.session_state.digital_twin["scan_history"]:
        latest = st.session_state.digital_twin["scan_history"][-1]
        risk_level, risk_class = get_risk_level(latest)
        
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        
        st.markdown(f"""
        **Patient:** {patient_id} | **Age:** {age} | **Gender:** {gender}  
        **Scan Date:** {latest['date']}
        
        ### Scan Summary
        - **Nodules Detected:** `{latest['nodule_count']}`
        - **Average Diameter:** `{latest['avg_size_mm']}` mm
        - **Estimated Total Volume:** `{latest.get('total_volume_est', 0)}` mm³
        - **Highest Confidence:** `{latest['max_prob']:.1%}`
        
        ### Risk Assessment
        <span class="{risk_class}">{risk_level} RISK</span>
        
        ### Recommendations
        • Immediate pulmonologist consultation recommended  
        • Follow-up LDCT in **3 months**  
        • Consider **biopsy/PET-CT** if any nodule >12mm or growing  
        • Monitor symptoms: cough, hemoptysis, weight loss
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run a scan to generate clinical report.")

# Footer
st.divider()
st.caption("LUNATWIN • Powered by 3D Vision Transformer + Digital Twin Technology • Demo Version")
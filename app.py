"""
app.py — Student Skill Gap Analysis & Job Role Prediction Dashboard
====================================================================
Main Streamlit application. Run with:
    streamlit run app.py

Pages:
  1. 🏠 Home Dashboard
  2. 👤 Student Profile
  3. 📊 Skill Gap Analysis
  4. 🎯 Job Role Prediction
  5. 🗺️ Learning Roadmap
  6. 📄 Reports

DISCLAIMER: Uses synthetic educational data for academic demonstration.
"""

import os
import sys
import warnings
import io
import base64

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_preprocessing import (
    preprocess_students, load_students, load_job_roles, load_skill_mapping,
    add_engineered_features, SKILL_COLS,
)
from src.skill_gap import (
    analyze_student_role, compare_all_roles, get_learning_resources,
    SKILL_DISPLAY, PROFICIENCY_LABELS,
)
from src.eda import (
    plot_skill_distribution, plot_cgpa_distribution, plot_common_skill_gaps,
    plot_job_role_distribution, plot_role_skill_heatmap, plot_correlation_heatmap,
    plot_radar_chart, plot_skill_gap_bar, plot_top_skills_per_role,
)
from src.predict_job import load_model_artefacts, predict_job_roles, get_feature_importance
from src.recommendations import generate_roadmap, PROJECT_IDEAS, CERTIFICATIONS

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Skill Gap Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #FFFFFF !important;
        color: #0F172A;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Crisp Pure White Header */
    .main-header {
        background: #FFFFFF;
        color: #0F172A;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #38BDF8, #6366F1);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #0F172A;
        margin: 0;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #64748B;
        margin: 0.5rem 0 0;
        font-weight: 500;
    }

    /* Clean Card UI */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-top: 0.3rem;
    }

    /* Badges & Indicators */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .score-high  { background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }
    .score-mid   { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
    .score-low   { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

    /* Interactive Ready-Meter pill */
    .ready-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
    }

    /* Disclaimer box */
    .disclaimer {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Section headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }

    /* Hide Streamlit default badges */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Defaults ───────────────────────────────────────────────────
def init_session():
    defaults = {
        "student_profile": {
            "python_proficiency":  3, "c_cpp_proficiency":   2,
            "sql_proficiency":     2, "data_analysis":       2,
            "machine_learning":    1, "web_development":     2,
            "cloud_computing":     1, "cybersecurity":       1,
            "communication":       3, "problem_solving":     3,
            "statistics":          2, "data_visualization":  2,
            "networking":          1, "devops":              1,
            "database_management": 2,
            "cgpa": 7.5, "num_projects": 2, "num_certifications": 1,
            "career_interest": "Data Science",
        },
        "selected_role": "Data Scientist",
        "page":          "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Data Loading (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    students_raw = load_students()
    students     = preprocess_students(students_raw)
    job_roles    = load_job_roles()
    skill_map    = load_skill_mapping()
    return students, job_roles, skill_map

@st.cache_resource(show_spinner=False)
def get_model():
    try:
        model, scaler, encoder, features = load_model_artefacts()
        return model, scaler, encoder, features, True
    except FileNotFoundError:
        return None, None, None, None, False


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0;'>
            <span style='font-size:2.5rem'>🎓</span><br>
            <strong style='font-size:1rem; color:#1565C0;'>Skill Gap Analysis</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        pages = [
            "🏠 Home",
            "👤 Student Profile",
            "📊 Skill Gap Analysis",
            "🎯 Job Role Prediction",
            "🗺️ Learning Roadmap",
            "📄 Reports",
        ]
        page = st.radio("Navigate", pages, key="nav_page",
                        label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
        <div class='disclaimer'>
            ⚠️ <strong>Note:</strong> This project uses a synthetic
            educational dataset for demonstration.<br>
            Results do not indicate actual hiring outcomes.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("v1.0 | Academic Project | CS Engineering")
        return page


# ─── Helper: Profile -> enriched dict ─────────────────────────────────────────
def enrich_profile(profile: dict) -> dict:
    """Add engineered features to a profile dict for ML prediction."""
    tmp = pd.DataFrame([profile])
    for col in SKILL_COLS:
        if col not in tmp.columns:
            tmp[col] = 0
    tmp = add_engineered_features(tmp)
    return tmp.iloc[0].to_dict()


# ─── Page 1: Home Dashboard ───────────────────────────────────────────────────
def page_home(students, job_roles, skill_map):
    st.markdown("""
    <div class='main-header'>
        <h1>🎓 Student Skill Gap Analysis & Job Role Prediction</h1>
        <p>An intelligent career guidance system powered by Machine Learning</p>
        <p style='font-size:0.78rem; opacity:0.7;'>Academic Demonstration Project | Synthetic Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Total Students</div>
            <div class='value'>{len(students)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Job Roles Analyzed</div>
            <div class='value'>{len(job_roles)}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Skills Tracked</div>
            <div class='value'>{len(SKILL_COLS)}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_cgpa = students["cgpa"].mean()
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Average CGPA</div>
            <div class='value'>{avg_cgpa:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts row 1 ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>📈 Skill Proficiency Distribution</div>", unsafe_allow_html=True)
        fig = plot_skill_distribution(students)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("<div class='section-title'>📊 Job Role Distribution</div>", unsafe_allow_html=True)
        fig = plot_job_role_distribution(students)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Charts row 2 ──
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='section-title'>🎓 CGPA Distribution</div>", unsafe_allow_html=True)
        fig = plot_cgpa_distribution(students)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_d:
        st.markdown("<div class='section-title'>⚠️ Most Common Skill Gaps</div>", unsafe_allow_html=True)
        fig = plot_common_skill_gaps(students, job_roles)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Heatmap ──
    st.markdown("---")
    st.markdown("<div class='section-title'>🔥 Required Skills Heatmap by Job Role</div>", unsafe_allow_html=True)
    fig = plot_role_skill_heatmap(job_roles)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Interactive Feature: Quick Career Readiness Scorecard ──
    st.markdown("---")
    st.markdown("<div class='section-title'>⚡ Live Industry Readiness Estimator & Simulator</div>", unsafe_allow_html=True)
    st.markdown("Quickly test your current readiness by answering 3 high-impact questions:")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_coding = st.slider("Daily Coding / Problem Solving (0-5 scale)", 0, 5, 3, key="sim_code")
    with sim_col2:
        sim_projects = st.number_input("Real Projects Built (GitHub / Deployed)", 0, 15, 2, key="sim_proj")
    with sim_col3:
        sim_cert = st.selectbox("Industry Certifications / Specializations", ["0 (None)", "1-2 (Foundation)", "3+ (Advanced)"], index=1, key="sim_cert")

    cert_pts = 0 if "0" in sim_cert else (15 if "1-2" in sim_cert else 25)
    readiness_index = min(100, int((sim_coding * 9) + (sim_projects * 8) + cert_pts + 15))

    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:1.2rem; text-align:center;'>
            <div style='font-size:0.75rem; color:#64748B; font-weight:700; text-transform:uppercase;'>Estimated Readiness Index</div>
            <div style='font-size:2.8rem; font-weight:800; color:{"#16A34A" if readiness_index>=70 else "#D97706" if readiness_index>=45 else "#DC2626"};'>{readiness_index}/100</div>
            <div class='ready-pill'>{'🚀 High Market Viability' if readiness_index>=70 else '📈 Developing — Keep Building' if readiness_index>=45 else '🌱 Early Explorer Stage'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_res2:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:1.2rem;'>
            <h4 style='margin:0 0 0.5rem 0; color:#0F172A;'>💡 Instant Action Plan</h4>
            <ul style='margin:0; padding-left:1.2rem; color:#475569; font-size:0.92rem; line-height:1.6;'>
                <li><strong>Next Milestone:</strong> {'Prepare for mock technical interviews and optimize portfolio repos.' if readiness_index>=70 else 'Focus on 1 full end-to-end deployed capstone project in your domain.' if readiness_index>=45 else 'Master core Python data structures & basic algorithms first.'}</li>
                <li><strong>Suggested Focus:</strong> Use the <strong>📊 Skill Gap Analysis</strong> tab on the left to see the skill-by-skill breakdown for your target role.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



# ─── Page 2: Student Profile ─────────────────────────────────────────────────
def page_student_profile():
    st.markdown("<div class='main-header'><h1>👤 Student Profile</h1><p>Enter your academic details and skill ratings</p></div>", unsafe_allow_html=True)

    profile = st.session_state["student_profile"]

    # ── Load demo profiles ──
    with st.expander("📋 Load a Demo Profile", expanded=False):
        demo_cols = st.columns(4)
        demos = {
            "🔬 Aspiring Data Scientist": {
                "python_proficiency": 4, "c_cpp_proficiency": 2, "sql_proficiency": 3,
                "data_analysis": 4, "machine_learning": 4, "web_development": 1,
                "cloud_computing": 2, "cybersecurity": 0, "communication": 3,
                "problem_solving": 4, "statistics": 4, "data_visualization": 4,
                "networking": 1, "devops": 1, "database_management": 2,
                "cgpa": 8.5, "num_projects": 4, "num_certifications": 2, "career_interest": "Data Science",
            },
            "💻 Web Developer": {
                "python_proficiency": 3, "c_cpp_proficiency": 2, "sql_proficiency": 4,
                "data_analysis": 2, "machine_learning": 1, "web_development": 5,
                "cloud_computing": 2, "cybersecurity": 1, "communication": 4,
                "problem_solving": 3, "statistics": 1, "data_visualization": 3,
                "networking": 2, "devops": 3, "database_management": 4,
                "cgpa": 7.8, "num_projects": 5, "num_certifications": 1, "career_interest": "Web Development",
            },
            "🛡️ Cybersecurity Track": {
                "python_proficiency": 3, "c_cpp_proficiency": 4, "sql_proficiency": 2,
                "data_analysis": 2, "machine_learning": 1, "web_development": 2,
                "cloud_computing": 2, "cybersecurity": 4, "communication": 3,
                "problem_solving": 4, "statistics": 2, "data_visualization": 1,
                "networking": 4, "devops": 2, "database_management": 2,
                "cgpa": 7.5, "num_projects": 2, "num_certifications": 3, "career_interest": "Cybersecurity",
            },
            "☁️ Cloud Engineer": {
                "python_proficiency": 3, "c_cpp_proficiency": 2, "sql_proficiency": 3,
                "data_analysis": 2, "machine_learning": 2, "web_development": 3,
                "cloud_computing": 5, "cybersecurity": 2, "communication": 3,
                "problem_solving": 3, "statistics": 1, "data_visualization": 2,
                "networking": 4, "devops": 4, "database_management": 3,
                "cgpa": 7.9, "num_projects": 3, "num_certifications": 2, "career_interest": "Cloud Computing",
            },
        }
        for i, (label, demo) in enumerate(demos.items()):
            with demo_cols[i % 4]:
                if st.button(label, use_container_width=True):
                    st.session_state["student_profile"] = demo.copy()
                    st.rerun()

    st.markdown("---")

    # ── Academic Info ──
    st.markdown("<div class='section-title'>🎓 Academic Information</div>", unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0,
                               value=float(profile.get("cgpa", 7.5)), step=0.1)
    with ac2:
        projects = st.number_input("Number of Projects Completed", min_value=0, max_value=20,
                                   value=int(profile.get("num_projects", 2)))
    with ac3:
        certs = st.number_input("Number of Certifications", min_value=0, max_value=20,
                                value=int(profile.get("num_certifications", 1)))

    interests = ["Data Science", "AI/ML", "Web Development", "Cybersecurity",
                 "Cloud Computing", "Software Development", "Networking", "Embedded Systems"]
    career_interest = st.selectbox("Career Interest Area",
                                   interests,
                                   index=interests.index(profile.get("career_interest", "Data Science"))
                                   if profile.get("career_interest", "Data Science") in interests else 0)

    st.markdown("---")
    st.markdown("<div class='section-title'>🛠️ Skill Proficiency Ratings</div>", unsafe_allow_html=True)
    st.caption("Rate each skill: 0 = No knowledge -> 5 = Expert level")

    skill_groups = {
        "Programming Skills": ["python_proficiency", "c_cpp_proficiency", "sql_proficiency", "database_management"],
        "Data & AI Skills":   ["data_analysis", "machine_learning", "statistics", "data_visualization"],
        "Development Skills": ["web_development", "devops"],
        "Infrastructure":     ["cloud_computing", "networking", "cybersecurity"],
        "Soft Skills":        ["communication", "problem_solving"],
    }

    new_skills = {}
    for group_name, cols in skill_groups.items():
        st.markdown(f"**{group_name}**")
        gcols = st.columns(min(len(cols), 4))
        for i, col in enumerate(cols):
            with gcols[i % 4]:
                val = st.slider(
                    SKILL_DISPLAY.get(col, col),
                    min_value=0, max_value=5,
                    value=int(profile.get(col, 0)),
                    key=f"skill_{col}",
                    help=f"Current value: {PROFICIENCY_LABELS.get(profile.get(col, 0), 'Unknown')}",
                )
                new_skills[col] = val
                level_label = PROFICIENCY_LABELS.get(val, "")
                st.caption(f"Level: **{level_label}**")

    # ── Save ──
    st.markdown("---")
    save_col, reset_col = st.columns([1, 1])
    with save_col:
        if st.button("💾 Save Profile", type="primary", use_container_width=True):
            updated = {
                **new_skills,
                "cgpa": cgpa,
                "num_projects": projects,
                "num_certifications": certs,
                "career_interest": career_interest,
            }
            st.session_state["student_profile"] = updated
            st.success("✅ Profile saved! Navigate to Skill Gap Analysis or Job Role Prediction.")

    with reset_col:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.session_state["student_profile"] = {
                "python_proficiency": 3, "c_cpp_proficiency": 2, "sql_proficiency": 2,
                "data_analysis": 2, "machine_learning": 1, "web_development": 2,
                "cloud_computing": 1, "cybersecurity": 1, "communication": 3,
                "problem_solving": 3, "statistics": 2, "data_visualization": 2,
                "networking": 1, "devops": 1, "database_management": 2,
                "cgpa": 7.5, "num_projects": 2, "num_certifications": 1,
                "career_interest": "Data Science",
            }
            st.rerun()

    # ── Profile Summary ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📋 Current Profile Summary</div>", unsafe_allow_html=True)
    current = st.session_state["student_profile"]
    skill_vals = {SKILL_DISPLAY.get(k, k): current.get(k, 0) for k in SKILL_COLS}
    skill_series = pd.Series(skill_vals)
    fig = px.bar(
        x=skill_series.index, y=skill_series.values,
        color=skill_series.values,
        color_continuous_scale="Blues",
        labels={"x": "Skill", "y": "Proficiency (0–5)"},
        title="Your Skill Profile",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 5.5]),
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page 3: Skill Gap Analysis ───────────────────────────────────────────────
def page_skill_gap(job_roles, skill_map):
    st.markdown("<div class='main-header'><h1>📊 Skill Gap Analysis</h1><p>Compare your skills against job role requirements</p></div>", unsafe_allow_html=True)

    profile = st.session_state["student_profile"]

    col_left, col_right = st.columns([1, 3])
    with col_left:
        role = st.selectbox(
            "🎯 Select Target Job Role",
            sorted(job_roles["job_role"].tolist()),
            index=sorted(job_roles["job_role"].tolist()).index(
                st.session_state.get("selected_role", "Data Scientist")
            ) if st.session_state.get("selected_role") in job_roles["job_role"].tolist() else 0,
        )
        st.session_state["selected_role"] = role

    # ── Run Analysis ──
    result = analyze_student_role(profile, role, job_roles, skill_map)
    score  = result["match_score"]
    gap_df = result["gap_df"]

    with col_right:
        # Score badge
        if score >= 75:
            badge_class = "score-high"
            emoji = "✅"
        elif score >= 50:
            badge_class = "score-mid"
            emoji = "⚠️"
        else:
            badge_class = "score-low"
            emoji = "❌"

        st.markdown(f"""
        <div style='text-align:center; padding:1rem;'>
            <p style='font-size:1rem; color:#607D8B; margin:0;'>Overall Skill Match</p>
            <div style='font-size:3.5rem; font-weight:900; color:{"#2E7D32" if score>=75 else "#F57F17" if score>=50 else "#B71C1C"};'>
                {score:.1f}%
            </div>
            <p style='font-size:0.9rem;'>{emoji} {'Good match!' if score>=75 else 'Moderate match — focused learning needed.' if score>=50 else 'Significant gaps — structured learning required.'}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Radar Chart ──
    st.markdown("---")
    from src.data_preprocessing import load_job_roles as _ljr
    from src.skill_gap import get_role_requirements
    requirements = get_role_requirements(job_roles, role)

    radar_fig = plot_radar_chart(profile, requirements, role)
    st.plotly_chart(radar_fig, use_container_width=True)

    # ── Gap Bar Chart ──
    bar_fig = plot_skill_gap_bar(gap_df, role)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ── Detailed Table ──
    st.markdown("---")
    col_s, col_m = st.columns(2)

    with col_s:
        st.markdown("<div class='section-title'>✅ Strong Skills</div>", unsafe_allow_html=True)
        strong = gap_df[gap_df["status"] == "✅ Meets Requirement"]
        if strong.empty:
            st.info("No skills fully meet requirements yet. Keep learning!")
        else:
            for _, row in strong.iterrows():
                st.success(f"**{row['display_name']}** — You: {row['student_score']} / Required: {row['required_score']}")

    with col_m:
        st.markdown("<div class='section-title'>🚀 Priority Improvements</div>", unsafe_allow_html=True)
        priority = result["priority_improvements"]
        if priority.empty:
            st.success("🎉 You meet all requirements for this role!")
        else:
            for _, row in priority.iterrows():
                gap_val = int(row["gap"])
                st.warning(f"**{row['display_name']}** — Gap: {gap_val} | You: {int(row['student_score'])} -> Need: {int(row['required_score'])}")

    # ── Project Check ──
    st.markdown("---")
    min_proj = result["min_projects_required"]
    stu_proj = result["student_projects"]
    if stu_proj >= min_proj:
        st.success(f"✅ Projects: You have **{stu_proj}** projects. Minimum required: {min_proj}")
    else:
        st.warning(f"⚠️ Projects: You have **{stu_proj}** project(s). Aim for at least **{min_proj}** for this role.")

    # ── All Roles Comparison ──
    st.markdown("---")
    st.markdown("<div class='section-title'>🔍 How You Match All Roles</div>", unsafe_allow_html=True)
    all_roles_df = compare_all_roles(profile, job_roles, skill_map)
    fig = px.bar(
        all_roles_df,
        x="match_score", y="job_role",
        orientation="h",
        color="match_score",
        color_continuous_scale="RdYlGn",
        labels={"match_score": "Match Score (%)", "job_role": "Job Role"},
        title="Skill Match Score Across All Job Roles",
        range_color=[0, 100],
    )
    fig.update_layout(height=380, coloraxis_showscale=False, xaxis=dict(range=[0, 105]))
    for i, row in all_roles_df.iterrows():
        fig.add_annotation(x=row["match_score"] + 1, y=row["job_role"],
                           text=f"{row['match_score']:.1f}%", showarrow=False,
                           font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)


# ─── Page 4: Job Role Prediction ──────────────────────────────────────────────
def page_prediction(job_roles, skill_map):
    st.markdown("<div class='main-header'><h1>🎯 Job Role Prediction</h1><p>ML-powered career prediction based on your profile</p></div>", unsafe_allow_html=True)

    model, scaler, encoder, features, model_loaded = get_model()

    if not model_loaded:
        st.error(
            "❌ **Model not found.** Please run the training script first:\n\n"
            "```\npython src/train_model.py\n```"
        )
        st.info("The model will be saved to the `models/` directory automatically.")
        return

    profile = st.session_state["student_profile"]
    enriched = enrich_profile(profile)

    if st.button("🔮 Predict My Best Job Roles", type="primary", use_container_width=True):
        with st.spinner("Analyzing your profile..."):
            predictions, top_role = predict_job_roles(
                enriched, model, scaler, encoder, features, top_n=8
            )

        st.markdown("---")
        st.markdown(f"<div class='section-title'>🏆 Top Predicted Role: {top_role}</div>", unsafe_allow_html=True)

        # ── Probability Gauge ──
        top3 = predictions.head(3)
        cols = st.columns(3)
        colors = ["#2196F3", "#42A5F5", "#90CAF9"]
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                prob = row["probability (%)"]
                rank = int(row["rank"])
                role = row["job_role"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    number={"suffix": "%", "font": {"size": 24}},
                    title={"text": f"#{rank} {role}", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": colors[i]},
                        "steps": [
                            {"range": [0, 40], "color": "#FFCDD2"},
                            {"range": [40, 70], "color": "#FFF9C4"},
                            {"range": [70, 100], "color": "#C8E6C9"},
                        ],
                    },
                ))
                fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

        # ── Full Table ──
        st.markdown("---")
        st.markdown("<div class='section-title'>📋 All Role Predictions</div>", unsafe_allow_html=True)
        styled = predictions.copy()
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class='disclaimer'>
            ℹ️ <strong>Note:</strong> These predictions are based on a <strong>synthetic dataset</strong>
            created for this academic project. They represent workflow demonstration only and do not
            guarantee employment or actual suitability for industry roles.
        </div>
        """, unsafe_allow_html=True)

        # ── Feature Importance ──
        fi_df = get_feature_importance(model, features)
        if not fi_df.empty:
            st.markdown("---")
            st.markdown("<div class='section-title'>📈 Feature Importance (What Matters Most)</div>", unsafe_allow_html=True)
            top_fi = fi_df.head(10).copy()
            top_fi["feature_display"] = top_fi["feature"].map(
                lambda x: SKILL_DISPLAY.get(x, x.replace("_", " ").title())
            )
            fig = px.bar(
                top_fi,
                x="importance", y="feature_display",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                labels={"importance": "Feature Importance", "feature_display": "Feature"},
                title="Top 10 Features Influencing Job Role Prediction",
            )
            fig.update_layout(height=380, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Click the button above to predict your best-fit job roles based on your current profile.")
        st.caption("Make sure you've saved your profile on the **Student Profile** page first.")


# ─── Page 5: Learning Roadmap ─────────────────────────────────────────────────
def page_roadmap(job_roles, skill_map):
    st.markdown("<div class='main-header'><h1>🗺️ Personalized Learning Roadmap</h1><p>Your step-by-step path to your target role</p></div>", unsafe_allow_html=True)

    profile = st.session_state["student_profile"]
    role    = st.selectbox(
        "🎯 Select your target role",
        sorted(job_roles["job_role"].tolist()),
        index=sorted(job_roles["job_role"].tolist()).index(
            st.session_state.get("selected_role", "Data Scientist")
        ) if st.session_state.get("selected_role") in job_roles["job_role"].tolist() else 0,
    )

    result  = analyze_student_role(profile, role, job_roles, skill_map)
    roadmap = generate_roadmap(profile, role, result)

    score = result["match_score"]
    st.markdown(f"""
    <div style='background:#E3F2FD; border-radius:8px; padding:1rem; margin-bottom:1rem;'>
        <strong>Your match score for {role}:</strong>
        <span style='color:{"#2E7D32" if score>=75 else "#F57F17" if score>=50 else "#B71C1C"}; font-size:1.3rem; font-weight:700;'>
            {score:.1f}%
        </span>
        &nbsp;|&nbsp; Priority skills to learn:
        <em>{', '.join(roadmap['priority_order'][:3]) or 'None — great match!'}</em>
    </div>
    """, unsafe_allow_html=True)

    # ── Learning Stages ──
    st.markdown("<div class='section-title'>📚 Learning Path</div>", unsafe_allow_html=True)

    tab_b, tab_i, tab_a = st.tabs(["🟢 Beginner", "🔵 Intermediate", "🔴 Advanced"])

    def render_resources(tab, df, label):
        with tab:
            if df.empty:
                st.info(f"No specific {label} resources needed for your current gaps. Great progress!")
            else:
                for _, row in df.iterrows():
                    skill_name = SKILL_DISPLAY.get(row.get("skill", ""), row.get("skill", ""))
                    st.markdown(f"""
                    <div style='border:1px solid #E3F2FD; border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem;'>
                        <span style='font-size:0.75rem; background:#E3F2FD; color:#1565C0; padding:0.1rem 0.5rem; border-radius:10px; font-weight:600;'>
                            {skill_name}
                        </span><br>
                        <strong>{row['title']}</strong><br>
                        <a href='{row['url']}' target='_blank'>🔗 {row['url']}</a>
                    </div>
                    """, unsafe_allow_html=True)

    render_resources(tab_b, roadmap["beginner_resources"], "beginner")
    render_resources(tab_i, roadmap["intermediate_resources"], "intermediate")
    render_resources(tab_a, roadmap["advanced_resources"], "advanced")

    # ── Projects ──
    st.markdown("---")
    st.markdown("<div class='section-title'>🛠️ Recommended Projects</div>", unsafe_allow_html=True)
    for i, idea in enumerate(roadmap["project_ideas"], 1):
        st.markdown(f"**{i}.** {idea}")

    # ── Certifications ──
    st.markdown("---")
    st.markdown("<div class='section-title'>🏅 Certification Pathway</div>", unsafe_allow_html=True)
    for cert_name, cert_url, duration in roadmap["certifications"]:
        st.markdown(f"""
        <div style='border:1px solid #E8F5E9; border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem; background:#FAFFFE;'>
            <strong>🏅 {cert_name}</strong> &nbsp;⏱ {duration}<br>
            <a href='{cert_url}' target='_blank'>🔗 {cert_url}</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer' style='margin-top:1rem;'>
        ⚠️ Estimated durations are approximate and depend on individual learning pace.
        These are career guidance suggestions, not employment guarantees.
    </div>
    """, unsafe_allow_html=True)

    # ── Skills Radar ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📡 Skills Radar</div>", unsafe_allow_html=True)
    from src.skill_gap import get_role_requirements
    requirements = get_role_requirements(job_roles, role)
    radar = plot_radar_chart(profile, requirements, role)
    st.plotly_chart(radar, use_container_width=True)


# ─── Page 6: Reports ─────────────────────────────────────────────────────────
def page_reports(students, job_roles, skill_map):
    st.markdown("<div class='main-header'><h1>📄 Reports & Downloads</h1><p>Export your analysis results</p></div>", unsafe_allow_html=True)

    profile = st.session_state["student_profile"]
    role    = st.selectbox(
        "📊 Select role for report",
        sorted(job_roles["job_role"].tolist()),
        index=sorted(job_roles["job_role"].tolist()).index(
            st.session_state.get("selected_role", "Data Scientist")
        ) if st.session_state.get("selected_role") in job_roles["job_role"].tolist() else 0,
    )

    result = analyze_student_role(profile, role, job_roles, skill_map)
    gap_df = result["gap_df"]

    # ── Skill Gap CSV ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📥 Download Skill Gap Report (CSV)</div>", unsafe_allow_html=True)

    export_df = gap_df[["display_name", "student_score", "required_score", "gap", "status", "importance"]].copy()
    export_df.columns = ["Skill", "Your Score", "Required Score", "Gap", "Status", "Importance (0-3)"]
    export_df["Match Score (%)"] = result["match_score"]
    export_df["Target Role"] = role

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Skill Gap CSV",
        data=csv_bytes,
        file_name=f"skill_gap_report_{role.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── All Roles Summary CSV ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📥 Download All Roles Comparison (CSV)</div>", unsafe_allow_html=True)
    all_roles_df = compare_all_roles(profile, job_roles, skill_map)
    all_csv = all_roles_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download All Roles CSV",
        data=all_csv,
        file_name="all_roles_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Student Dataset CSV ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📥 Download Full Dataset (CSV)</div>", unsafe_allow_html=True)
    student_csv = students.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Student Dataset CSV",
        data=student_csv,
        file_name="students_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Report Preview ──
    st.markdown("---")
    st.markdown("<div class='section-title'>📋 Analysis Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | Target Role | **{role}** |
    | Overall Match Score | **{result['match_score']:.1f}%** |
    | Strong Skills | **{len(result['strong_skills'])}** |
    | Skills Needing Improvement | **{len(result['missing_skills'])}** |
    | Projects (You / Required) | **{result['student_projects']} / {result['min_projects_required']}** |
    | CGPA | **{profile.get('cgpa', 0)}** |
    | Certifications | **{profile.get('num_certifications', 0)}** |
    """)

    st.markdown("---")
    st.markdown("<div class='section-title'>📊 Full Gap Table</div>", unsafe_allow_html=True)
    st.dataframe(export_df.drop(columns=["Match Score (%)", "Target Role"]),
                 use_container_width=True, hide_index=True)

    # ── Model Report ──
    reports_dir = os.path.join(ROOT, "reports")
    report_path = os.path.join(reports_dir, "classification_report.txt")
    if os.path.exists(report_path):
        st.markdown("---")
        st.markdown("<div class='section-title'>🤖 Model Classification Report</div>", unsafe_allow_html=True)
        with open(report_path, "r") as f:
            report_text = f.read()
        st.code(report_text)


# ─── Main Router ──────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    # Data loading with error handling
    try:
        students, job_roles, skill_map = get_data()
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found!\n\n{e}")
        st.code("python data/generate_datasets.py", language="bash")
        st.stop()

    if page == "🏠 Home":
        page_home(students, job_roles, skill_map)
    elif page == "👤 Student Profile":
        page_student_profile()
    elif page == "📊 Skill Gap Analysis":
        page_skill_gap(job_roles, skill_map)
    elif page == "🎯 Job Role Prediction":
        page_prediction(job_roles, skill_map)
    elif page == "🗺️ Learning Roadmap":
        page_roadmap(job_roles, skill_map)
    elif page == "📄 Reports":
        page_reports(students, job_roles, skill_map)


if __name__ == "__main__":
    main()


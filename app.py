import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="6G Smart Factory Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# =========================================================
# FUTURISTIC UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
    radial-gradient(circle at 10% 10%, rgba(0,229,255,.10), transparent 30%),
    radial-gradient(circle at 90% 20%, rgba(170,0,255,.10), transparent 30%),
    linear-gradient(135deg,#040713,#081329 55%,#10051d);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align:center;
    font-size:38px;
    font-weight:900;
    letter-spacing:1px;
    background:linear-gradient(90deg,#00e5ff,#7c4dff,#ff3cac);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    color:#a9b8d0;
    font-size:15px;
    margin-bottom:22px;
}

.section-title {
    font-size:23px;
    font-weight:800;
    color:white;
    margin-top:22px;
    margin-bottom:12px;
}

.kpi-card {
    background:linear-gradient(145deg,
        rgba(12,31,60,.95),
        rgba(30,10,55,.95));
    border:1px solid rgba(0,229,255,.25);
    border-radius:18px;
    padding:16px;
    text-align:center;
    box-shadow:0 0 18px rgba(0,229,255,.08);
    transition:.3s;
}

.kpi-card:hover {
    transform:translateY(-4px);
    box-shadow:0 0 28px rgba(0,229,255,.20);
}

.kpi-label {
    color:#91a6c4;
    font-size:12px;
    letter-spacing:1px;
}

.kpi-value {
    color:#ffffff;
    font-size:25px;
    font-weight:900;
    margin-top:5px;
}

.chart-card {
    background:rgba(7,18,38,.72);
    border:1px solid rgba(124,77,255,.22);
    border-radius:16px;
    padding:8px;
    margin-bottom:12px;
}

.info-box {
    background:rgba(8,22,44,.80);
    border-left:4px solid #00e5ff;
    border-radius:12px;
    padding:15px;
    color:#d7e5f8;
}

div[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#040817,#110622);
}

div[data-testid="stSidebar"] * {
    color:#dce8ff;
}

[data-testid="stMetric"] {
    background:rgba(10,25,50,.75);
    border:1px solid rgba(124,77,255,.25);
    border-radius:14px;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOADERS
# =========================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except:
            return None
    return None

def dfile(name):
    return load_csv(os.path.join(DATA, name))

def rfile(name):
    return load_csv(os.path.join(RESULTS, name))

def image_exists(name):
    return os.path.exists(os.path.join(DATA, name))

def show_chart_image(col, filename, title):
    path = os.path.join(DATA, filename)

    if os.path.exists(path):
        with col:
            st.markdown(
                f'<div class="chart-card"><b style="color:white;">{title}</b>',
                unsafe_allow_html=True
            )
            st.image(path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

def show_table(df, title):
    if df is not None:
        st.markdown(
            f'<div class="section-title">{title}</div>',
            unsafe_allow_html=True
        )
        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏭 6G SMART FACTORY INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">6G Network Performance → Manufacturing Efficiency → Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🛰️ CONTROL CENTER")

page = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Mission Control",
        "📡 Network Analytics",
        "🏭 Manufacturing Analytics",
        "🔗 Network × Manufacturing",
        "🤖 Machine Learning",
        "📊 Research Results"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### PROJECT
**6G Smart Factory**

**Technology**
- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Streamlit

**Analysis**
- Network KPIs
- Manufacturing KPIs
- Correlations
- Machine Learning
""")

# =========================================================
# FILES
# =========================================================

network = rfile("network_findings.csv")
manufacturing = rfile("manufacturing_findings.csv")
correlations = rfile("network_manufacturing_correlations.csv")
relationships = rfile("network_manufacturing_relationships.csv")
models = rfile("model_results.csv")
errors = rfile("prediction_errors.csv")
kpis = rfile("project_kpis.csv")
research = rfile("research_summary.csv")

# =========================================================
# MISSION CONTROL
# =========================================================

if page == "🚀 Mission Control":

    st.markdown(
        '<div class="section-title">⚡ Project Mission Control</div>',
        unsafe_allow_html=True
    )

    c1,c2,c3,c4 = st.columns(4)

    cards = [
        ("PROJECT", "30 DAYS"),
        ("DOMAIN", "6G + AI"),
        ("ML MODELS", "3"),
        ("PIPELINE", "END-TO-END")
    ]

    for col,(label,value) in zip([c1,c2,c3,c4],cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">🧠 Analytical Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    📥 Data → 🧹 Cleaning → 🔍 EDA → 📡 6G Analysis →
    🏭 Manufacturing → 🔗 Relationships → 🤖 ML → 📊 Results
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📈 Key Visualizations</div>',
        unsafe_allow_html=True
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "network_performance_index.png",
        "📡 Network Performance Index"
    )

    show_chart_image(
        c2,
        "efficiency_distribution.png",
        "🏭 Efficiency Distribution"
    )

# =========================================================
# NETWORK
# =========================================================

elif page == "📡 Network Analytics":

    st.markdown(
        '<div class="section-title">📡 6G Network Performance Analytics</div>',
        unsafe_allow_html=True
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "latency_distribution.png",
        "⏱️ Latency Distribution"
    )

    show_chart_image(
        c2,
        "packet_loss_distribution.png",
        "📦 Packet Loss Distribution"
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "latency_vs_efficiency.png",
        "⏱️ Latency vs Efficiency"
    )

    show_chart_image(
        c2,
        "latency_vs_error_rate.png",
        "⚠️ Latency vs Error Rate"
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "latency_vs_production_speed.png",
        "🏭 Latency vs Production Speed"
    )

    show_chart_image(
        c2,
        "packet_loss_vs_efficiency.png",
        "📦 Packet Loss vs Efficiency"
    )

    show_table(network, "📋 Network Findings")

# =========================================================
# MANUFACTURING
# =========================================================

elif page == "🏭 Manufacturing Analytics":

    st.markdown(
        '<div class="section-title">🏭 Manufacturing Efficiency Analytics</div>',
        unsafe_allow_html=True
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "production_speed_distribution.png",
        "⚙️ Production Speed Distribution"
    )

    show_chart_image(
        c2,
        "efficiency_distribution.png",
        "📊 Efficiency Distribution"
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "production_speed_by_efficiency.png",
        "⚙️ Production Speed by Efficiency"
    )

    show_chart_image(
        c2,
        "packet_loss_efficiency.png",
        "📦 Packet Loss Efficiency"
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "latency_vs_production_speed.png",
        "⏱️ Latency vs Production Speed"
    )

    show_chart_image(
        c2,
        "latency_vs_efficiency.png",
        "⏱️ Latency vs Efficiency"
    )

    show_table(manufacturing, "📋 Manufacturing Findings")

# =========================================================
# RELATIONSHIPS
# =========================================================

elif page == "🔗 Network × Manufacturing":

    st.markdown(
        '<div class="section-title">🔗 Network × Manufacturing Intelligence</div>',
        unsafe_allow_html=True
    )

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "network_vs_manufacturing_importance.png",
        "📊 Network vs Manufacturing Importance"
    )

    show_chart_image(
        c2,
        "feature_importance.png",
        "⭐ Feature Importance"
    )

    if correlations is not None:

        st.markdown(
            '<div class="section-title">📈 Correlation Data</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            correlations,
            use_container_width=True,
            height=320
        )

    if relationships is not None:

        st.markdown(
            '<div class="section-title">🔗 Relationship Results</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            relationships,
            use_container_width=True,
            height=320
        )

    st.info(
        "Correlation indicates statistical association and should not be interpreted as proof of causation."
    )

# =========================================================
# MACHINE LEARNING
# =========================================================

elif page == "🤖 Machine Learning":

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Intelligence</div>',
        unsafe_allow_html=True
    )

    if models is not None:

        show_table(
            models,
            "📊 Model Evaluation"
        )

        numeric = models.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric) > 0:

            xcol = models.columns[0]
            ycol = numeric[-1]

            try:

                fig = px.bar(
                    models,
                    x=xcol,
                    y=ycol,
                    color=xcol,
                    template="plotly_dark",
                    title="Machine Learning Model Comparison"
                )

                fig.update_layout(
                    height=430,
                    margin=dict(l=30,r=30,t=60,b=40),
                    font=dict(size=12)
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except:
                pass

    c1,c2 = st.columns(2)

    show_chart_image(
        c1,
        "feature_importance.png",
        "⭐ Feature Importance"
    )

    show_chart_image(
        c2,
        "network_vs_manufacturing_importance.png",
        "🔗 Network vs Manufacturing Importance"
    )

    show_table(
        errors,
        "⚠️ Prediction Error Analysis"
    )

# =========================================================
# RESEARCH
# =========================================================

elif page == "📊 Research Results":

    st.markdown(
        '<div class="section-title">📊 Research Results & Evidence</div>',
        unsafe_allow_html=True
    )

    if research is not None:
        show_table(
            research,
            "📚 Research Summary"
        )

    if kpis is not None:
        show_table(
            kpis,
            "📈 Project KPIs"
        )

    st.markdown(
        '<div class="section-title">🗂️ Generated Evidence</div>',
        unsafe_allow_html=True
    )

    evidence = [
        "correlation_matrix.csv",
        "cross_validation_results.csv",
        "efficiency_percentage.csv",
        "efficiency_summary.csv",
        "feature_importance.csv",
        "manufacturing_findings.csv",
        "model_results.csv",
        "network_findings.csv",
        "network_manufacturing_correlations.csv",
        "network_manufacturing_relationships.csv",
        "predictions.csv",
        "prediction_errors.csv",
        "project_kpis.csv",
        "research_summary.csv"
    ]

    for i in range(0,len(evidence),3):

        cols = st.columns(3)

        for col,file in zip(cols,evidence[i:i+3]):

            if os.path.exists(os.path.join(RESULTS,file)):
                with col:
                    st.success("✓ " + file)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8295b3;
        font-size:13px;
        padding:8px;">
        <b>6G Smart Factory Network Analysis & Machine Learning</b><br>
        Unified Mentor Project • Sneha S • Electronics & Communication Engineering
    </div>
    """,
    unsafe_allow_html=True
)

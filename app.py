import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="6G Smart Factory Network Analysis",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 6G Smart Factory Network Analysis")
st.subheader("Impact of 6G Network Performance on Manufacturing Efficiency")

st.markdown("""
This interactive dashboard analyzes the relationship between 6G network
performance indicators and manufacturing efficiency in smart factories.
""")

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

def load_csv(filename):
    path = os.path.join(RESULTS, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Network Performance",
        "Manufacturing Efficiency",
        "Network–Manufacturing Relationships",
        "Machine Learning",
        "Research Results"
    ]
)

# Load results
project_kpis = load_csv("project_kpis.csv")
network_findings = load_csv("network_findings.csv")
manufacturing_findings = load_csv("manufacturing_findings.csv")
correlations = load_csv("network_manufacturing_correlations.csv")
relationships = load_csv("network_manufacturing_relationships.csv")
model_results = load_csv("model_results.csv")
prediction_errors = load_csv("prediction_errors.csv")
research_summary = load_csv("research_summary.csv")

if page == "Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Project Duration", "30 Days")
    c2.metric("Analysis Domain", "6G + Smart Factory")
    c3.metric("ML Models", "3")
    c4.metric("Analysis Type", "Network + Manufacturing")

    st.divider()

    st.markdown("### Project Objectives")
    st.write("""
    • Analyze 6G network performance indicators  
    • Study manufacturing efficiency indicators  
    • Identify statistical relationships between network and manufacturing metrics  
    • Build machine-learning models for efficiency-status prediction  
    • Evaluate prediction performance and errors  
    """)

    st.markdown("### Project Pipeline")

    st.code(
        "Data Collection → Preprocessing → EDA → KPI Analysis → "
        "Relationship Analysis → Machine Learning → Evaluation → Insights"
    )

elif page == "Network Performance":
    st.header("📡 Network Performance Analysis")

    if network_findings is not None:
        st.dataframe(network_findings, use_container_width=True)
    else:
        st.warning("Network findings file not available.")

    st.markdown("### Network Indicators")
    st.write("""
    The analysis focuses on communication-performance indicators such as
    latency, packet loss, network performance and related network metrics.
    """)

elif page == "Manufacturing Efficiency":
    st.header("🏭 Manufacturing Efficiency Analysis")

    if manufacturing_findings is not None:
        st.dataframe(manufacturing_findings, use_container_width=True)
    else:
        st.warning("Manufacturing findings file not available.")

    st.markdown("### Manufacturing Indicators")
    st.write("""
    Manufacturing-side analysis examines production speed, efficiency,
    error-related indicators and other operational measurements.
    """)

elif page == "Network–Manufacturing Relationships":
    st.header("🔗 Network–Manufacturing Relationship Analysis")

    if correlations is not None:
        st.subheader("Correlation Results")
        st.dataframe(correlations, use_container_width=True)

    if relationships is not None:
        st.subheader("Relationship Results")
        st.dataframe(relationships, use_container_width=True)

    st.info(
        "Correlation indicates statistical association and should not be "
        "interpreted as proof of causation."
    )

elif page == "Machine Learning":
    st.header("🤖 Machine Learning Analysis")

    if model_results is not None:
        st.subheader("Model Results")
        st.dataframe(model_results, use_container_width=True)

    st.markdown("### Models")
    st.write("""
    • Logistic Regression  
    • Decision Tree  
    • Random Forest  
    """)

    if prediction_errors is not None:
        st.subheader("Prediction Error Analysis")
        st.dataframe(prediction_errors, use_container_width=True)

elif page == "Research Results":
    st.header("📊 Research Summary")

    if research_summary is not None:
        st.dataframe(research_summary, use_container_width=True)

    if project_kpis is not None:
        st.subheader("Project KPIs")
        st.dataframe(project_kpis, use_container_width=True)

st.divider()

st.caption(
    "6G Smart Factory Network Analysis & Machine Learning | "
    "Unified Mentor Project | Sneha S"
)

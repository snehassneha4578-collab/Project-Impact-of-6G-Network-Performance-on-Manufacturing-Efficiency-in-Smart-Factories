import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DAY 20 - 6G SMART FACTORY STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Analysis",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title(
    "📡 6G Smart Factory Network Analysis"
)

st.subheader(
    "Impact of 6G Network Performance "
    "on Manufacturing Efficiency"
)

st.write(
    "This dashboard provides an interactive "
    "analysis of network performance and "
    "smart-factory manufacturing efficiency."
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

st.success(
    "Dataset loaded successfully!"
)

st.write(
    f"Dataset contains {df.shape[0]} rows "
    f"and {df.shape[1]} columns."
)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.write(
    "Use the filters below to explore the dataset."
)

efficiency_options = (
    df["Efficiency_Status"]
    .unique()
    .tolist()
)

selected_efficiency = st.sidebar.multiselect(
    "Select Efficiency Status",
    options=efficiency_options,
    default=efficiency_options
)

# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df["Efficiency_Status"].isin(
        selected_efficiency
    )
]

# Prevent empty-filter errors
if filtered_df.empty:
    st.warning(
        "Please select at least one Efficiency Status."
    )
    st.stop()

st.write(
    f"Filtered Records: {len(filtered_df)}"
)

# ============================================================
# KPI CALCULATIONS
# ============================================================

average_latency = (
    filtered_df[
        "Network_Latency_ms"
    ].mean()
)

average_packet_loss = (
    filtered_df[
        "Packet_Loss_%"
    ].mean()
)

average_production_speed = (
    filtered_df[
        "Production_Speed_units_per_hr"
    ].mean()
)

average_error_rate = (
    filtered_df[
        "Error_Rate_%"
    ].mean()
)

average_defect_rate = (
    filtered_df[
        "Quality_Control_Defect_Rate_%"
    ].mean()
)

average_maintenance_score = (
    filtered_df[
        "Predictive_Maintenance_Score"
    ].mean()
)

total_records = len(
    filtered_df
)

high_efficiency_records = (
    filtered_df[
        filtered_df[
            "Efficiency_Status"
        ] == "High"
    ].shape[0]
)

# ============================================================
# KPI CARDS - ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Latency",
    f"{average_latency:.2f} ms"
)

col2.metric(
    "Average Packet Loss",
    f"{average_packet_loss:.2f}%"
)

col3.metric(
    "Production Speed",
    f"{average_production_speed:.2f} units/hr"
)

col4.metric(
    "Error Rate",
    f"{average_error_rate:.2f}%"
)

# ============================================================
# KPI CARDS - ROW 2
# ============================================================

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Defect Rate",
    f"{average_defect_rate:.2f}%"
)

col6.metric(
    "Maintenance Score",
    f"{average_maintenance_score:.2f}"
)

col7.metric(
    "Total Records",
    total_records
)

col8.metric(
    "High Efficiency Records",
    high_efficiency_records
)

# ============================================================
# EFFICIENCY DISTRIBUTION
# ============================================================

st.header(
    "Efficiency Distribution"
)

efficiency_counts = (
    filtered_df[
        "Efficiency_Status"
    ].value_counts()
)

fig, ax = plt.subplots()

efficiency_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Efficiency Status"
)

ax.set_ylabel(
    "Number of Records"
)

ax.set_title(
    "Smart Factory Efficiency Distribution"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# NETWORK PERFORMANCE ANALYSIS
# ============================================================

st.header(
    "Network Performance Analysis"
)

# ------------------------------------------------------------
# Average Latency
# ------------------------------------------------------------

latency_data = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Network_Latency_ms"
    ].mean()
)

fig, ax = plt.subplots()

latency_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Efficiency Status"
)

ax.set_ylabel(
    "Average Latency (ms)"
)

ax.set_title(
    "Average Network Latency by Efficiency"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ------------------------------------------------------------
# Packet Loss
# ------------------------------------------------------------

packet_loss_data = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Packet_Loss_%"
    ].mean()
)

fig, ax = plt.subplots()

packet_loss_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Efficiency Status"
)

ax.set_ylabel(
    "Average Packet Loss (%)"
)

ax.set_title(
    "Average Packet Loss by Efficiency"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# MANUFACTURING PERFORMANCE
# ============================================================

st.header(
    "Manufacturing Performance Analysis"
)

# ------------------------------------------------------------
# Production Speed
# ------------------------------------------------------------

production_data = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Production_Speed_units_per_hr"
    ].mean()
)

fig, ax = plt.subplots()

production_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Efficiency Status"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Production Speed by Efficiency"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ------------------------------------------------------------
# Defect Rate
# ------------------------------------------------------------

defect_data = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Quality_Control_Defect_Rate_%"
    ].mean()
)

fig, ax = plt.subplots()

defect_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Efficiency Status"
)

ax.set_ylabel(
    "Defect Rate (%)"
)

ax.set_title(
    "Defect Rate by Efficiency"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# NETWORK VS MANUFACTURING RELATIONSHIP
# ============================================================

st.header(
    "Network vs Manufacturing Relationship"
)

# ------------------------------------------------------------
# Latency vs Production Speed
# ------------------------------------------------------------

fig, ax = plt.subplots()

ax.scatter(
    filtered_df[
        "Network_Latency_ms"
    ],
    filtered_df[
        "Production_Speed_units_per_hr"
    ]
)

ax.set_xlabel(
    "Network Latency (ms)"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Network Latency vs Production Speed"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ------------------------------------------------------------
# Packet Loss vs Production Speed
# ------------------------------------------------------------

fig, ax = plt.subplots()

ax.scatter(
    filtered_df[
        "Packet_Loss_%"
    ],
    filtered_df[
        "Production_Speed_units_per_hr"
    ]
)

ax.set_xlabel(
    "Packet Loss (%)"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Packet Loss vs Production Speed"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# FILTERED DATASET
# ============================================================

st.header(
    "Filtered Dataset"
)

st.dataframe(
    filtered_df,
    use_container_width=True
)

csv_data = filtered_df.to_csv(
    index=False
)

st.download_button(
    label="Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_smart_factory_data.csv",
    mime="text/csv"
)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

st.header(
    "Correlation Analysis"
)

correlation_columns = [
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]

correlation_matrix = (
    filtered_df[
        correlation_columns
    ].corr()
)

st.dataframe(
    correlation_matrix,
    use_container_width=True
)

# ============================================================
# NETWORK & MANUFACTURING SUMMARY
# ============================================================

network_summary = pd.DataFrame({

    "Metric": [
        "Average Latency (ms)",
        "Average Packet Loss (%)",
        "Average Production Speed (units/hr)",
        "Average Error Rate (%)"
    ],

    "Value": [
        average_latency,
        average_packet_loss,
        average_production_speed,
        average_error_rate
    ]

})

st.header(
    "Network & Manufacturing Summary"
)

st.table(
    network_summary
)

# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown(
    """
### Project

**Impact of 6G Network Performance
on Manufacturing Efficiency**

### Technologies

• Python

• Pandas

• Matplotlib

• Scikit-learn

• Streamlit

### Domain

6G + Smart Factory + Machine Learning
"""
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.write(
    "6G Smart Factory Network Analysis "
    "Project | Machine Learning Internship"
)

print(
    "DAY 20 STREAMLIT DASHBOARD COMPLETED SUCCESSFULLY!"
)



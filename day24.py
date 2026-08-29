import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

print("LIBRARIES IMPORTED SUCCESSFULLY!")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Manufacturing Analysis",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title(
    "🏭 6G Smart Factory Network Analysis"
)

st.subheader(
    "Smart Factory Manufacturing Performance Analysis"
)

st.write(
    "This dashboard provides an interactive analysis "
    "of production performance, machine condition, "
    "manufacturing efficiency and network impact."
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

st.success(
    "Cleaned Smart Factory dataset loaded successfully!"
)

st.write(
    f"Dataset contains {df.shape[0]} rows "
    f"and {df.shape[1]} columns."
)

# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)

st.sidebar.write(
    "Select efficiency categories to analyze."
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

filtered_df = df[
    df["Efficiency_Status"].isin(
        selected_efficiency
    )
]

st.sidebar.write(
    f"Filtered Records: {len(filtered_df)}"
)

# ============================================================
# SAFETY CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No records selected. Please select at least "
        "one Efficiency Status from the sidebar."
    )

    st.stop()

# ============================================================
# SMART FACTORY MANUFACTURING ANALYSIS
# ============================================================

st.header(
    "🏭 Smart Factory Manufacturing Analysis"
)

st.write(
    "Interactive analysis of production "
    "performance, machine condition and "
    "manufacturing efficiency."
)

# ============================================================
# MANUFACTURING KPIs
# ============================================================

average_production_speed = (
    filtered_df[
        "Production_Speed_units_per_hr"
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

average_power = (
    filtered_df[
        "Power_Consumption_kW"
    ].mean()
)

average_temperature = (
    filtered_df[
        "Temperature_C"
    ].mean()
)

average_vibration = (
    filtered_df[
        "Vibration_Hz"
    ].mean()
)

# ============================================================
# KPI DISPLAY
# ============================================================

col1, col2 = st.columns(2)

col1.metric(
    "Average Production Speed",
    f"{average_production_speed:.2f} units/hr"
)

col2.metric(
    "Average Defect Rate",
    f"{average_defect_rate:.2f}%"
)

col1, col2 = st.columns(2)

col1.metric(
    "Average Maintenance Score",
    f"{average_maintenance_score:.2f}"
)

col2.metric(
    "Average Power Consumption",
    f"{average_power:.2f} kW"
)

col1, col2 = st.columns(2)

col1.metric(
    "Average Temperature",
    f"{average_temperature:.2f} °C"
)

col2.metric(
    "Average Vibration",
    f"{average_vibration:.2f} Hz"
)

# ============================================================
# EFFICIENCY DISTRIBUTION
# ============================================================

efficiency_counts = (
    filtered_df[
        "Efficiency_Status"
    ].value_counts()
)

st.subheader(
    "Manufacturing Efficiency Distribution"
)

st.bar_chart(
    efficiency_counts
)

# ============================================================
# PRODUCTION SPEED BY EFFICIENCY
# ============================================================

production_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Production_Speed_units_per_hr"
    ].mean()
)

st.subheader(
    "Average Production Speed by Efficiency"
)

st.bar_chart(
    production_by_efficiency
)

# ============================================================
# DEFECT RATE BY EFFICIENCY
# ============================================================

defect_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Quality_Control_Defect_Rate_%"
    ].mean()
)

st.subheader(
    "Average Defect Rate by Efficiency"
)

st.bar_chart(
    defect_by_efficiency
)

# ============================================================
# MAINTENANCE SCORE
# ============================================================

maintenance_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Predictive_Maintenance_Score"
    ].mean()
)

st.subheader(
    "Average Predictive Maintenance Score"
)

st.bar_chart(
    maintenance_by_efficiency
)

# ============================================================
# POWER CONSUMPTION
# ============================================================

power_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Power_Consumption_kW"
    ].mean()
)

st.subheader(
    "Average Power Consumption by Efficiency"
)

st.bar_chart(
    power_by_efficiency
)

# ============================================================
# TEMPERATURE
# ============================================================

temperature_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Temperature_C"
    ].mean()
)

st.subheader(
    "Average Temperature by Efficiency"
)

st.bar_chart(
    temperature_by_efficiency
)

# ============================================================
# VIBRATION
# ============================================================

vibration_by_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Vibration_Hz"
    ].mean()
)

st.subheader(
    "Average Vibration by Efficiency"
)

st.bar_chart(
    vibration_by_efficiency
)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

production_power_corr = (
    filtered_df[
        "Production_Speed_units_per_hr"
    ].corr(
        filtered_df[
            "Power_Consumption_kW"
        ]
    )
)

production_defect_corr = (
    filtered_df[
        "Production_Speed_units_per_hr"
    ].corr(
        filtered_df[
            "Quality_Control_Defect_Rate_%"
        ]
    )
)

temperature_vibration_corr = (
    filtered_df[
        "Temperature_C"
    ].corr(
        filtered_df[
            "Vibration_Hz"
        ]
    )
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Production vs Power",
    f"{production_power_corr:.3f}"
)

col2.metric(
    "Production vs Defect Rate",
    f"{production_defect_corr:.3f}"
)

col3.metric(
    "Temperature vs Vibration",
    f"{temperature_vibration_corr:.3f}"
)

# ============================================================
# PRODUCTION VS POWER
# ============================================================

st.subheader(
    "Production Speed vs Power Consumption"
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    filtered_df[
        "Power_Consumption_kW"
    ],
    filtered_df[
        "Production_Speed_units_per_hr"
    ]
)

ax.set_xlabel(
    "Power Consumption (kW)"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Production Speed vs Power Consumption"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# PRODUCTION VS DEFECT RATE
# ============================================================

st.subheader(
    "Production Speed vs Defect Rate"
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    filtered_df[
        "Quality_Control_Defect_Rate_%"
    ],
    filtered_df[
        "Production_Speed_units_per_hr"
    ]
)

ax.set_xlabel(
    "Defect Rate (%)"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Production Speed vs Defect Rate"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# TEMPERATURE VS VIBRATION
# ============================================================

st.subheader(
    "Temperature vs Vibration"
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    filtered_df[
        "Temperature_C"
    ],
    filtered_df[
        "Vibration_Hz"
    ]
)

ax.set_xlabel(
    "Temperature (°C)"
)

ax.set_ylabel(
    "Vibration (Hz)"
)

ax.set_title(
    "Temperature vs Vibration"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# PRODUCTION PERFORMANCE SCORE
# ============================================================

production_min = (
    df[
        "Production_Speed_units_per_hr"
    ].min()
)

production_max = (
    df[
        "Production_Speed_units_per_hr"
    ].max()
)

if production_max == production_min:

    production_score = 100

else:

    production_score = (
        100
        *
        (
            average_production_speed
            - production_min
        )
        /
        (
            production_max
            - production_min
        )
    )

# ============================================================
# QUALITY SCORE
# ============================================================

defect_min = (
    df[
        "Quality_Control_Defect_Rate_%"
    ].min()
)

defect_max = (
    df[
        "Quality_Control_Defect_Rate_%"
    ].max()
)

if defect_max == defect_min:

    quality_score = 100

else:

    quality_score = (
        100
        *
        (
            1
            -
            (
                average_defect_rate
                - defect_min
            )
            /
            (
                defect_max
                - defect_min
            )
        )
    )

# ============================================================
# PRODUCTION PERFORMANCE
# ============================================================

production_performance_score = (
    0.6 * production_score
    +
    0.4 * quality_score
)

# Keep score inside 0-100
production_performance_score = max(
    0,
    min(
        100,
        production_performance_score
    )
)

if production_performance_score >= 75:

    production_health = "High"

elif production_performance_score >= 50:

    production_health = "Moderate"

else:

    production_health = "Low"

st.subheader(
    "🏭 Production Performance Score"
)

st.metric(
    "Production Performance",
    f"{production_performance_score:.2f}/100"
)

st.info(
    f"Production Health: {production_health}"
)

# ============================================================
# EFFICIENCY-WISE MANUFACTURING SUMMARY
# ============================================================

efficiency_manufacturing_summary = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        [
            "Production_Speed_units_per_hr",
            "Quality_Control_Defect_Rate_%",
            "Predictive_Maintenance_Score",
            "Power_Consumption_kW",
            "Temperature_C",
            "Vibration_Hz",
            "Error_Rate_%"
        ]
    ]
    .mean()
    .round(3)
)

st.subheader(
    "Efficiency-wise Manufacturing Performance"
)

st.dataframe(
    efficiency_manufacturing_summary,
    width="stretch"
)

# ============================================================
# NETWORK VS MANUFACTURING
# ============================================================

st.header(
    "🔗 Network Performance vs Manufacturing"
)

st.write(
    "This section investigates the "
    "relationship between communication "
    "performance and manufacturing KPIs."
)

# ============================================================
# LATENCY VS DEFECT RATE
# ============================================================

st.subheader(
    "Network Latency vs Defect Rate"
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    filtered_df[
        "Network_Latency_ms"
    ],
    filtered_df[
        "Quality_Control_Defect_Rate_%"
    ]
)

ax.set_xlabel(
    "Network Latency (ms)"
)

ax.set_ylabel(
    "Defect Rate (%)"
)

ax.set_title(
    "Network Latency vs Defect Rate"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# PACKET LOSS VS DEFECT RATE
# ============================================================

st.subheader(
    "Packet Loss vs Defect Rate"
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.scatter(
    filtered_df[
        "Packet_Loss_%"
    ],
    filtered_df[
        "Quality_Control_Defect_Rate_%"
    ]
)

ax.set_xlabel(
    "Packet Loss (%)"
)

ax.set_ylabel(
    "Defect Rate (%)"
)

ax.set_title(
    "Packet Loss vs Defect Rate"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# NETWORK LATENCY BY EFFICIENCY
# ============================================================

latency_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Network_Latency_ms"
    ].mean()
)

st.subheader(
    "Network Latency by Efficiency"
)

st.bar_chart(
    latency_efficiency
)

# ============================================================
# PACKET LOSS BY EFFICIENCY
# ============================================================

packet_efficiency = (
    filtered_df.groupby(
        "Efficiency_Status"
    )[
        "Packet_Loss_%"
    ].mean()
)

st.subheader(
    "Packet Loss by Efficiency"
)

st.bar_chart(
    packet_efficiency
)

# ============================================================
# FINAL MANUFACTURING SUMMARY
# ============================================================

manufacturing_summary = pd.DataFrame({

    "Metric": [
        "Average Production Speed",
        "Average Defect Rate",
        "Average Maintenance Score",
        "Average Power Consumption",
        "Average Temperature",
        "Average Vibration",
        "Production Performance Score",
        "Production Health"
    ],

    "Value": [

        f"{average_production_speed:.2f} units/hr",

        f"{average_defect_rate:.2f}%",

        f"{average_maintenance_score:.2f}",

        f"{average_power:.2f} kW",

        f"{average_temperature:.2f} °C",

        f"{average_vibration:.2f} Hz",

        f"{production_performance_score:.2f}/100",

        production_health
    ]
})

st.header(
    "📊 Manufacturing Monitoring Summary"
)

st.table(
    manufacturing_summary
)

# ============================================================
# FILTERED DATASET
# ============================================================

st.header(
    "📋 Filtered Smart Factory Dataset"
)

st.dataframe(
    filtered_df,
    width="stretch"
)

# ============================================================
# DOWNLOAD DATA
# ============================================================

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
# SIDEBAR INFORMATION
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

st.markdown("---")

st.write(
    "6G Smart Factory Network Analysis "
    "| Machine Learning Internship"
)

print(
    "DAY 24 SMART FACTORY MANUFACTURING "
    "ANALYSIS COMPLETED SUCCESSFULLY!"
)

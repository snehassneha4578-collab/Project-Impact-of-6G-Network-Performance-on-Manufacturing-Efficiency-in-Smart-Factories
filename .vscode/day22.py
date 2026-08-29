# DAY 22 — SMART FACTORY REAL-TIME MONITORING & ALERT SYSTEM
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Monitoring",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title(
    "🏭 6G Smart Factory Real-Time Monitoring"
)

st.subheader(
    "Network Performance, Manufacturing KPIs "
    "and Intelligent Alert System"
)

st.write(
    "This dashboard monitors smart-factory network "
    "and manufacturing conditions and provides "
    "machine-learning-based efficiency predictions "
    "and operational alerts."
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load(
    "data/optimized_random_forest_model.pkl"
)

label_encoder = joblib.load(
    "data/optimized_random_forest_label_encoder.pkl"
)

st.success(
    "Dataset and optimized Random Forest model loaded successfully!"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏭 Factory Monitoring Controls"
)

st.sidebar.write(
    "Adjust the monitoring thresholds below."
)

latency_threshold = st.sidebar.number_input(
    "Maximum Network Latency (ms)",
    min_value=0.0,
    value=50.0
)

packet_loss_threshold = st.sidebar.number_input(
    "Maximum Packet Loss (%)",
    min_value=0.0,
    value=5.0
)

temperature_threshold = st.sidebar.number_input(
    "Maximum Temperature (°C)",
    min_value=0.0,
    value=85.0
)

vibration_threshold = st.sidebar.number_input(
    "Maximum Vibration (Hz)",
    min_value=0.0,
    value=60.0
)

error_rate_threshold = st.sidebar.number_input(
    "Maximum Error Rate (%)",
    min_value=0.0,
    value=5.0
)

defect_rate_threshold = st.sidebar.number_input(
    "Maximum Defect Rate (%)",
    min_value=0.0,
    value=5.0
)

# ============================================================
# LATEST FACTORY RECORD
# ============================================================

latest_record = df.iloc[-1]

# ============================================================
# CURRENT FACTORY VALUES
# ============================================================

latency = latest_record[
    "Network_Latency_ms"
]

packet_loss = latest_record[
    "Packet_Loss_%"
]

temperature = latest_record[
    "Temperature_C"
]

vibration = latest_record[
    "Vibration_Hz"
]

power = latest_record[
    "Power_Consumption_kW"
]

production_speed = latest_record[
    "Production_Speed_units_per_hr"
]

defect_rate = latest_record[
    "Quality_Control_Defect_Rate_%"
]

maintenance_score = latest_record[
    "Predictive_Maintenance_Score"
]

error_rate = latest_record[
    "Error_Rate_%"
]

# ============================================================
# CURRENT FACTORY STATUS
# ============================================================

st.header(
    "📊 Current Factory Status"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Network Latency",
    f"{latency:.2f} ms"
)

col2.metric(
    "Packet Loss",
    f"{packet_loss:.2f}%"
)

col3.metric(
    "Temperature",
    f"{temperature:.2f} °C"
)

col4.metric(
    "Vibration",
    f"{vibration:.2f} Hz"
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Production Speed",
    f"{production_speed:.2f} units/hr"
)

col6.metric(
    "Defect Rate",
    f"{defect_rate:.2f}%"
)

col7.metric(
    "Maintenance Score",
    f"{maintenance_score:.2f}"
)

col8.metric(
    "Error Rate",
    f"{error_rate:.2f}%"
)

# ============================================================
# NETWORK HEALTH
# ============================================================

st.header(
    "📡 Network Health Monitoring"
)

network_alerts = []

if latency > latency_threshold:

    network_alerts.append(
        "⚠️ High Network Latency Detected"
    )

if packet_loss > packet_loss_threshold:

    network_alerts.append(
        "⚠️ High Packet Loss Detected"
    )

if len(network_alerts) == 0:

    st.success(
        "✅ Network condition is within acceptable limits."
    )

else:

    for alert in network_alerts:

        st.warning(alert)

# ============================================================
# MACHINE HEALTH
# ============================================================

st.header(
    "⚙️ Machine Health Monitoring"
)

machine_alerts = []

if temperature > temperature_threshold:

    machine_alerts.append(
        "🔥 High Temperature Detected"
    )

if vibration > vibration_threshold:

    machine_alerts.append(
        "⚠️ High Machine Vibration Detected"
    )

if error_rate > error_rate_threshold:

    machine_alerts.append(
        "⚠️ High Machine Error Rate Detected"
    )

if defect_rate > defect_rate_threshold:

    machine_alerts.append(
        "⚠️ High Product Defect Rate Detected"
    )

if maintenance_score < 50:

    machine_alerts.append(
        "🔧 Low Predictive Maintenance Score"
    )

if len(machine_alerts) == 0:

    st.success(
        "✅ Machine and manufacturing conditions are normal."
    )

else:

    for alert in machine_alerts:

        st.warning(alert)

# ============================================================
# OVERALL ALERT STATUS
# ============================================================

st.header(
    "🚨 Overall Factory Alert Status"
)

total_alerts = (
    len(network_alerts)
    +
    len(machine_alerts)
)

if total_alerts == 0:

    st.success(
        "🟢 FACTORY STATUS: NORMAL"
    )

elif total_alerts <= 2:

    st.warning(
        f"🟡 FACTORY STATUS: WARNING "
        f"({total_alerts} alert(s))"
    )

else:

    st.error(
        f"🔴 FACTORY STATUS: CRITICAL "
        f"({total_alerts} alert(s))"
    )

# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

st.header(
    "🤖 AI-Based Efficiency Prediction"
)

prediction_input = pd.DataFrame({

    "Network_Latency_ms": [
        latency
    ],

    "Packet_Loss_%": [
        packet_loss
    ],

    "Temperature_C": [
        temperature
    ],

    "Vibration_Hz": [
        vibration
    ],

    "Power_Consumption_kW": [
        power
    ],

    "Quality_Control_Defect_Rate_%": [
        defect_rate
    ],

    "Production_Speed_units_per_hr": [
        production_speed
    ],

    "Predictive_Maintenance_Score": [
        maintenance_score
    ],

    "Error_Rate_%": [
        error_rate
    ]

})

prediction_encoded = model.predict(
    prediction_input
)

prediction = label_encoder.inverse_transform(
    prediction_encoded
)[0]

st.subheader(
    "Predicted Efficiency Status"
)

if prediction == "High":

    st.success(
        f"🟢 Predicted Efficiency: {prediction}"
    )

elif prediction == "Medium":

    st.warning(
        f"🟡 Predicted Efficiency: {prediction}"
    )

else:

    st.error(
        f"🔴 Predicted Efficiency: {prediction}"
    )

# ============================================================
# PREDICTION PROBABILITY
# ============================================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        prediction_input
    )[0]

    probability_df = pd.DataFrame({

        "Efficiency Status":
            label_encoder.classes_,

        "Probability (%)":
            probabilities * 100

    })

    st.subheader(
        "Prediction Confidence"
    )

    st.dataframe(
        probability_df,
        width="stretch"
    )

    fig, ax = plt.subplots()

    ax.bar(
        probability_df[
            "Efficiency Status"
        ],

        probability_df[
            "Probability (%)"
        ]
    )

    ax.set_xlabel(
        "Efficiency Status"
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    ax.set_title(
        "AI Efficiency Prediction Confidence"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

# ============================================================
# FACTORY TREND ANALYSIS
# ============================================================

st.header(
    "📈 Factory Performance Trends"
)

trend_records = st.slider(
    "Number of Recent Records",
    min_value=100,
    max_value=min(5000, len(df)),
    value=min(1000, len(df)),
    step=100
)

recent_df = df.tail(
    trend_records
)

# ============================================================
# LATENCY TREND
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    recent_df["Network_Latency_ms"]
)

ax.set_xlabel(
    "Record"
)

ax.set_ylabel(
    "Latency (ms)"
)

ax.set_title(
    "Network Latency Trend"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# PACKET LOSS TREND
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    recent_df["Packet_Loss_%"]
)

ax.set_xlabel(
    "Record"
)

ax.set_ylabel(
    "Packet Loss (%)"
)

ax.set_title(
    "Packet Loss Trend"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# PRODUCTION TREND
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    recent_df[
        "Production_Speed_units_per_hr"
    ]
)

ax.set_xlabel(
    "Record"
)

ax.set_ylabel(
    "Production Speed (units/hr)"
)

ax.set_title(
    "Production Speed Trend"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# TEMPERATURE TREND
# ============================================================

fig, ax = plt.subplots()

ax.plot(
    recent_df["Temperature_C"]
)

ax.set_xlabel(
    "Record"
)

ax.set_ylabel(
    "Temperature (°C)"
)

ax.set_title(
    "Machine Temperature Trend"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# EFFICIENCY DISTRIBUTION
# ============================================================

st.header(
    "📊 Efficiency Distribution"
)

efficiency_counts = (
    df["Efficiency_Status"]
    .value_counts()
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
    "Factory Efficiency Distribution"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# FACTORY DATA
# ============================================================

st.header(
    "📋 Recent Factory Data"
)

st.dataframe(
    recent_df.tail(100),
    width="stretch"
)

# ============================================================
# DOWNLOAD DATA
# ============================================================

csv_data = recent_df.to_csv(
    index=False
)

st.download_button(
    label="Download Recent Factory Data",
    data=csv_data,
    file_name="recent_factory_monitoring_data.csv",
    mime="text/csv"
)

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.sidebar.markdown(
    """
### 🤖 Machine Learning

**Model:**

Optimized Random Forest

**Target:**

Efficiency_Status

**Purpose:**

Manufacturing Efficiency Prediction
"""
)

st.sidebar.markdown(
    """
### 📡 Network Parameters

• Network Latency

• Packet Loss

### ⚙️ Manufacturing Parameters

• Temperature

• Vibration

• Power Consumption

• Production Speed

• Defect Rate

• Maintenance Score

• Error Rate
"""
)

st.sidebar.success(
    "AI Monitoring System Active"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "6G Smart Factory Network Analysis "
    "| Day 22 | Machine Learning Internship"
)

print(
    "DAY 22 SMART FACTORY REAL-TIME "
    "MONITORING AND ALERT SYSTEM COMPLETED SUCCESSFULLY!"
)


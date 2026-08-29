import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Analysis",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📡 6G Smart Factory Network Analysis")

st.subheader(
    "Impact of 6G Network Performance "
    "on Manufacturing Efficiency"
)

st.write(
    "This dashboard provides an interactive analysis "
    "of network performance and smart-factory "
    "manufacturing efficiency."
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

# ============================================================
# LOAD OPTIMIZED RANDOM FOREST MODEL
# ============================================================

model = joblib.load(
    "data/optimized_random_forest_model.pkl"
)

label_encoder = joblib.load(
    "data/optimized_random_forest_label_encoder.pkl"
)

st.success(
    "Dataset and Machine Learning model loaded successfully!"
)

st.write(
    f"Dataset contains {df.shape[0]} rows "
    f"and {df.shape[1]} columns."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

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

filtered_df = df[
    df["Efficiency_Status"].isin(selected_efficiency)
]

st.write(
    f"Filtered Records: {len(filtered_df)}"
)

# ============================================================
# KPI CALCULATIONS
# ============================================================

average_latency = (
    filtered_df["Network_Latency_ms"].mean()
)

average_packet_loss = (
    filtered_df["Packet_Loss_%"].mean()
)

average_production_speed = (
    filtered_df["Production_Speed_units_per_hr"].mean()
)

average_error_rate = (
    filtered_df["Error_Rate_%"].mean()
)

average_defect_rate = (
    filtered_df["Quality_Control_Defect_Rate_%"].mean()
)

average_maintenance_score = (
    filtered_df["Predictive_Maintenance_Score"].mean()
)

total_records = len(filtered_df)

high_efficiency_records = (
    filtered_df[
        filtered_df["Efficiency_Status"] == "High"
    ].shape[0]
)

# ============================================================
# KPI DISPLAY
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

st.header("Efficiency Distribution")

efficiency_counts = (
    filtered_df["Efficiency_Status"]
    .value_counts()
)

fig, ax = plt.subplots()

efficiency_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Efficiency Status")
ax.set_ylabel("Number of Records")
ax.set_title("Smart Factory Efficiency Distribution")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# NETWORK PERFORMANCE ANALYSIS
# ============================================================

st.header("Network Performance Analysis")

# Average Latency

latency_data = filtered_df.groupby(
    "Efficiency_Status"
)["Network_Latency_ms"].mean()

fig, ax = plt.subplots()

latency_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Efficiency Status")
ax.set_ylabel("Average Latency (ms)")
ax.set_title("Average Network Latency by Efficiency")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# Average Packet Loss

packet_loss_data = filtered_df.groupby(
    "Efficiency_Status"
)["Packet_Loss_%"].mean()

fig, ax = plt.subplots()

packet_loss_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Efficiency Status")
ax.set_ylabel("Average Packet Loss (%)")
ax.set_title("Average Packet Loss by Efficiency")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# Production Speed

production_data = filtered_df.groupby(
    "Efficiency_Status"
)["Production_Speed_units_per_hr"].mean()

fig, ax = plt.subplots()

production_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Efficiency Status")
ax.set_ylabel("Production Speed (units/hr)")
ax.set_title("Production Speed by Efficiency")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# Defect Rate

defect_data = filtered_df.groupby(
    "Efficiency_Status"
)["Quality_Control_Defect_Rate_%"].mean()

fig, ax = plt.subplots()

defect_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Efficiency Status")
ax.set_ylabel("Defect Rate (%)")
ax.set_title("Defect Rate by Efficiency")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# NETWORK VS MANUFACTURING RELATIONSHIP
# ============================================================

st.header("Network vs Manufacturing Relationship")

# Latency vs Production Speed

fig, ax = plt.subplots()

ax.scatter(
    filtered_df["Network_Latency_ms"],
    filtered_df["Production_Speed_units_per_hr"]
)

ax.set_xlabel("Network Latency (ms)")
ax.set_ylabel("Production Speed (units/hr)")
ax.set_title("Network Latency vs Production Speed")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# Packet Loss vs Production Speed

fig, ax = plt.subplots()

ax.scatter(
    filtered_df["Packet_Loss_%"],
    filtered_df["Production_Speed_units_per_hr"]
)

ax.set_xlabel("Packet Loss (%)")
ax.set_ylabel("Production Speed (units/hr)")
ax.set_title("Packet Loss vs Production Speed")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)

# ============================================================
# FILTERED DATASET
# ============================================================

st.header("Filtered Dataset")

st.dataframe(
    filtered_df,
    width="stretch"
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

st.header("Correlation Analysis")

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
    width="stretch"
)

# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

st.header(
    "🤖 Manufacturing Efficiency Prediction"
)

st.write(
    "Enter machine and network conditions "
    "to predict the expected efficiency status."
)

# ============================================================
# USER INPUTS
# ============================================================

latency_input = st.number_input(
    "Network Latency (ms)",
    min_value=0.0,
    value=20.0
)

packet_loss_input = st.number_input(
    "Packet Loss (%)",
    min_value=0.0,
    value=1.5
)

temperature_input = st.number_input(
    "Temperature (°C)",
    value=70.0
)

vibration_input = st.number_input(
    "Vibration (Hz)",
    min_value=0.0,
    value=35.0
)

power_input = st.number_input(
    "Power Consumption (kW)",
    min_value=0.0,
    value=50.0
)

defect_rate_input = st.number_input(
    "Quality Control Defect Rate (%)",
    min_value=0.0,
    value=2.0
)

production_speed_input = st.number_input(
    "Production Speed (units/hr)",
    min_value=0.0,
    value=100.0
)

maintenance_score_input = st.number_input(
    "Predictive Maintenance Score",
    min_value=0.0,
    value=85.0
)

error_rate_input = st.number_input(
    "Error Rate (%)",
    min_value=0.0,
    value=1.0
)

# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "Network_Latency_ms": [
        latency_input
    ],

    "Packet_Loss_%": [
        packet_loss_input
    ],

    "Temperature_C": [
        temperature_input
    ],

    "Vibration_Hz": [
        vibration_input
    ],

    "Power_Consumption_kW": [
        power_input
    ],

    "Quality_Control_Defect_Rate_%": [
        defect_rate_input
    ],

    "Production_Speed_units_per_hr": [
        production_speed_input
    ],

    "Predictive_Maintenance_Score": [
        maintenance_score_input
    ],

    "Error_Rate_%": [
        error_rate_input
    ]
})

st.subheader("Prediction Input")

st.dataframe(
    input_data,
    width="stretch"
)

# ============================================================
# PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state[
        "prediction_history"
    ] = []

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔮 Predict Efficiency"):

    prediction_encoded = model.predict(
        input_data
    )

    # If model was trained using LabelEncoder,
    # convert encoded prediction back to class name.

    prediction_result = (
        label_encoder.inverse_transform(
            prediction_encoded
        )[0]
    )

    st.success(
        f"Predicted Efficiency Status: "
        f"{prediction_result}"
    )

    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    if hasattr(model, "predict_proba"):

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )

        probability_df = pd.DataFrame({

            "Efficiency Status":
                label_encoder.classes_,

            "Probability":
                probabilities

        })

        probability_df[
            "Probability (%)"
        ] = (
            probability_df[
                "Probability"
            ] * 100
        )

        st.subheader(
            "Prediction Probabilities"
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
            "Model Prediction Probability"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # ========================================================
    # SAVE PREDICTION HISTORY
    # ========================================================

    prediction_record = input_data.copy()

    prediction_record[
        "Predicted_Efficiency"
    ] = prediction_result

    st.session_state[
        "prediction_history"
    ].append(
        prediction_record
    )

    st.info(
        "The prediction is generated using the "
        "trained Machine Learning model based "
        "on the entered network and manufacturing "
        "conditions."
    )

# ============================================================
# PREDICTION HISTORY DISPLAY
# ============================================================

if st.session_state[
    "prediction_history"
]:

    st.subheader(
        "Prediction History"
    )

    history_df = pd.concat(
        st.session_state[
            "prediction_history"
        ],
        ignore_index=True
    )

    st.dataframe(
        history_df,
        width="stretch"
    )

    history_csv = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv"
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
### Machine Learning Model

**Purpose:**

Manufacturing Efficiency Prediction

**Target:**

Efficiency_Status

**Inputs:**

Network + Machine + Manufacturing KPIs
"""
)

st.sidebar.success(
    "ML Model: Loaded"
)

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

• Joblib

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
    "Project | Machine Learning Internship"
)

print(
    "DAY 21 PREDICTION-ENABLED STREAMLIT "
    "DASHBOARD COMPLETED SUCCESSFULLY!"
)

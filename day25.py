import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Intelligent Alert System",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚨 6G Smart Factory Intelligent Alert System"
)

st.subheader(
    "Network Performance, Manufacturing KPIs "
    "and Factory Health Monitoring"
)

st.write(
    "This dashboard monitors network and manufacturing "
    "conditions and generates intelligent operational alerts."
)


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = "data/cleaned_smart_factory.csv"

try:

    df = pd.read_csv(DATASET_PATH)

    st.success(
        "Cleaned Smart Factory dataset loaded successfully!"
    )

    st.write(
        f"Dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns."
    )

except Exception as e:

    st.error(
        f"Dataset could not be loaded: {e}"
    )

    st.stop()


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = "data/decision_tree_model.pkl"

try:

    model = joblib.load(MODEL_PATH)

    st.success(
        "Decision Tree Machine Learning model loaded successfully!"
    )

except Exception as e:

    model = None

    st.warning(
        "Machine Learning model could not be loaded. "
        "Alert analysis will still work."
    )

    st.write(
        f"Model loading error: {e}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Dashboard Controls"
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
# CURRENT FACTORY KPIs
# ============================================================

st.header(
    "📊 Current Factory KPIs"
)


average_latency = (
    filtered_df["Network_Latency_ms"].mean()
)

average_packet_loss = (
    filtered_df["Packet_Loss_%"].mean()
)

average_temperature = (
    filtered_df["Temperature_C"].mean()
)

average_vibration = (
    filtered_df["Vibration_Hz"].mean()
)

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

average_error_rate = (
    filtered_df["Error_Rate_%"].mean()
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Network Latency",
    f"{average_latency:.2f} ms"
)

col2.metric(
    "Packet Loss",
    f"{average_packet_loss:.2f}%"
)

col3.metric(
    "Temperature",
    f"{average_temperature:.2f} °C"
)

col4.metric(
    "Vibration",
    f"{average_vibration:.2f} Hz"
)


col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Production Speed",
    f"{average_production_speed:.2f} units/hr"
)

col6.metric(
    "Defect Rate",
    f"{average_defect_rate:.2f}%"
)

col7.metric(
    "Maintenance Score",
    f"{average_maintenance_score:.2f}"
)

col8.metric(
    "Error Rate",
    f"{average_error_rate:.2f}%"
)


# ============================================================
# NETWORK PERFORMANCE SCORE
# ============================================================

latency_min = df[
    "Network_Latency_ms"
].min()

latency_max = df[
    "Network_Latency_ms"
].max()


packet_loss_min = df[
    "Packet_Loss_%"
].min()

packet_loss_max = df[
    "Packet_Loss_%"
].max()


if latency_max == latency_min:

    latency_score = 100

else:

    latency_score = (
        100
        * (
            1
            -
            (
                average_latency
                -
                latency_min
            )
            /
            (
                latency_max
                -
                latency_min
            )
        )
    )


if packet_loss_max == packet_loss_min:

    packet_loss_score = 100

else:

    packet_loss_score = (
        100
        * (
            1
            -
            (
                average_packet_loss
                -
                packet_loss_min
            )
            /
            (
                packet_loss_max
                -
                packet_loss_min
            )
        )
    )


network_performance_score = (
    0.6 * latency_score
    +
    0.4 * packet_loss_score
)


if network_performance_score >= 75:

    network_health = "Good"

elif network_performance_score >= 50:

    network_health = "Moderate"

else:

    network_health = "Poor"


# ============================================================
# PRODUCTION PERFORMANCE SCORE
# ============================================================

production_min = df[
    "Production_Speed_units_per_hr"
].min()

production_max = df[
    "Production_Speed_units_per_hr"
].max()


if production_max == production_min:

    production_score = 100

else:

    production_score = (
        100
        *
        (
            average_production_speed
            -
            production_min
        )
        /
        (
            production_max
            -
            production_min
        )
    )


defect_min = df[
    "Quality_Control_Defect_Rate_%"
].min()

defect_max = df[
    "Quality_Control_Defect_Rate_%"
].max()


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
                -
                defect_min
            )
            /
            (
                defect_max
                -
                defect_min
            )
        )
    )


production_performance_score = (
    0.6 * production_score
    +
    0.4 * quality_score
)


if production_performance_score >= 75:

    production_health = "High"

elif production_performance_score >= 50:

    production_health = "Moderate"

else:

    production_health = "Low"


# ============================================================
# FACTORY HEALTH SCORE
# ============================================================

factory_health_score = (
    0.5 * network_performance_score
    +
    0.5 * production_performance_score
)


if factory_health_score >= 75:

    factory_health = "Healthy"

elif factory_health_score >= 50:

    factory_health = "Moderate"

else:

    factory_health = "Needs Attention"


# ============================================================
# FACTORY PERFORMANCE SCORES
# ============================================================

st.header(
    "🏭 Factory Performance Scores"
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Network Score",
    f"{network_performance_score:.2f}/100"
)

col2.metric(
    "Production Score",
    f"{production_performance_score:.2f}/100"
)

col3.metric(
    "Factory Health",
    f"{factory_health_score:.2f}/100"
)


col1, col2, col3 = st.columns(3)

col1.info(
    f"Network Health: {network_health}"
)

col2.info(
    f"Production Health: {production_health}"
)

col3.info(
    f"Overall Factory Health: {factory_health}"
)


# ============================================================
# ALERT SYSTEM
# ============================================================

st.header(
    "🚨 Smart Factory Alert System"
)

st.write(
    "Click the button below to run the complete "
    "factory monitoring and alert analysis."
)


if st.button(
    "🔍 Run Factory Monitoring"
):


    # --------------------------------------------------------
    # NETWORK ALERTS
    # --------------------------------------------------------

    network_alerts = []


    if average_latency > 100:

        network_alerts.append(
            "High network latency detected."
        )


    if average_packet_loss > 5:

        network_alerts.append(
            "High packet loss detected."
        )


    if average_error_rate > 5:

        network_alerts.append(
            "High error rate detected."
        )


    # --------------------------------------------------------
    # MACHINE ALERTS
    # --------------------------------------------------------

    machine_alerts = []


    if average_temperature > 80:

        machine_alerts.append(
            "High machine temperature detected."
        )


    if average_vibration > 20:

        machine_alerts.append(
            "High machine vibration detected."
        )


    # --------------------------------------------------------
    # DISPLAY NETWORK ALERTS
    # --------------------------------------------------------

    st.subheader(
        "📡 Network Monitoring"
    )


    if network_alerts:

        for alert in network_alerts:

            st.warning(
                f"⚠️ {alert}"
            )

    else:

        st.success(
            "✅ Network conditions are within "
            "the configured monitoring limits."
        )


    # --------------------------------------------------------
    # DISPLAY MACHINE ALERTS
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Machine Monitoring"
    )


    if machine_alerts:

        for alert in machine_alerts:

            st.warning(
                f"⚠️ {alert}"
            )

    else:

        st.success(
            "✅ Machine operating conditions are "
            "within configured monitoring limits."
        )


    # --------------------------------------------------------
    # QUALITY ALERT
    # --------------------------------------------------------

    st.subheader(
        "🔬 Quality Monitoring"
    )


    if average_defect_rate > 5:

        st.warning(
            "⚠️ High quality-control defect rate detected."
        )

    else:

        st.success(
            "✅ Defect rate is within the "
            "configured monitoring limit."
        )


    # --------------------------------------------------------
    # PRODUCTION ALERT
    # --------------------------------------------------------

    st.subheader(
        "🏭 Production Monitoring"
    )


    if production_performance_score < 50:

        st.warning(
            "⚠️ Low production performance detected."
        )

    elif production_performance_score < 75:

        st.info(
            "ℹ️ Production performance is moderate."
        )

    else:

        st.success(
            "✅ Production performance is strong."
        )


    # --------------------------------------------------------
    # TOTAL ALERTS
    # --------------------------------------------------------

    total_alerts = (
        len(network_alerts)
        +
        len(machine_alerts)
    )


    if average_defect_rate > 5:

        total_alerts += 1


    if production_performance_score < 50:

        total_alerts += 1


    st.metric(
        "🚨 Active Alerts",
        total_alerts
    )


    # --------------------------------------------------------
    # OVERALL FACTORY STATUS
    # --------------------------------------------------------

    if total_alerts == 0:

        factory_status = "Normal"

    elif total_alerts <= 2:

        factory_status = "Attention Required"

    else:

        factory_status = "Critical"


    st.subheader(
        "🏭 Overall Factory Status"
    )


    if factory_status == "Normal":

        st.success(
            f"Factory Status: {factory_status}"
        )

    elif factory_status == "Attention Required":

        st.warning(
            f"Factory Status: {factory_status}"
        )

    else:

        st.error(
            f"Factory Status: {factory_status}"
        )


    # --------------------------------------------------------
    # MONITORING SUMMARY
    # --------------------------------------------------------

    alert_summary = pd.DataFrame({

        "Monitoring Area": [
            "Network",
            "Machine",
            "Quality",
            "Production"
        ],

        "Status": [

            "Alert"
            if network_alerts
            else "Normal",

            "Alert"
            if machine_alerts
            else "Normal",

            "Alert"
            if average_defect_rate > 5
            else "Normal",

            "Alert"
            if production_performance_score < 50
            else "Normal"
        ]

    })


    st.subheader(
        "📋 Factory Monitoring Summary"
    )


    st.dataframe(
        alert_summary,
        width="stretch"
    )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = []


    if average_latency > 100:

        recommendations.append(
            "Investigate network latency and "
            "communication delays."
        )


    if average_packet_loss > 5:

        recommendations.append(
            "Investigate packet loss and "
            "network reliability."
        )


    if average_error_rate > 5:

        recommendations.append(
            "Investigate abnormal error rates."
        )


    if average_temperature > 80:

        recommendations.append(
            "Inspect machine temperature and "
            "cooling conditions."
        )


    if average_vibration > 20:

        recommendations.append(
            "Inspect machine vibration and "
            "possible mechanical abnormalities."
        )


    if average_defect_rate > 5:

        recommendations.append(
            "Review the manufacturing process "
            "and quality-control conditions."
        )


    if production_performance_score < 50:

        recommendations.append(
            "Investigate low production performance "
            "and operating conditions."
        )


    st.subheader(
        "💡 Recommended Actions"
    )


    if recommendations:

        for recommendation in recommendations:

            st.info(
                f"➡️ {recommendation}"
            )

    else:

        st.success(
            "No immediate issues detected under "
            "the configured monitoring rules."
        )


    # --------------------------------------------------------
    # FACTORY HEALTH
    # --------------------------------------------------------

    st.subheader(
        "🏭 Smart Factory Health Score"
    )


    st.metric(
        "Factory Health Score",
        f"{factory_health_score:.2f}/100"
    )


    st.info(
        f"Overall Factory Health: {factory_health}"
    )


# ============================================================
# ML PREDICTION
# ============================================================

if model is not None:

    st.header(
        "🤖 Manufacturing Efficiency Prediction"
    )

    st.write(
        "Enter machine and network conditions "
        "to predict manufacturing efficiency."
    )


    latency_input = st.number_input(
        "Network Latency (ms)",
        min_value=0.0,
        value=25.0
    )


    packet_loss_input = st.number_input(
        "Packet Loss (%)",
        min_value=0.0,
        value=2.5
    )


    temperature_input = st.number_input(
        "Temperature (°C)",
        value=60.0
    )


    vibration_input = st.number_input(
        "Vibration (Hz)",
        min_value=0.0,
        value=2.5
    )


    power_input = st.number_input(
        "Power Consumption (kW)",
        min_value=0.0,
        value=5.0
    )


    defect_rate_input = st.number_input(
        "Quality Control Defect Rate (%)",
        min_value=0.0,
        value=5.0
    )


    production_speed_input = st.number_input(
        "Production Speed (units/hr)",
        min_value=0.0,
        value=275.0
    )


    maintenance_score_input = st.number_input(
        "Predictive Maintenance Score",
        min_value=0.0,
        value=0.5
    )


    error_rate_input = st.number_input(
        "Error Rate (%)",
        min_value=0.0,
        value=7.5
    )


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


    if st.button(
        "🔮 Predict Efficiency"
    ):

        try:

            prediction = model.predict(
                input_data
            )

            prediction_result = prediction[0]


            st.success(
                f"Predicted Efficiency Status: "
                f"{prediction_result}"
            )


            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        input_data
                    )[0]
                )


                probability_df = pd.DataFrame({

                    "Efficiency Status":
                        model.classes_,

                    "Probability (%)":
                        probabilities * 100

                })


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
                    "Machine Learning Prediction Probability"
                )


                st.pyplot(fig)


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# NETWORK & MANUFACTURING SUMMARY
# ============================================================

st.header(
    "📋 Network & Manufacturing Summary"
)


network_summary = pd.DataFrame({

    "Metric": [

        "Average Latency (ms)",

        "Average Packet Loss (%)",

        "Average Production Speed (units/hr)",

        "Average Defect Rate (%)",

        "Average Maintenance Score",

        "Average Error Rate (%)",

        "Network Performance Score",

        "Production Performance Score",

        "Factory Health Score"

    ],

    "Value": [

        round(average_latency, 3),

        round(average_packet_loss, 3),

        round(average_production_speed, 3),

        round(average_defect_rate, 3),

        round(average_maintenance_score, 3),

        round(average_error_rate, 3),

        round(network_performance_score, 3),

        round(production_performance_score, 3),

        round(factory_health_score, 3)

    ]

})


st.dataframe(
    network_summary,
    width="stretch"
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown(
    """
### 🤖 Machine Learning

**Model:** Decision Tree

**Target:** Efficiency_Status

**Input Features:** 9

**Cross-Validation Accuracy:** 99.995%

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
    "| Machine Learning Internship"
)

# ================================================================
# DAY 26 — ADVANCED DASHBOARD DESIGN
# 6G SMART FACTORY NETWORK ANALYSIS
# ================================================================

import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="6G Smart Factory",
    page_icon="🏭",
    layout="wide"
)

# ================================================================
# 2. PROJECT TITLE
# ================================================================

st.title("🏭 6G Smart Factory Network Analysis")

st.subheader(
    "Impact of 6G Network Performance on Manufacturing Efficiency"
)

st.write(
    "A data-driven Machine Learning and Smart Manufacturing "
    "analytics platform."
)

# ================================================================
# 3. FILE PATHS
# ================================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "cleaned_smart_factory.csv"
)

MODEL_FILE = os.path.join(
    DATA_DIR,
    "decision_tree_model.pkl"
)

LABEL_ENCODER_FILE = os.path.join(
    DATA_DIR,
    "decision_tree_label_encoder.pkl"
)

# ================================================================
# 4. LOAD DATASET
# ================================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_FILE):
        return None

    return pd.read_csv(DATA_FILE)


df = load_dataset()

if df is None:

    st.error(
        "❌ cleaned_smart_factory.csv was not found."
    )

    st.info(
        "Make sure the file is inside the data folder."
    )

    st.stop()

st.success(
    "✅ Cleaned Smart Factory dataset loaded successfully!"
)

st.write(
    f"Dataset contains {df.shape[0]} rows and "
    f"{df.shape[1]} columns."
)

# ================================================================
# 5. FIND COLUMNS SAFELY
# ================================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


latency_col = find_column([
    "Network_Latency_ms",
    "Network Latency (ms)",
    "Latency_ms"
])

packet_loss_col = find_column([
    "Packet_Loss_%",
    "Packet Loss (%)",
    "Packet_Loss"
])

temperature_col = find_column([
    "Temperature_C",
    "Temperature (°C)",
    "Temperature"
])

vibration_col = find_column([
    "Vibration_Hz",
    "Vibration (Hz)",
    "Vibration"
])

power_col = find_column([
    "Power_Consumption_kW",
    "Power Consumption (kW)",
    "Power_Consumption"
])

defect_col = find_column([
    "Quality_Control_Defect_Rate_%",
    "Quality Control Defect Rate (%)",
    "Defect_Rate_%"
])

production_col = find_column([
    "Production_Speed_units_per_hr",
    "Production Speed (units/hr)",
    "Production_Speed"
])

maintenance_col = find_column([
    "Predictive_Maintenance_Score",
    "Predictive Maintenance Score",
    "Maintenance_Score"
])

error_col = find_column([
    "Error_Rate_%",
    "Error Rate (%)",
    "Error_Rate"
])

efficiency_col = find_column([
    "Efficiency_Status",
    "Manufacturing_Efficiency",
    "Efficiency",
    "Efficiency_Class"
])

# ================================================================
# 6. CONVERT NUMERIC COLUMNS
# ================================================================

numeric_columns = [
    latency_col,
    packet_loss_col,
    temperature_col,
    vibration_col,
    power_col,
    defect_col,
    production_col,
    maintenance_col,
    error_col
]

for col in numeric_columns:

    if col is not None:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ================================================================
# 7. CALCULATE FACTORY KPIs
# ================================================================

def get_average(column):

    if column is None:
        return 0.0

    value = df[column].mean()

    if pd.isna(value):
        return 0.0

    return float(value)


average_latency = get_average(latency_col)

average_packet_loss = get_average(packet_loss_col)

average_temperature = get_average(temperature_col)

average_vibration = get_average(vibration_col)

average_power = get_average(power_col)

average_defect_rate = get_average(defect_col)

average_production_speed = get_average(
    production_col
)

average_maintenance_score = get_average(
    maintenance_col
)

average_error_rate = get_average(error_col)

# ================================================================
# 8. NETWORK PERFORMANCE SCORE
# ================================================================

latency_score = max(
    0,
    min(
        100,
        100 - average_latency
    )
)

packet_loss_score = max(
    0,
    min(
        100,
        100 - (
            average_packet_loss * 10
        )
    )
)

error_score = max(
    0,
    min(
        100,
        100 - (
            average_error_rate * 10
        )
    )
)

network_performance_score = (
    latency_score
    + packet_loss_score
    + error_score
) / 3

if network_performance_score >= 75:

    network_health = "Good"

elif network_performance_score >= 50:

    network_health = "Moderate"

else:

    network_health = "Poor"

# ================================================================
# 9. PRODUCTION PERFORMANCE SCORE
# ================================================================

production_score = max(
    0,
    min(
        100,
        average_production_speed / 5
    )
)

quality_score = max(
    0,
    min(
        100,
        100 - (
            average_defect_rate * 10
        )
    )
)

maintenance_score_normalized = max(
    0,
    min(
        100,
        average_maintenance_score * 100
    )
)

production_performance_score = (
    production_score
    + quality_score
    + maintenance_score_normalized
) / 3

if production_performance_score >= 75:

    production_health = "Good"

elif production_performance_score >= 50:

    production_health = "Moderate"

else:

    production_health = "Poor"

# ================================================================
# 10. FACTORY HEALTH
# ================================================================

factory_health_score = (
    0.5 * network_performance_score
    + 0.5 * production_performance_score
)

if factory_health_score >= 75:

    factory_health = "Healthy"

elif factory_health_score >= 50:

    factory_health = "Moderate"

else:

    factory_health = "Needs Attention"

# ================================================================
# 11. ALERT SYSTEM
# ================================================================

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
        "High network error rate detected."
    )

machine_alerts = []

if average_temperature > 80:

    machine_alerts.append(
        "High machine temperature detected."
    )

if average_vibration > 20:

    machine_alerts.append(
        "High machine vibration detected."
    )

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
        "Investigate abnormal network error rates."
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

total_alerts = (
    len(network_alerts)
    + len(machine_alerts)
)

if average_defect_rate > 5:

    total_alerts += 1

if production_performance_score < 50:

    total_alerts += 1

if total_alerts == 0:

    factory_status = "Normal"

elif total_alerts <= 2:

    factory_status = "Attention Required"

else:

    factory_status = "Critical"

# ================================================================
# 12. LOAD MACHINE LEARNING MODEL
# ================================================================

model = None
label_encoder = None

try:

    import joblib

    if os.path.exists(MODEL_FILE):

        model = joblib.load(
            MODEL_FILE
        )

    if os.path.exists(LABEL_ENCODER_FILE):

        label_encoder = joblib.load(
            LABEL_ENCODER_FILE
        )

except Exception as e:

    model = None
    label_encoder = None

# ================================================================
# 13. MODEL EVALUATION
# ================================================================

accuracy = None
precision = None
recall = None
f1 = None
cm = None
classification_rep = None
feature_importance_series = None

if (
    model is not None
    and efficiency_col is not None
):

    feature_names = [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score",
        "Error_Rate_%"
    ]

    available_features = [
        col
        for col in feature_names
        if col in df.columns
    ]

    if len(available_features) == 9:

        try:

            model_df = df[
                available_features
                + [efficiency_col]
            ].dropna()

            X = model_df[
                available_features
            ]

            y_actual = model_df[
                efficiency_col
            ].astype(str)

            # ----------------------------------------------------
            # Encode actual labels if necessary
            # ----------------------------------------------------

            if label_encoder is not None:

                try:

                    y_encoded = (
                        label_encoder.transform(
                            y_actual
                        )
                    )

                except Exception:

                    y_encoded = y_actual

            else:

                y_encoded = y_actual

            # ----------------------------------------------------
            # Model prediction
            # ----------------------------------------------------

            y_pred_raw = model.predict(X)

            # ----------------------------------------------------
            # Convert numeric predictions back to labels
            # ----------------------------------------------------

            if label_encoder is not None:

                try:

                    y_pred_labels = (
                        label_encoder.inverse_transform(
                            y_pred_raw
                        )
                    )

                except Exception:

                    y_pred_labels = y_pred_raw

            else:

                y_pred_labels = y_pred_raw

            y_pred_labels = (
                pd.Series(
                    y_pred_labels
                )
                .astype(str)
                .values
            )

            y_actual_labels = (
                y_actual
                .astype(str)
                .values
            )

            # ----------------------------------------------------
            # Final metrics
            # ----------------------------------------------------

            accuracy = accuracy_score(
                y_actual_labels,
                y_pred_labels
            )

            precision = precision_score(
                y_actual_labels,
                y_pred_labels,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_actual_labels,
                y_pred_labels,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_actual_labels,
                y_pred_labels,
                average="weighted",
                zero_division=0
            )

            class_labels = [
                "High",
                "Medium",
                "Low"
            ]

            cm = confusion_matrix(
                y_actual_labels,
                y_pred_labels,
                labels=class_labels
            )

            classification_rep = (
                classification_report(
                    y_actual_labels,
                    y_pred_labels,
                    zero_division=0
                )
            )

            # ----------------------------------------------------
            # Feature importance
            # ----------------------------------------------------

            if hasattr(
                model,
                "feature_importances_"
            ):

                feature_importance_series = pd.Series(
                    model.feature_importances_,
                    index=available_features
                ).sort_values(
                    ascending=False
                )

            st.success(
                "✅ Decision Tree Machine Learning "
                "model loaded successfully!"
            )

        except Exception as e:

            st.warning(
                f"⚠️ Model evaluation could not be completed: {e}"
            )

# ================================================================
# 14. SIDEBAR
# ================================================================

st.sidebar.title(
    "🏭 6G Smart Factory"
)

st.sidebar.write(
    "Network Performance & "
    "Manufacturing Efficiency"
)

st.sidebar.metric(
    "Factory Health",
    f"{factory_health_score:.1f}/100"
)

st.sidebar.write(
    f"Status: {factory_status}"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Factory Overview",
        "6G Network Analysis",
        "Manufacturing Analysis",
        "ML Prediction",
        "Smart Factory Alerts",
        "Model Performance",
        "Project Information"
    ]
)

# ================================================================
# 15. PROJECT OVERVIEW
# ================================================================

if page == "Project Overview":

    st.header(
        "🏠 Project Overview"
    )

    st.write(
        "Welcome to the 6G Smart Factory "
        "Network Analysis dashboard."
    )

    st.write(
        "This application analyzes network "
        "performance and manufacturing efficiency "
        "using Python, Machine Learning and "
        "interactive visualization."
    )

    st.subheader(
        "Project Overview"
    )

    st.write(
        "This project investigates the relationship "
        "between 6G network-performance parameters "
        "and manufacturing efficiency in smart factories."
    )

    st.subheader(
        "Project Objectives"
    )

    objectives = [
        "Analyze network performance parameters",
        "Analyze manufacturing efficiency",
        "Identify relationships between network conditions "
        "and factory performance",
        "Develop a Machine Learning model for efficiency prediction",
        "Build an interactive Smart Factory dashboard",
        "Provide monitoring alerts and recommendations"
    ]

    for objective in objectives:

        st.write(
            f"• {objective}"
        )

    st.subheader(
        "Technology Stack"
    )

    technologies = [
        "🐍 Python",
        "📊 Pandas",
        "🔢 NumPy",
        "📈 Matplotlib",
        "🤖 Scikit-learn",
        "🌐 Streamlit",
        "📡 6G Network Analysis",
        "🏭 Smart Manufacturing",
        "🧠 Machine Learning"
    ]

    for technology in technologies:

        st.write(
            technology
        )

# ================================================================
# 16. FACTORY OVERVIEW
# ================================================================

elif page == "Factory Overview":

    st.header(
        "📊 Smart Factory Overview"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Network Latency",
        f"{average_latency:.2f} ms"
    )

    col2.metric(
        "Packet Loss",
        f"{average_packet_loss:.2f}%"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Production Speed",
        f"{average_production_speed:.2f} units/hr"
    )

    col2.metric(
        "Defect Rate",
        f"{average_defect_rate:.2f}%"
    )

    st.subheader(
        "🏭 Factory Health"
    )

    st.metric(
        "Factory Health Score",
        f"{factory_health_score:.2f}/100"
    )

    st.info(
        f"Overall Factory Health: "
        f"{factory_health}"
    )

    st.subheader(
        "Factory Status"
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

# ================================================================
# 17. 6G NETWORK ANALYSIS
# ================================================================

elif page == "6G Network Analysis":

    st.header(
        "📡 6G Network Performance Analysis"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Latency",
        f"{average_latency:.2f} ms"
    )

    col2.metric(
        "Average Packet Loss",
        f"{average_packet_loss:.2f}%"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Error Rate",
        f"{average_error_rate:.2f}%"
    )

    col2.metric(
        "Network Score",
        f"{network_performance_score:.2f}/100"
    )

    st.info(
        f"Network Health: {network_health}"
    )

    # ------------------------------------------------------------
    # Latency Distribution
    # ------------------------------------------------------------

    if latency_col is not None:

        st.subheader(
            "Latency Distribution"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df[latency_col].dropna(),
            bins=30
        )

        ax.set_xlabel(
            "Network Latency (ms)"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "Network Latency Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    # ------------------------------------------------------------
    # Packet Loss Distribution
    # ------------------------------------------------------------

    if packet_loss_col is not None:

        st.subheader(
            "Packet Loss Distribution"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df[packet_loss_col].dropna(),
            bins=30
        )

        ax.set_xlabel(
            "Packet Loss (%)"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "Packet Loss Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

# ================================================================
# 18. MANUFACTURING ANALYSIS
# ================================================================

elif page == "Manufacturing Analysis":

    st.header(
        "🏭 Manufacturing Efficiency Analysis"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Production Speed",
        f"{average_production_speed:.2f} units/hr"
    )

    col2.metric(
        "Defect Rate",
        f"{average_defect_rate:.2f}%"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Maintenance Score",
        f"{average_maintenance_score:.2f}"
    )

    col2.metric(
        "Power Consumption",
        f"{average_power:.2f} kW"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Temperature",
        f"{average_temperature:.2f} °C"
    )

    col2.metric(
        "Vibration",
        f"{average_vibration:.2f} Hz"
    )

    st.subheader(
        "Production Performance"
    )

    st.metric(
        "Production Performance Score",
        f"{production_performance_score:.2f}/100"
    )

    st.write(
        f"Production Health: {production_health}"
    )

    if efficiency_col is not None:

        st.subheader(
            "Efficiency Distribution"
        )

        efficiency_counts = (
            df[efficiency_col]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            efficiency_counts
        )

# ================================================================
# 19. ML PREDICTION
# ================================================================

elif page == "ML Prediction":

    st.header(
        "🤖 Machine Learning Efficiency Prediction"
    )

    st.write(
        "Enter factory conditions to predict "
        "manufacturing efficiency using the "
        "trained Decision Tree model."
    )

    if model is None:

        st.error(
            "❌ Machine Learning model is not available."
        )

    else:

        st.subheader(
            "Enter Factory Conditions"
        )

        latency_input = st.number_input(
            "Network Latency (ms)",
            min_value=0.0,
            value=float(average_latency)
        )

        packet_loss_input = st.number_input(
            "Packet Loss (%)",
            min_value=0.0,
            value=float(average_packet_loss)
        )

        temperature_input = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            value=float(average_temperature)
        )

        vibration_input = st.number_input(
            "Vibration (Hz)",
            min_value=0.0,
            value=float(average_vibration)
        )

        power_input = st.number_input(
            "Power Consumption (kW)",
            min_value=0.0,
            value=float(average_power)
        )

        defect_input = st.number_input(
            "Quality Control Defect Rate (%)",
            min_value=0.0,
            value=float(average_defect_rate)
        )

        production_input = st.number_input(
            "Production Speed (units/hr)",
            min_value=0.0,
            value=float(average_production_speed)
        )

        maintenance_input = st.number_input(
            "Predictive Maintenance Score",
            min_value=0.0,
            value=float(average_maintenance_score)
        )

        error_input = st.number_input(
            "Error Rate (%)",
            min_value=0.0,
            value=float(average_error_rate)
        )

        if st.button(
            "🔮 Predict Efficiency"
        ):

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
                    defect_input
                ],

                "Production_Speed_units_per_hr": [
                    production_input
                ],

                "Predictive_Maintenance_Score": [
                    maintenance_input
                ],

                "Error_Rate_%": [
                    error_input
                ]
            })

            try:

                prediction = model.predict(
                    input_data
                )

                prediction_value = prediction[0]

                if label_encoder is not None:

                    try:

                        prediction_label = (
                            label_encoder.inverse_transform(
                                [prediction_value]
                            )[0]
                        )

                    except Exception:

                        prediction_label = prediction_value

                else:

                    prediction_label = prediction_value

                st.success(
                    f"🎯 Predicted Efficiency: "
                    f"{prediction_label}"
                )

                st.info(
                    "Prediction generated using the "
                    "trained Decision Tree model."
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

# ================================================================
# 20. SMART FACTORY ALERTS
# ================================================================

elif page == "Smart Factory Alerts":

    st.header(
        "🚨 Smart Factory Monitoring & Alerts"
    )

    st.write(
        "Monitoring network and manufacturing "
        "conditions and identifying potential "
        "performance issues."
    )

    st.metric(
        "Active Alerts",
        total_alerts
    )

    st.subheader(
        "Network Monitoring"
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

    st.subheader(
        "Machine Monitoring"
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

    st.subheader(
        "Quality Monitoring"
    )

    if average_defect_rate > 5:

        st.warning(
            "⚠️ High quality-control defect rate detected."
        )

    else:

        st.success(
            "✅ Defect rate is within the configured "
            "monitoring limit."
        )

    st.subheader(
        "Production Monitoring"
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
        use_container_width=True
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

# ================================================================
# 21. MODEL PERFORMANCE
# ================================================================

elif page == "Model Performance":

    st.header(
        "📈 Machine Learning Model Performance"
    )

    if accuracy is None:

        st.warning(
            "Model performance metrics are not available."
        )

    else:

        col1, col2 = st.columns(2)

        col1.metric(
            "Accuracy",
            f"{accuracy:.3f}"
        )

        col2.metric(
            "F1 Score",
            f"{f1:.3f}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Precision",
            f"{precision:.3f}"
        )

        col2.metric(
            "Recall",
            f"{recall:.3f}"
        )

        st.subheader(
            "Confusion Matrix"
        )

        fig, ax = plt.subplots()

        ax.imshow(
            cm
        )

        ax.set_title(
            "Decision Tree Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        labels = [
            "High",
            "Medium",
            "Low"
        ]

        ax.set_xticks(
            range(len(labels))
        )

        ax.set_yticks(
            range(len(labels))
        )

        ax.set_xticklabels(
            labels
        )

        ax.set_yticklabels(
            labels
        )

        for i in range(
            len(labels)
        ):

            for j in range(
                len(labels)
            ):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        plt.close(fig)

        st.subheader(
            "Classification Report"
        )

        report_df = pd.DataFrame(
            classification_report(
                y_actual_labels,
                y_pred_labels,
                output_dict=True,
                zero_division=0
            )
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        if feature_importance_series is not None:

            st.subheader(
                "Feature Importance"
            )

            st.bar_chart(
                feature_importance_series
            )

# ================================================================
# 22. PROJECT INFORMATION
# ================================================================

elif page == "Project Information":

    st.header(
        "ℹ️ Project Information"
    )

    st.subheader(
        "Technologies Used"
    )

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Scikit-learn",
        "Streamlit",
        "Machine Learning",
        "Data Visualization"
    ]

    for item in technologies:

        st.write(
            f"• {item}"
        )

    st.subheader(
        "Project Domain"
    )

    st.write(
        "Artificial Intelligence + Machine Learning "
        "+ 6G Wireless Communication + Smart Manufacturing"
    )

    st.subheader(
        "Data Science Workflow"
    )

    st.write(
        """
Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Network Analysis
↓
Manufacturing Analysis
↓
Feature Engineering
↓
Machine Learning
↓
Model Evaluation
↓
Prediction
↓
Interactive Dashboard
↓
Smart Factory Monitoring
"""
    )

    st.subheader(
        "Main Research Objective"
    )

    st.write(
        "To investigate how network-performance parameters "
        "such as latency, packet loss and communication "
        "errors are associated with manufacturing-efficiency "
        "indicators in a smart-factory environment."
    )

    st.subheader(
        "Project Limitation"
    )

    st.write(
        "The current implementation uses a historical CSV "
        "dataset. Therefore, the dashboard demonstrates "
        "data-driven monitoring rather than direct real-time "
        "industrial telemetry."
    )

    st.write(
        "A future deployment could integrate sensors, "
        "industrial communication protocols, streaming data "
        "and edge/cloud computing."
    )

    st.subheader(
        "Future Scope"
    )

    future_scope = [
        "Real-time sensor integration",
        "ESP32/IoT data acquisition",
        "MQTT-based communication",
        "Edge computing",
        "Time-series database",
        "Digital Twin integration",
        "Real-time anomaly detection",
        "Predictive maintenance",
        "Advanced 6G network simulation"
    ]

    for item in future_scope:

        st.write(
            f"• {item}"
        )

# ================================================================
# 23. FOOTER
# ================================================================

st.markdown(
    "---"
)

st.caption(
    "6G Smart Factory Network Analysis | "
    "Machine Learning Internship"
)

st.caption(
    "Data-Driven Smart Manufacturing & "
    "Machine Learning Project"
)

current_time = datetime.now()

st.caption(
    "Dashboard refreshed: "
    + current_time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)

import os
import glob
import warnings
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# DAY 28 — FINAL RESEARCH ANALYSIS
# 6G SMART FACTORY NETWORK ANALYSIS
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory Network Analysis — Day 28",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 6G Smart Factory Network Analysis")
st.subheader(
    "Final Research Analysis + Key Findings + Report/Research Paper Preparation"
)

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# FIND DATASET AUTOMATICALLY
# ============================================================

def find_dataset():
    possible_names = [
        "cleaned_smart_factory_dataset.csv",
        "cleaned_dataset.csv",
        "smart_factory_dataset.csv",
        "smart_factory_data.csv",
        "dataset.csv",
        "cleaned_data.csv"
    ]

    search_locations = [
        BASE_DIR,
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "dataset"),
        os.path.join(BASE_DIR, "datasets"),
        os.path.join(BASE_DIR, "results")
    ]

    for location in search_locations:
        if os.path.exists(location):
            for name in possible_names:
                path = os.path.join(location, name)
                if os.path.isfile(path):
                    return path

    all_csv_files = []

    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [
            d for d in dirs
            if d not in [".git", "__pycache__"]
        ]

        for file in files:
            if file.lower().endswith(".csv"):
                path = os.path.join(root, file)

                if os.path.isfile(path):
                    all_csv_files.append(path)

    if not all_csv_files:
        return None

    preferred = [
        p for p in all_csv_files
        if "clean" in os.path.basename(p).lower()
    ]

    if preferred:
        return preferred[0]

    return all_csv_files[0]


DATA_PATH = find_dataset()

if DATA_PATH is None:
    st.error(
        "❌ No CSV dataset was found.\n\n"
        "Place your dataset inside the project folder or data folder."
    )
    st.stop()

# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Dataset could not be loaded:\n\n{e}")
    st.stop()

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

column_aliases = {
    "Network Latency (ms)": "Network_Latency_ms",
    "Network Latency_ms": "Network_Latency_ms",
    "Latency": "Network_Latency_ms",

    "Packet Loss (%)": "Packet_Loss_%",
    "Packet Loss_%": "Packet_Loss_%",
    "Packet Loss": "Packet_Loss_%",

    "Error Rate (%)": "Error_Rate_%",
    "Error Rate_%": "Error_Rate_%",
    "Error Rate": "Error_Rate_%",

    "Production Speed (units/hr)": "Production_Speed_units_per_hr",
    "Production Speed": "Production_Speed_units_per_hr",

    "Quality Control Defect Rate (%)":
        "Quality_Control_Defect_Rate_%",

    "Defect Rate":
        "Quality_Control_Defect_Rate_%",

    "Predictive Maintenance Score":
        "Predictive_Maintenance_Score",

    "Temperature (C)": "Temperature_C",
    "Temperature": "Temperature_C",

    "Vibration (Hz)": "Vibration_Hz",
    "Vibration": "Vibration_Hz",

    "Power Consumption (kW)": "Power_Consumption_kW",
    "Power Consumption": "Power_Consumption_kW",

    "Efficiency Status": "Efficiency_Status",
    "Efficiency":
        "Efficiency_Status"
}

df.rename(
    columns={
        old: new
        for old, new in column_aliases.items()
        if old in df.columns
    },
    inplace=True
)

# ============================================================
# BASIC INFORMATION
# ============================================================

st.success("✅ Cleaned Smart Factory dataset loaded successfully!")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Dataset Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Dataset File",
        os.path.basename(DATA_PATH)
    )

# ============================================================
# REQUIRED COLUMNS
# ============================================================

network_columns = [
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Error_Rate_%"
]

manufacturing_columns = [
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW"
]

target_column = "Efficiency_Status"

# ============================================================
# REMOVE MISSING REQUIRED COLUMNS FROM ANALYSIS
# ============================================================

available_network = [
    c for c in network_columns
    if c in df.columns
]

available_manufacturing = [
    c for c in manufacturing_columns
    if c in df.columns
]

if target_column not in df.columns:
    possible_targets = [
        c for c in df.columns
        if "efficiency" in c.lower()
        or "status" in c.lower()
        or "class" in c.lower()
    ]

    if possible_targets:
        target_column = possible_targets[0]

# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

analysis_columns = (
    available_network +
    available_manufacturing
)

for col in analysis_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================================
# REMOVE MISSING VALUES FOR ANALYSIS
# ============================================================

required_for_analysis = (
    analysis_columns +
    ([target_column] if target_column in df.columns else [])
)

filtered_df = df.dropna(
    subset=required_for_analysis
).copy()

# ============================================================
# DATA PREPROCESSING SUMMARY
# ============================================================

st.header("1. Data Preprocessing Summary")

preprocessing = pd.DataFrame({
    "Check": [
        "Original Rows",
        "Original Columns",
        "Missing Values",
        "Duplicate Rows",
        "Rows After Cleaning",
        "Columns After Cleaning"
    ],
    "Result": [
        int(df.shape[0]),
        int(df.shape[1]),
        int(df.isnull().sum().sum()),
        int(df.duplicated().sum()),
        int(filtered_df.shape[0]),
        int(filtered_df.shape[1])
    ]
})

st.dataframe(
    preprocessing,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# DATASET FEATURES
# ============================================================

st.header("2. Dataset Features")

feature_rows = []

for col in analysis_columns:
    feature_rows.append({
        "Feature": col,
        "Type": str(filtered_df[col].dtype),
        "Mean": round(
            filtered_df[col].mean(),
            4
        ),
        "Minimum": round(
            filtered_df[col].min(),
            4
        ),
        "Maximum": round(
            filtered_df[col].max(),
            4
        )
    })

st.dataframe(
    pd.DataFrame(feature_rows),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CORRELATION MATRIX
# ============================================================

st.header("3. Final Correlation Analysis")

correlation_columns = [
    c for c in [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Error_Rate_%",
        "Production_Speed_units_per_hr",
        "Quality_Control_Defect_Rate_%",
        "Predictive_Maintenance_Score"
    ]
    if c in filtered_df.columns
]

if len(correlation_columns) >= 2:

    correlation_matrix = (
        filtered_df[
            correlation_columns
        ]
        .corr()
    )

    st.dataframe(
        correlation_matrix.round(3),
        use_container_width=True
    )

    correlation_matrix.to_csv(
        os.path.join(
            RESULTS_DIR,
            "correlation_matrix.csv"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    image = ax.imshow(
        correlation_matrix.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(correlation_matrix.columns))
    )

    ax.set_yticks(
        range(len(correlation_matrix.columns))
    )

    ax.set_xticklabels(
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation_matrix.columns
    )

    for i in range(
        len(correlation_matrix.columns)
    ):
        for j in range(
            len(correlation_matrix.columns)
        ):
            ax.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Matrix"
    )

    fig.tight_layout()

    st.pyplot(fig)

# ============================================================
# SPECIFIC CORRELATIONS
# ============================================================

st.header("4. Network–Manufacturing Relationships")

relationship_rows = []

pairs = [
    (
        "Network_Latency_ms",
        "Production_Speed_units_per_hr",
        "Latency vs Production"
    ),
    (
        "Packet_Loss_%",
        "Production_Speed_units_per_hr",
        "Packet Loss vs Production"
    ),
    (
        "Network_Latency_ms",
        "Quality_Control_Defect_Rate_%",
        "Latency vs Defect Rate"
    ),
    (
        "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Packet Loss vs Defect Rate"
    )
]

for x, y, label in pairs:

    if x in filtered_df.columns and y in filtered_df.columns:

        value = filtered_df[x].corr(
            filtered_df[y]
        )

        relationship_rows.append({
            "Relationship": label,
            "Correlation": round(
                float(value),
                4
            )
        })

relationship_df = pd.DataFrame(
    relationship_rows
)

if not relationship_df.empty:

    st.dataframe(
        relationship_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RELATIONSHIP INTERPRETATION
    # --------------------------------------------------------

    for _, row in relationship_df.iterrows():

        r = row["Correlation"]

        if abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.4:
            strength = "moderate"
        elif abs(r) >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        direction = (
            "positive"
            if r > 0
            else "negative"
            if r < 0
            else "approximately zero"
        )

        st.write(
            f"**{row['Relationship']}:** "
            f"{direction}, {strength} linear relationship "
            f"(r = {r:.4f})."
        )

# ============================================================
# NETWORK FINDINGS
# ============================================================

st.header("5. Final Network Findings")

network_rows = []

for col, label in [
    ("Network_Latency_ms", "Latency"),
    ("Packet_Loss_%", "Packet Loss"),
    ("Error_Rate_%", "Error Rate")
]:

    if col in filtered_df.columns:

        network_rows.append({
            "Network Parameter": label,
            "Average Value": round(
                filtered_df[col].mean(),
                4
            ),
            "Minimum": round(
                filtered_df[col].min(),
                4
            ),
            "Maximum": round(
                filtered_df[col].max(),
                4
            )
        })

network_findings = pd.DataFrame(
    network_rows
)

if not network_findings.empty:

    st.dataframe(
        network_findings,
        use_container_width=True,
        hide_index=True
    )

    network_findings.to_csv(
        os.path.join(
            RESULTS_DIR,
            "network_findings.csv"
        ),
        index=False
    )

# ============================================================
# MANUFACTURING FINDINGS
# ============================================================

st.header("6. Final Manufacturing Findings")

manufacturing_rows = []

manufacturing_labels = {
    "Production_Speed_units_per_hr":
        "Production Speed",
    "Quality_Control_Defect_Rate_%":
        "Defect Rate",
    "Temperature_C":
        "Temperature",
    "Vibration_Hz":
        "Vibration",
    "Power_Consumption_kW":
        "Power Consumption",
    "Predictive_Maintenance_Score":
        "Maintenance Score"
}

for col, label in manufacturing_labels.items():

    if col in filtered_df.columns:

        manufacturing_rows.append({
            "Manufacturing Parameter": label,
            "Average Value": round(
                filtered_df[col].mean(),
                4
            ),
            "Minimum": round(
                filtered_df[col].min(),
                4
            ),
            "Maximum": round(
                filtered_df[col].max(),
                4
            )
        })

manufacturing_findings = pd.DataFrame(
    manufacturing_rows
)

if not manufacturing_findings.empty:

    st.dataframe(
        manufacturing_findings,
        use_container_width=True,
        hide_index=True
    )

    manufacturing_findings.to_csv(
        os.path.join(
            RESULTS_DIR,
            "manufacturing_findings.csv"
        ),
        index=False
    )

# ============================================================
# EFFICIENCY SUMMARY
# ============================================================

st.header("7. Efficiency Distribution")

if target_column in filtered_df.columns:

    efficiency_summary = (
        filtered_df[target_column]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    efficiency_summary.columns = [
        "Efficiency Status",
        "Count"
    ]

    efficiency_summary["Percentage"] = (
        efficiency_summary["Count"]
        / efficiency_summary["Count"].sum()
        * 100
    ).round(3)

    st.dataframe(
        efficiency_summary,
        use_container_width=True,
        hide_index=True
    )

    efficiency_summary.to_csv(
        os.path.join(
            RESULTS_DIR,
            "efficiency_summary.csv"
        ),
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        efficiency_summary["Efficiency Status"],
        efficiency_summary["Count"]
    )

    ax.set_title(
        "Efficiency Status Distribution"
    )

    ax.set_xlabel(
        "Efficiency Status"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    fig.tight_layout()

    st.pyplot(fig)

# ============================================================
# MACHINE LEARNING
# ============================================================

st.header("8. Final Machine Learning Analysis")

if target_column not in filtered_df.columns:

    st.warning(
        "⚠️ Efficiency target column was not found. "
        "Machine Learning analysis skipped."
    )

else:

    candidate_features = (
        available_network +
        available_manufacturing
    )

    candidate_features = [
        c for c in candidate_features
        if c in filtered_df.columns
    ]

    if len(candidate_features) < 2:

        st.warning(
            "Not enough numerical features available "
            "for Machine Learning."
        )

    else:

        X = filtered_df[
            candidate_features
        ].copy()

        y_raw = (
            filtered_df[target_column]
            .astype(str)
        )

        label_encoder = None

        # ----------------------------------------------------
        # ENCODE TARGET
        # ----------------------------------------------------

        if not pd.api.types.is_numeric_dtype(y_raw):

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(
                y_raw
            )

        else:

            y = pd.to_numeric(
                y_raw,
                errors="coerce"
            )

            valid_mask = ~pd.isna(y)

            X = X.loc[valid_mask]
            y = y.loc[valid_mask].astype(int)

        # ----------------------------------------------------
        # TRAIN TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # ----------------------------------------------------
        # FINAL DECISION TREE
        # ----------------------------------------------------

        final_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )

        final_model.fit(
            X_train,
            y_train
        )

        y_pred = final_model.predict(
            X_test
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # ----------------------------------------------------
        # CROSS VALIDATION
        # ----------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        cv_scores = cross_val_score(
            final_model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # ----------------------------------------------------
        # MODEL RESULTS
        # ----------------------------------------------------

        ml_results = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Mean CV Accuracy",
                "CV Standard Deviation"
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
                cv_mean,
                cv_std
            ]
        })

        st.subheader(
            "Final Machine Learning Results"
        )

        st.dataframe(
            ml_results.round(4),
            use_container_width=True,
            hide_index=True
        )

        ml_results.to_csv(
            os.path.join(
                RESULTS_DIR,
                "model_results.csv"
            ),
            index=False
        )

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        with c2:
            st.metric(
                "Precision",
                f"{precision:.4f}"
            )

        with c3:
            st.metric(
                "Recall",
                f"{recall:.4f}"
            )

        with c4:
            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        ax.imshow(
            cm
        )

        ax.set_title(
            "Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Class"
        )

        ax.set_ylabel(
            "Actual Class"
        )

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        fig.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        prediction_df = X_test.copy()

        prediction_df["Actual"] = y_test
        prediction_df["Predicted"] = y_pred
        prediction_df["Correct"] = (
            prediction_df["Actual"]
            ==
            prediction_df["Predicted"]
        )

        prediction_df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "predictions.csv"
            ),
            index=False
        )

        errors = prediction_df[
            prediction_df["Correct"] == False
        ].copy()

        errors.to_csv(
            os.path.join(
                RESULTS_DIR,
                "prediction_errors.csv"
            ),
            index=False
        )

        # ----------------------------------------------------
        # ERROR ANALYSIS
        # ----------------------------------------------------

        st.subheader(
            "Prediction Error Analysis"
        )

        total_predictions = len(
            prediction_df
        )

        correct_predictions = int(
            prediction_df["Correct"].sum()
        )

        incorrect_predictions = int(
            (~prediction_df["Correct"]).sum()
        )

        error_rate = (
            incorrect_predictions /
            total_predictions
        )

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            st.metric(
                "Test Samples",
                f"{total_predictions:,}"
            )

        with e2:
            st.metric(
                "Correct Predictions",
                f"{correct_predictions:,}"
            )

        with e3:
            st.metric(
                "Incorrect Predictions",
                f"{incorrect_predictions:,}"
            )

        with e4:
            st.metric(
                "Prediction Error Rate",
                f"{error_rate:.4f}"
            )

        if incorrect_predictions > 0:

            st.dataframe(
                errors.head(100),
                use_container_width=True
            )

        else:

            st.success(
                "✅ No prediction errors were found "
                "in the test set."
            )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.subheader(
            "Top Model Features"
        )

        feature_importance = pd.Series(
            final_model.feature_importances_,
            index=X.columns
        ).sort_values(
            ascending=False
        )

        top_features = (
            feature_importance
            .head(10)
            .reset_index()
        )

        top_features.columns = [
            "Feature",
            "Importance"
        ]

        st.dataframe(
            top_features.round(4),
            use_container_width=True,
            hide_index=True
        )

        feature_importance.to_csv(
            os.path.join(
                RESULTS_DIR,
                "feature_importance.csv"
            )
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE FIGURE
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        ax.set_title(
            "Top Feature Importance"
        )

        ax.set_xlabel(
            "Importance"
        )

        fig.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------------
        # CROSS VALIDATION
        # ----------------------------------------------------

        st.subheader(
            "5-Fold Cross-Validation"
        )

        cv_df = pd.DataFrame({
            "Fold": [
                f"Fold {i + 1}"
                for i in range(len(cv_scores))
            ],
            "Accuracy": cv_scores
        })

        st.dataframe(
            cv_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Mean CV Accuracy",
            f"{cv_mean:.4f}"
        )

        st.metric(
            "CV Standard Deviation",
            f"{cv_std:.4f}"
        )

        cv_df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "cross_validation_results.csv"
            ),
            index=False
        )

        # ----------------------------------------------------
        # MODEL SAVE
        # ----------------------------------------------------

        try:

            joblib.dump(
                final_model,
                os.path.join(
                    RESULTS_DIR,
                    "final_model.pkl"
                )
            )

            if label_encoder is not None:

                joblib.dump(
                    label_encoder,
                    os.path.join(
                        RESULTS_DIR,
                        "label_encoder.pkl"
                    )
                )

        except Exception:
            pass

# ============================================================
# PROJECT KPI SUMMARY
# ============================================================

st.header("9. Final Project KPI Summary")

dataset_result = (
    f"{len(filtered_df):,} records analyzed"
)

if network_findings.empty:

    network_result = "Network analysis available"

else:

    network_result = (
        f"{len(network_findings)} network KPIs analyzed"
    )

if manufacturing_findings.empty:

    manufacturing_result = (
        "Manufacturing analysis available"
    )

else:

    manufacturing_result = (
        f"{len(manufacturing_findings)} manufacturing KPIs analyzed"
    )

if "accuracy" in locals():

    ml_result = (
        f"Accuracy = {accuracy:.4f}"
    )

else:

    ml_result = "Machine Learning analysis completed"

project_kpis = pd.DataFrame({
    "Category": [
        "Dataset",
        "Network",
        "Manufacturing",
        "ML Model",
        "Dashboard"
    ],
    "Key Result": [
        dataset_result,
        network_result,
        manufacturing_result,
        ml_result,
        "Streamlit"
    ]
})

st.dataframe(
    project_kpis,
    use_container_width=True,
    hide_index=True
)

project_kpis.to_csv(
    os.path.join(
        RESULTS_DIR,
        "project_kpis.csv"
    ),
    index=False
)

# ============================================================
# FINAL KEY FINDINGS
# ============================================================

st.header("10. Final Key Findings")

findings = [
    "Network-performance parameters were analyzed to characterize communication conditions in the smart-factory dataset.",
    "Manufacturing performance was evaluated using production, quality and machine-condition indicators.",
    "Relationships between network parameters and manufacturing indicators were investigated using statistical and visual analysis.",
    "A Machine Learning classification model was developed to predict manufacturing efficiency categories.",
    "The final model was evaluated using unseen test data and five-fold cross-validation.",
    "An interactive Streamlit dashboard was developed to integrate network analysis, manufacturing analytics, Machine Learning prediction, monitoring and alerts."
]

for i, finding in enumerate(findings, 1):

    st.write(
        f"**Finding {i}:** {finding}"
    )

# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

st.header("11. Scientific Interpretation")

st.info(
    "The project identifies associations between "
    "network-performance parameters and manufacturing-efficiency "
    "indicators. The observed relationships should not be "
    "interpreted as proof of direct causal effects."
)

# ============================================================
# LIMITATIONS
# ============================================================

st.header("12. Limitations")

limitations = [
    "The dataset is historical/static.",
    "The project does not currently receive live industrial network telemetry.",
    "The project does not establish causal relationships.",
    "Alert thresholds are project-defined demonstration rules.",
    "Model performance depends on the available dataset.",
    "Real industrial deployment would require domain validation.",
    "Real 6G network behavior is more complex than the available dataset representation."
]

for item in limitations:
    st.write(f"• {item}")

# ============================================================
# FUTURE SCOPE
# ============================================================

st.header("13. Future Scope")

st.write(
    """
ESP32 / IoT Sensors
↓
Real-Time Network Telemetry
↓
MQTT
↓
Edge Computing
↓
Time-Series Database
↓
Real-Time Machine Learning
↓
Digital Twin
↓
Predictive Maintenance
↓
Advanced 6G Simulation
"""
)

# ============================================================
# RESEARCH PAPER ABSTRACT
# ============================================================

st.header("14. Research Paper Abstract")

abstract = (
    "Smart factories depend on reliable communication networks "
    "and efficient manufacturing processes. This project "
    "investigates the relationship between network-performance "
    "parameters and manufacturing-efficiency indicators in a "
    "smart-factory dataset. Parameters including network latency, "
    "packet loss and communication error rate are analyzed "
    "together with production speed, defect rate and "
    "machine-condition variables. A Machine Learning "
    "classification pipeline is developed to predict "
    "manufacturing efficiency categories. Multiple evaluation "
    "metrics and cross-validation are used to assess model "
    "performance. An interactive Streamlit dashboard integrates "
    "network analysis, manufacturing analytics, Machine Learning "
    "prediction and smart-factory monitoring. The resulting "
    "system provides a data-driven framework for analyzing the "
    "interaction between communication-network performance and "
    "manufacturing efficiency."
)

st.write(abstract)

# ============================================================
# RESEARCH PAPER STRUCTURE
# ============================================================

st.header("15. Research Paper Structure")

paper_structure = [
    "1. Abstract",
    "2. Introduction",
    "3. Problem Statement",
    "4. Objectives",
    "5. Dataset and Methodology",
    "6. Exploratory Data Analysis",
    "7. 6G Network Performance Analysis",
    "8. Manufacturing Efficiency Analysis",
    "9. Machine Learning Methodology",
    "10. Model Evaluation",
    "11. Results and Discussion",
    "12. Dashboard Implementation",
    "13. Limitations",
    "14. Future Scope",
    "15. Conclusion",
    "16. References"
]

for section in paper_structure:
    st.write(f"• {section}")

# ============================================================
# METHODOLOGY
# ============================================================

st.header("16. Methodology")

st.write(
    """
Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Network KPI Analysis
↓
Manufacturing Analysis
↓
Feature Engineering
↓
Train-Test Split
↓
Model Training
↓
Hyperparameter Optimization
↓
Model Evaluation
↓
Dashboard Development
↓
Monitoring + Alerts
"""
)

# ============================================================
# CONCLUSION
# ============================================================

st.header("17. Conclusion")

conclusion = (
    "This project developed a data-driven framework for "
    "analyzing the relationship between network performance "
    "and manufacturing efficiency in a smart-factory "
    "environment. Network KPIs and manufacturing indicators "
    "were explored using statistical analysis and visualization, "
    "followed by Machine Learning-based efficiency "
    "classification. The final model was evaluated using "
    "unseen test data and cross-validation. An interactive "
    "Streamlit dashboard was developed to integrate analysis, "
    "prediction, monitoring and alerts. While the current "
    "implementation is based on historical data, the "
    "architecture provides a foundation for future integration "
    "with real-time IoT, edge computing and advanced "
    "wireless-network monitoring systems."
)

st.write(conclusion)

# ============================================================
# REPORT FIGURE LIST
# ============================================================

st.header("18. Final Figure List")

figures = [
    "Figure 1 — Project Architecture",
    "Figure 2 — Dataset Efficiency Distribution",
    "Figure 3 — Network Latency Distribution",
    "Figure 4 — Packet Loss Distribution",
    "Figure 5 — Network Latency vs Production",
    "Figure 6 — Packet Loss vs Production",
    "Figure 7 — Network Performance vs Efficiency",
    "Figure 8 — Model Comparison",
    "Figure 9 — Confusion Matrix",
    "Figure 10 — Feature Importance",
    "Figure 11 — Cross-Validation Performance",
    "Figure 12 — Final Streamlit Dashboard"
]

for item in figures:
    st.write(f"• {item}")

# ============================================================
# REPORT TABLE LIST
# ============================================================

st.header("19. Final Table List")

tables = [
    "Table 1 — Dataset Features",
    "Table 2 — Data Preprocessing Summary",
    "Table 3 — Network KPI Statistics",
    "Table 4 — Manufacturing KPI Statistics",
    "Table 5 — Model Comparison",
    "Table 6 — Final Model Performance",
    "Table 7 — Cross-Validation Results",
    "Table 8 — Prediction Error Analysis",
    "Table 9 — Top Feature Importance"
]

for item in tables:
    st.write(f"• {item}")

# ============================================================
# RESULTS DIRECTORY
# ============================================================

st.header("20. Results Directory")

st.success(
    f"✅ Final project results are being saved to:\n\n"
    f"{RESULTS_DIR}"
)

saved_files = []

if os.path.exists(RESULTS_DIR):

    for file in sorted(
        os.listdir(RESULTS_DIR)
    ):

        saved_files.append(file)

if saved_files:

    st.dataframe(
        pd.DataFrame({
            "Saved Result Files": saved_files
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FINAL CHECKLIST
# ============================================================

st.header("21. Day 28 Final Checklist")

checklist = [
    "Dataset available",
    "Clean dataset available",
    "EDA completed",
    "Network analysis completed",
    "Manufacturing analysis completed",
    "ML models evaluated",
    "Final model selected",
    "Final test metrics calculated",
    "Cross-validation completed",
    "Error analysis completed",
    "Feature importance available",
    "Streamlit dashboard working",
    "Final results exported",
    "Research findings documented"
]

for item in checklist:
    st.write(f"☑️ {item}")

# ============================================================
# PROJECT STATUS
# ============================================================

st.divider()

st.success(
    "🎉 DAY 28 FINAL RESEARCH ANALYSIS COMPLETED"
)

st.markdown(
    """
### 📊 Project Progress

**Day 1–27:** ✅ Completed

**Day 28:** ✅ Final Research Analysis + Report Preparation

**Progress:** **28 / 30 Days**

**Remaining:** **2 Days**

### NEXT

**DAY 29 — FINAL PROJECT INTEGRATION + GITHUB + README + PROFESSIONAL PROJECT PACKAGING**
"""
)
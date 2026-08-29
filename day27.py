# ============================================================
# DAY 27
# FINAL MODEL VALIDATION + ROBUSTNESS TESTING
# 6G SMART FACTORY NETWORK ANALYSIS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="6G Smart Factory - Day 27",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🏭 6G Smart Factory Network Analysis")

st.subheader(
    "Final Model Validation + Robustness Testing + Project Results"
)

st.write(
    "Day 27 evaluates the final Machine Learning model "
    "using unseen test data, cross-validation, error analysis "
    "and prediction consistency."
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

# ============================================================
# FILE PATHS
# ============================================================

dataset_path = os.path.join(
    DATA_DIR,
    "cleaned_smart_factory.csv"
)

model_path = os.path.join(
    DATA_DIR,
    "decision_tree_day15_model.pkl"
)

encoder_path = os.path.join(
    DATA_DIR,
    "day15_label_encoder.pkl"
)

# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(dataset_path)

    st.success(
        "✅ Cleaned Smart Factory dataset loaded successfully!"
    )

    st.info(
        f"Dataset contains {df.shape[0]} rows "
        f"and {df.shape[1]} columns."
    )

except Exception as e:

    st.error(
        f"❌ Dataset could not be loaded: {e}"
    )

    st.stop()

# ============================================================
# FEATURE LIST
# ============================================================

features = [
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

# ============================================================
# TARGET COLUMN
# ============================================================

possible_targets = [
    "Efficiency_Status",
    "Efficiency",
    "Manufacturing_Efficiency",
    "Efficiency_Category"
]

target_column = None

for column in possible_targets:

    if column in df.columns:

        target_column = column
        break

if target_column is None:

    st.error(
        "❌ Efficiency target column could not be found."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()

# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    st.error(
        f"❌ Missing ML features: {missing_features}"
    )

    st.stop()

# ============================================================
# PREPARE X
# ============================================================

X = df[features].copy()

# Convert explicitly to numeric
for column in features:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# ============================================================
# PREPARE Y
# ============================================================

y_original = df[target_column].astype(str)

# ============================================================
# REMOVE MISSING VALUES
# ============================================================

valid_rows = (
    X.notna().all(axis=1)
    &
    y_original.notna()
)

X = X.loc[valid_rows].copy()

y_original = y_original.loc[valid_rows].copy()

# ============================================================
# CONVERT TO NUMPY
# IMPORTANT FIX FOR ARROW / PANDAS ERROR
# ============================================================

X = X.to_numpy(dtype=np.float64)

y_original = y_original.to_numpy()

# ============================================================
# LOAD MODEL
# ============================================================

model = None
label_encoder = None

try:

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        st.success(
            "✅ Decision Tree Machine Learning model loaded successfully!"
        )

    else:

        st.error(
            "❌ Decision Tree model file not found."
        )

        st.write(model_path)

        st.stop()

except Exception as e:

    st.error(
        f"❌ Model loading error: {e}"
    )

    st.stop()

# ============================================================
# LOAD LABEL ENCODER
# ============================================================

try:

    if os.path.exists(encoder_path):

        label_encoder = joblib.load(
            encoder_path
        )

        st.success(
            "✅ Label encoder loaded successfully!"
        )

    else:

        st.warning(
            "⚠️ Label encoder file not found. "
            "Attempting to use model classes."
        )

except Exception as e:

    st.warning(
        f"⚠️ Label encoder could not be loaded: {e}"
    )

# ============================================================
# ENCODE TARGET
# ============================================================

try:

    if label_encoder is not None:

        y = label_encoder.transform(
            y_original
        )

    else:

        # Fallback using model classes
        model_classes = np.array(
            model.classes_
        )

        y = np.array(
            [
                np.where(
                    model_classes == label
                )[0][0]
                for label in y_original
            ]
        )

except Exception as e:

    st.error(
        f"❌ Target encoding failed: {e}"
    )

    st.write(
        "Actual target labels:"
    )

    st.write(
        np.unique(y_original)
    )

    st.write(
        "Model classes:"
    )

    st.write(
        getattr(model, "classes_", "Unavailable")
    )

    st.stop()

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except Exception as e:

    st.error(
        f"❌ Train/test split failed: {e}"
    )

    st.stop()

# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

try:

    y_pred = model.predict(
        X_test
    )

    # Convert predictions to NumPy
    y_pred = np.asarray(
        y_pred
    )

except Exception as e:

    st.error(
        f"❌ Prediction failed: {e}"
    )

    st.stop()

# ============================================================
# TEST METRICS
# ============================================================

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

# ============================================================
# TRAINING PERFORMANCE
# ============================================================

try:

    train_predictions = model.predict(
        X_train
    )

    train_predictions = np.asarray(
        train_predictions
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

except Exception:

    train_accuracy = np.nan

# ============================================================
# ACCURACY GAP
# ============================================================

if not np.isnan(train_accuracy):

    accuracy_gap = (
        train_accuracy
        -
        accuracy
    )

else:

    accuracy_gap = np.nan

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

try:

    class_report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

except Exception:

    class_report = "Classification report unavailable."

# ============================================================
# CONFUSION MATRIX
# ============================================================

try:

    cm = confusion_matrix(
        y_test,
        y_pred
    )

except Exception as e:

    st.error(
        f"❌ Confusion matrix error: {e}"
    )

    cm = None

# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

st.header(
    "📊 Final Model Results"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

col2.metric(
    "Precision",
    f"{precision:.4f}"
)

col3.metric(
    "Recall",
    f"{recall:.4f}"
)

col4.metric(
    "F1 Score",
    f"{f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "📋 Classification Report"
)

st.code(
    class_report
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader(
    "🔲 Confusion Matrix"
)

if cm is not None:

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title(
        "Final Decision Tree Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    # Use original class names if possible
    if label_encoder is not None:

        try:

            class_names = (
                label_encoder.classes_
            )

            ax.set_xticks(
                range(len(class_names))
            )

            ax.set_yticks(
                range(len(class_names))
            )

            ax.set_xticklabels(
                class_names
            )

            ax.set_yticklabels(
                class_names
            )

        except Exception:

            pass

    st.pyplot(
        fig
    )

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Actual vs Predicted Class Distribution"
)

actual_distribution = pd.Series(
    y_test
).value_counts().sort_index()

predicted_distribution = pd.Series(
    y_pred
).value_counts().sort_index()

distribution_df = pd.DataFrame({
    "Actual": actual_distribution,
    "Predicted": predicted_distribution
}).fillna(0)

st.dataframe(
    distribution_df,
    use_container_width=True
)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

results_df = pd.DataFrame({

    "Actual_Encoded": y_test,

    "Predicted_Encoded": y_pred

})

# Convert labels back to names
if label_encoder is not None:

    try:

        results_df["Actual"] = (
            label_encoder.inverse_transform(
                results_df["Actual_Encoded"]
            )
        )

        results_df["Predicted"] = (
            label_encoder.inverse_transform(
                results_df["Predicted_Encoded"]
            )
        )

    except Exception:

        results_df["Actual"] = (
            results_df["Actual_Encoded"]
        )

        results_df["Predicted"] = (
            results_df["Predicted_Encoded"]
        )

else:

    results_df["Actual"] = (
        results_df["Actual_Encoded"]
    )

    results_df["Predicted"] = (
        results_df["Predicted_Encoded"]
    )

# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct_predictions = (
    results_df["Actual_Encoded"]
    ==
    results_df["Predicted_Encoded"]
)

correct_count = int(
    correct_predictions.sum()
)

incorrect_count = (
    len(results_df)
    -
    correct_count
)

prediction_error_rate = (
    incorrect_count
    /
    len(results_df)
)

# ============================================================
# DISPLAY PREDICTION COUNTS
# ============================================================

st.subheader(
    "🎯 Prediction Analysis"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test Samples",
    len(results_df)
)

col2.metric(
    "Correct Predictions",
    correct_count
)

col3.metric(
    "Incorrect Predictions",
    incorrect_count
)

st.metric(
    "Prediction Error Rate",
    f"{prediction_error_rate:.4f}"
)

# ============================================================
# SHOW ACTUAL VS PREDICTED
# ============================================================

st.subheader(
    "🔍 Actual vs Predicted Values"
)

st.dataframe(
    results_df.head(50),
    use_container_width=True
)

# ============================================================
# ERROR ANALYSIS
# ============================================================

error_df = results_df[
    results_df["Actual_Encoded"]
    !=
    results_df["Predicted_Encoded"]
].copy()

st.subheader(
    "❌ Prediction Error Analysis"
)

st.write(
    f"Incorrect predictions: {len(error_df)}"
)

if len(error_df) > 0:

    st.dataframe(
        error_df.head(100),
        use_container_width=True
    )

else:

    st.success(
        "✅ No prediction errors found in the test set."
    )

# ============================================================
# TRAINING VS TESTING
# ============================================================

st.subheader(
    "📈 Training vs Testing Performance"
)

if not np.isnan(train_accuracy):

    comparison_df = pd.DataFrame({

        "Dataset": [
            "Training",
            "Testing"
        ],

        "Accuracy": [
            train_accuracy,
            accuracy
        ]

    })

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.bar_chart(
        comparison_df.set_index(
            "Dataset"
        )
    )

    st.write(
        f"Training-Test Accuracy Gap: "
        f"{accuracy_gap:.4f}"
    )

    if accuracy_gap > 0.10:

        st.warning(
            "⚠️ The training-test accuracy gap "
            "is relatively large and may indicate "
            "possible overfitting."
        )

    else:

        st.success(
            "✅ The training-test accuracy gap "
            "does not indicate a large performance difference."
        )

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

st.subheader(
    "🔄 5-Fold Cross-Validation"
)

try:

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_mean = cv_scores.mean()

    cv_std = cv_scores.std()

    col1, col2 = st.columns(2)

    col1.metric(
        "Mean CV Accuracy",
        f"{cv_mean:.4f}"
    )

    col2.metric(
        "CV Standard Deviation",
        f"{cv_std:.4f}"
    )

    cv_df = pd.DataFrame({

        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],

        "Accuracy": cv_scores

    })

    st.dataframe(
        cv_df,
        use_container_width=True
    )

    st.bar_chart(
        cv_df.set_index(
            "Fold"
        )
    )

except Exception as e:

    cv_scores = np.array([])

    cv_mean = np.nan

    cv_std = np.nan

    st.warning(
        f"⚠️ Cross-validation could not be completed: {e}"
    )

# ============================================================
# MODEL CONFIDENCE
# ============================================================

st.subheader(
    "🤖 Prediction Confidence Analysis"
)

try:

    probabilities = model.predict_proba(
        X_test
    )

    probabilities = np.asarray(
        probabilities
    )

    max_probability = (
        probabilities.max(
            axis=1
        )
    )

    confidence_df = pd.DataFrame({

        "Actual": results_df["Actual"],

        "Predicted": results_df["Predicted"],

        "Confidence": max_probability

    })

    st.dataframe(
        confidence_df.head(50),
        use_container_width=True
    )

    low_confidence = confidence_df[
        confidence_df["Confidence"] < 0.60
    ]

    st.write(
        f"Predictions with confidence below 60%: "
        f"{len(low_confidence)}"
    )

    if len(low_confidence) > 0:

        st.dataframe(
            low_confidence,
            use_container_width=True
        )

except Exception as e:

    st.info(
        f"Probability analysis unavailable: {e}"
    )

# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

final_results = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Mean CV Accuracy",

        "CV Standard Deviation",

        "Training Accuracy",

        "Testing Accuracy",

        "Prediction Error Rate"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1,

        cv_mean,

        cv_std,

        train_accuracy,

        accuracy,

        prediction_error_rate

    ]

})

st.subheader(
    "📋 Final Performance Summary"
)

st.dataframe(
    final_results.round(4),
    use_container_width=True
)

# ============================================================
# DOWNLOAD FINAL RESULTS
# ============================================================

results_csv = final_results.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Final Model Results",
    results_csv,
    "final_model_results.csv",
    "text/csv"
)

# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

predictions_csv = results_df.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Predictions",
    predictions_csv,
    "model_predictions.csv",
    "text/csv"
)

# ============================================================
# DOWNLOAD ERRORS
# ============================================================

errors_csv = error_df.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Prediction Errors",
    errors_csv,
    "prediction_errors.csv",
    "text/csv"
)

# ============================================================
# FINAL MODEL SUMMARY
# ============================================================

st.subheader(
    "🤖 Final Model Summary"
)

st.write(
    "Model: Decision Tree Classifier"
)

st.write(
    f"Test Accuracy: {accuracy:.4f}"
)

st.write(
    f"Precision: {precision:.4f}"
)

st.write(
    f"Recall: {recall:.4f}"
)

st.write(
    f"F1 Score: {f1:.4f}"
)

if not np.isnan(cv_mean):

    st.write(
        f"Mean CV Accuracy: {cv_mean:.4f}"
    )

    st.write(
        f"CV Standard Deviation: {cv_std:.4f}"
    )

# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

st.subheader(
    "🔬 Scientific Interpretation"
)

st.info(
    "The final classifier was evaluated on unseen test data "
    "using accuracy, precision, recall and F1-score. "
    "Five-fold cross-validation was additionally used to "
    "assess performance consistency across multiple data splits. "
    "Prediction errors were analyzed to identify efficiency "
    "classes that may be more difficult for the model to distinguish."
)

st.warning(
    "Important: The project identifies associations between "
    "network-performance parameters and manufacturing-efficiency "
    "indicators. The results should not be interpreted as proof "
    "of direct causal effects."
)

# ============================================================
# DAY 27 COMPLETION
# ============================================================

st.success(
    "🎉 DAY 27 FINAL MODEL VALIDATION COMPLETED"
)

st.caption(
    "6G Smart Factory Network Analysis | "
    "Machine Learning Internship"
)

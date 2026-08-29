import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

print("LIBRARIES IMPORTED SUCCESSFULLY!")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

print("DATASET LOADED SUCCESSFULLY!")

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

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

X = df[features]

y = df["Efficiency_Status"]

# --------------------------------------------------
# ENCODE TARGET
# --------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTARGET CLASSES")

print(label_encoder.classes_)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTRAINING DATA SHAPE")

print(X_train.shape)

print("\nTESTING DATA SHAPE")

print(X_test.shape)

# --------------------------------------------------
# CREATE MODELS
# --------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

logistic_model.fit(
    X_train,
    y_train
)

decision_tree_model.fit(
    X_train,
    y_train
)

random_forest_model.fit(
    X_train,
    y_train
)

print("\nALL THREE MODELS TRAINED SUCCESSFULLY!")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

logistic_pred = logistic_model.predict(
    X_test
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

random_forest_pred = random_forest_model.predict(
    X_test
)

# --------------------------------------------------
# PREDICTION PROBABILITIES
# --------------------------------------------------

logistic_prob = logistic_model.predict_proba(
    X_test
)

decision_tree_prob = decision_tree_model.predict_proba(
    X_test
)

random_forest_prob = random_forest_model.predict_proba(
    X_test
)

# --------------------------------------------------
# BINARY TARGET FOR ROC-AUC
# --------------------------------------------------

y_test_binary = label_binarize(
    y_test,
    classes=range(len(label_encoder.classes_))
)

# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

logistic_auc = roc_auc_score(
    y_test_binary,
    logistic_prob,
    multi_class="ovr",
    average="weighted"
)

decision_tree_auc = roc_auc_score(
    y_test_binary,
    decision_tree_prob,
    multi_class="ovr",
    average="weighted"
)

random_forest_auc = roc_auc_score(
    y_test_binary,
    random_forest_prob,
    multi_class="ovr",
    average="weighted"
)

print("\nLOGISTIC REGRESSION ROC-AUC")

print(logistic_auc)

print("\nDECISION TREE ROC-AUC")

print(decision_tree_auc)

print("\nRANDOM FOREST ROC-AUC")

print(random_forest_auc)

# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

# --------------------------------------------------
# MODEL EVALUATION TABLE
# --------------------------------------------------

evaluation = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        random_forest_accuracy
    ],

    "ROC_AUC": [
        logistic_auc,
        decision_tree_auc,
        random_forest_auc
    ]

})

print("\nMODEL EVALUATION")

print(evaluation)

# --------------------------------------------------
# CONVERT TO PERCENTAGE
# --------------------------------------------------

evaluation["Accuracy_Percentage"] = (
    evaluation["Accuracy"] * 100
)

evaluation["ROC_AUC_Percentage"] = (
    evaluation["ROC_AUC"] * 100
)

print("\nMODEL PERFORMANCE (%)")

print(
    evaluation[
        [
            "Model",
            "Accuracy_Percentage",
            "ROC_AUC_Percentage"
        ]
    ]
)

# --------------------------------------------------
# BEST MODEL BASED ON ROC-AUC
# --------------------------------------------------

best_auc_model = evaluation.loc[
    evaluation["ROC_AUC"].idxmax()
]

print("\nBEST MODEL BASED ON ROC-AUC")

print(best_auc_model)

# --------------------------------------------------
# BEST MODEL BASED ON ACCURACY
# --------------------------------------------------

best_accuracy_model = evaluation.loc[
    evaluation["Accuracy"].idxmax()
]

print("\nBEST MODEL BASED ON ACCURACY")

print(best_accuracy_model)

# --------------------------------------------------
# CLASSIFICATION REPORTS
# --------------------------------------------------

print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=label_encoder.classes_
    )
)

print("\nDECISION TREE CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        decision_tree_pred,
        target_names=label_encoder.classes_
    )
)

print("\nRANDOM FOREST CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        random_forest_pred,
        target_names=label_encoder.classes_
    )
)

# --------------------------------------------------
# LOGISTIC REGRESSION ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(
    label_encoder.classes_
):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        logistic_prob[:, i]
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} AUC = {roc_auc_value:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "Logistic Regression ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.show()

# --------------------------------------------------
# DECISION TREE ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(
    label_encoder.classes_
):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        decision_tree_prob[:, i]
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} AUC = {roc_auc_value:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "Decision Tree ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.show()

# --------------------------------------------------
# RANDOM FOREST ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(
    label_encoder.classes_
):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        random_forest_prob[:, i]
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} AUC = {roc_auc_value:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "Random Forest ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.show()

# --------------------------------------------------
# FINAL MODEL COMPARISON
# --------------------------------------------------

comparison = evaluation[
    [
        "Model",
        "Accuracy_Percentage",
        "ROC_AUC_Percentage"
    ]
]

print("\nFINAL MODEL COMPARISON")

print(comparison)

print(
    "\nMODEL WITH HIGHEST ROC-AUC:"
)

print(
    best_auc_model["Model"]
)

print(
    "\nMODEL WITH HIGHEST TEST ACCURACY:"
)

print(
    best_accuracy_model["Model"]
)

# --------------------------------------------------
# SAVE FINAL EVALUATION
# --------------------------------------------------

evaluation.to_csv(
    "data/final_model_evaluation_day15.csv",
    index=False
)

print(
    "\nFINAL MODEL EVALUATION SAVED SUCCESSFULLY!"
)

# --------------------------------------------------
# SAVE ROC-AUC RESULTS
# --------------------------------------------------

roc_auc_summary = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "ROC_AUC": [
        logistic_auc,
        decision_tree_auc,
        random_forest_auc
    ]

})

roc_auc_summary.to_csv(
    "data/roc_auc_results.csv",
    index=False
)

print(
    "\nROC-AUC RESULTS SAVED SUCCESSFULLY!"
)

# --------------------------------------------------
# SAVE MODELS
# --------------------------------------------------

joblib.dump(
    logistic_model,
    "data/logistic_regression_day15_model.pkl"
)

joblib.dump(
    decision_tree_model,
    "data/decision_tree_day15_model.pkl"
)

joblib.dump(
    random_forest_model,
    "data/random_forest_day15_model.pkl"
)

joblib.dump(
    label_encoder,
    "data/day15_label_encoder.pkl"
)

print(
    "\nALL TRAINED MODELS SAVED SUCCESSFULLY!"
)

print(
    "\nDAY 15 FINAL MODEL EVALUATION COMPLETED SUCCESSFULLY!"
)
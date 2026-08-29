import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("LIBRARIES IMPORTED SUCCESSFULLY!")

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

print("DATASET LOADED SUCCESSFULLY!")

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

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTARGET CLASSES")
print(label_encoder.classes_)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# LOGISTIC REGRESSION

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_scores = cross_val_score(
    logistic_model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print("\nLOGISTIC REGRESSION CROSS-VALIDATION SCORES")
print(logistic_scores)

logistic_mean = logistic_scores.mean()
logistic_std = logistic_scores.std()

print("\nLOGISTIC REGRESSION MEAN ACCURACY")
print(logistic_mean)

print("\nLOGISTIC REGRESSION STANDARD DEVIATION")
print(logistic_std)


# DECISION TREE

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_scores = cross_val_score(
    decision_tree_model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print("\nDECISION TREE CROSS-VALIDATION SCORES")
print(decision_tree_scores)

decision_tree_mean = decision_tree_scores.mean()
decision_tree_std = decision_tree_scores.std()

print("\nDECISION TREE MEAN ACCURACY")
print(decision_tree_mean)

print("\nDECISION TREE STANDARD DEVIATION")
print(decision_tree_std)


# RANDOM FOREST

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_scores = cross_val_score(
    random_forest_model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy"
)

print("\nRANDOM FOREST CROSS-VALIDATION SCORES")
print(random_forest_scores)

random_forest_mean = random_forest_scores.mean()
random_forest_std = random_forest_scores.std()

print("\nRANDOM FOREST MEAN ACCURACY")
print(random_forest_mean)

print("\nRANDOM FOREST STANDARD DEVIATION")
print(random_forest_std)


# MODEL COMPARISON

cv_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Mean_Accuracy": [
        logistic_mean,
        decision_tree_mean,
        random_forest_mean
    ],

    "Standard_Deviation": [
        logistic_std,
        decision_tree_std,
        random_forest_std
    ]

})

print("\nCROSS-VALIDATION MODEL COMPARISON")
print(cv_results)


# ACCURACY PERCENTAGE

cv_results["Mean_Accuracy_Percentage"] = (
    cv_results["Mean_Accuracy"] * 100
)

print("\nCROSS-VALIDATION ACCURACY (%)")

print(
    cv_results[
        [
            "Model",
            "Mean_Accuracy_Percentage"
        ]
    ]
)


# STANDARD DEVIATION PERCENTAGE

cv_results["Standard_Deviation_Percentage"] = (
    cv_results["Standard_Deviation"] * 100
)

print("\nMODEL PERFORMANCE VARIATION (%)")

print(
    cv_results[
        [
            "Model",
            "Standard_Deviation_Percentage"
        ]
    ]
)


# BEST MODEL

best_model = cv_results.loc[
    cv_results["Mean_Accuracy"].idxmax()
]

print("\nBEST MODEL BASED ON CROSS-VALIDATION")
print(best_model)


# FINAL EVALUATION

final_evaluation = cv_results[
    [
        "Model",
        "Mean_Accuracy_Percentage",
        "Standard_Deviation_Percentage"
    ]
]

print("\nFINAL MODEL EVALUATION")
print(final_evaluation)


# SAVE CROSS-VALIDATION RESULTS

final_evaluation.to_csv(
    "data/cross_validation_results.csv",
    index=False
)

print("\nCROSS-VALIDATION RESULTS SAVED SUCCESSFULLY!")


# SAVE BEST MODEL SUMMARY

best_model_name = best_model["Model"]

best_model_score = best_model[
    "Mean_Accuracy_Percentage"
]

best_model_summary = pd.DataFrame({

    "Best_Model": [
        best_model_name
    ],

    "Mean_CV_Accuracy_Percentage": [
        best_model_score
    ]

})

best_model_summary.to_csv(
    "data/best_model_summary.csv",
    index=False
)

print("\nBEST MODEL SUMMARY SAVED SUCCESSFULLY!")

print("\nDAY 13 CROSS-VALIDATION MODEL EVALUATION COMPLETED SUCCESSFULLY!")
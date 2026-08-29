import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# BASE RANDOM FOREST
# --------------------------------------------------

base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

print("\nBASE RANDOM FOREST CREATED")


# --------------------------------------------------
# FAST HYPERPARAMETER GRID
# --------------------------------------------------

param_grid = {

    "n_estimators": [
        50,
        100
    ],

    "max_depth": [
        None,
        10
    ],

    "min_samples_split": [
        2
    ],

    "min_samples_leaf": [
        1
    ],

    "max_features": [
        "sqrt"
    ]
}


# --------------------------------------------------
# GRID SEARCH
# --------------------------------------------------

grid_search = GridSearchCV(

    estimator=base_model,

    param_grid=param_grid,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1,

    verbose=1
)

print("\nSTARTING FAST HYPERPARAMETER TUNING...")

grid_search.fit(
    X,
    y_encoded
)

print("\nHYPERPARAMETER TUNING COMPLETED!")


# --------------------------------------------------
# BEST PARAMETERS
# --------------------------------------------------

print("\nBEST PARAMETERS")

print(
    grid_search.best_params_
)


print("\nBEST CROSS-VALIDATION ACCURACY")

print(
    grid_search.best_score_
)


print("\nBEST CROSS-VALIDATION ACCURACY (%)")

print(
    grid_search.best_score_ * 100
)


# --------------------------------------------------
# OPTIMIZED MODEL
# --------------------------------------------------

optimized_model = grid_search.best_estimator_

print("\nOPTIMIZED RANDOM FOREST MODEL")

print(
    optimized_model
)


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

print(
    X_train.shape
)

print("\nTESTING DATA SHAPE")

print(
    X_test.shape
)


# --------------------------------------------------
# TRAIN OPTIMIZED MODEL
# --------------------------------------------------

optimized_model.fit(

    X_train,

    y_train
)

print("\nOPTIMIZED MODEL TRAINING COMPLETED!")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = optimized_model.predict(
    X_test
)

print("\nOPTIMIZED MODEL PREDICTIONS")

print(
    y_pred[:20]
)


# --------------------------------------------------
# TEST ACCURACY
# --------------------------------------------------

test_accuracy = accuracy_score(

    y_test,

    y_pred
)

print("\nOPTIMIZED RANDOM FOREST TEST ACCURACY")

print(
    test_accuracy
)

print("\nOPTIMIZED RANDOM FOREST TEST ACCURACY (%)")

print(
    test_accuracy * 100
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print(
    "\nOPTIMIZED RANDOM FOREST CLASSIFICATION REPORT"
)

print(

    classification_report(

        y_test,

        y_pred,

        target_names=label_encoder.classes_

    )

)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred
)

print(
    "\nOPTIMIZED RANDOM FOREST CONFUSION MATRIX"
)

print(
    cm
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        optimized_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)

print(
    "\nOPTIMIZED RANDOM FOREST FEATURE IMPORTANCE"
)

print(
    feature_importance
)


# --------------------------------------------------
# BASE VS OPTIMIZED RANDOM FOREST
# --------------------------------------------------

try:

    day12_results = pd.read_csv(
        "data/model_comparison_day12.csv"
    )

    base_rf_accuracy = day12_results[
        day12_results["Model"] == "Random Forest"
    ]["Accuracy"].iloc[0]

    rf_comparison = pd.DataFrame({

        "Model": [
            "Base Random Forest",
            "Optimized Random Forest"
        ],

        "Accuracy": [
            base_rf_accuracy,
            test_accuracy
        ]

    })

    rf_comparison["Accuracy_Percentage"] = (

        rf_comparison["Accuracy"] * 100

    )

    print(
        "\nBASE VS OPTIMIZED RANDOM FOREST"
    )

    print(
        rf_comparison
    )

    improvement = (

        test_accuracy -
        base_rf_accuracy

    )

    print(
        "\nACCURACY IMPROVEMENT"
    )

    print(
        improvement
    )

    print(
        "\nACCURACY IMPROVEMENT (PERCENTAGE POINTS)"
    )

    print(
        improvement * 100
    )

    rf_comparison.to_csv(

        "data/base_vs_optimized_random_forest.csv",

        index=False
    )

    print(
        "\nBASE VS OPTIMIZED RESULTS SAVED SUCCESSFULLY!"
    )

except FileNotFoundError:

    print(
        "\nDAY 12 MODEL COMPARISON FILE NOT FOUND."
    )

    print(
        "SKIPPING BASE VS OPTIMIZED COMPARISON."
    )


# --------------------------------------------------
# SAVE BEST PARAMETERS
# --------------------------------------------------

best_parameters = pd.DataFrame(

    [grid_search.best_params_]

)

best_parameters.to_csv(

    "data/best_random_forest_parameters.csv",

    index=False
)

print(
    "\nBEST PARAMETERS SAVED SUCCESSFULLY!"
)


# --------------------------------------------------
# SAVE OPTIMIZED MODEL RESULTS
# --------------------------------------------------

optimized_results = pd.DataFrame({

    "Metric": [

        "Best_CV_Accuracy",

        "Test_Accuracy"

    ],

    "Value": [

        grid_search.best_score_,

        test_accuracy

    ]

})

optimized_results.to_csv(

    "data/optimized_random_forest_results.csv",

    index=False
)

print(
    "\nOPTIMIZED MODEL RESULTS SAVED SUCCESSFULLY!"
)


# --------------------------------------------------
# SAVE FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance.to_csv(

    "data/optimized_random_forest_feature_importance.csv",

    index=False
)

print(
    "\nFEATURE IMPORTANCE SAVED SUCCESSFULLY!"
)


# --------------------------------------------------
# SAVE OPTIMIZED MODEL
# --------------------------------------------------

joblib.dump(

    optimized_model,

    "data/optimized_random_forest_model.pkl"

)

print(
    "\nOPTIMIZED RANDOM FOREST MODEL SAVED SUCCESSFULLY!"
)


# --------------------------------------------------
# SAVE LABEL ENCODER
# --------------------------------------------------

joblib.dump(

    label_encoder,

    "data/optimized_random_forest_label_encoder.pkl"

)

print(
    "\nOPTIMIZED RANDOM FOREST LABEL ENCODER SAVED SUCCESSFULLY!"
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print(
    "\nDAY 14 RANDOM FOREST HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!"
)
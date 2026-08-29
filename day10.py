import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("SCIKIT-LEARN IMPORTED SUCCESSFULLY")

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

print("\nDATASET LOADED SUCCESSFULLY")

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

print("\nFEATURE DATA")
print(X.head())

print("\nTARGET DATA")
print(y.head())

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTARGET CLASSES")
print(label_encoder.classes_)

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

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

print("\nMODEL TRAINING COMPLETED!")

y_pred = model.predict(
    X_test
)

print("\nNUMERICAL PREDICTIONS")
print(y_pred[:20])

predicted_classes = label_encoder.inverse_transform(
    y_pred
)

actual_classes = label_encoder.inverse_transform(
    y_test
)

print("\nPREDICTED EFFICIENCY")
print(predicted_classes[:20])

print("\nACTUAL EFFICIENCY")
print(actual_classes[:20])

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMODEL ACCURACY")
print(accuracy)

print("\nMODEL ACCURACY (%)")
print(accuracy * 100)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(7, 5))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.yticks(
    range(len(label_encoder.classes_)),
    label_encoder.classes_
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()

plt.show()

coefficient_df = pd.DataFrame(
    model.coef_,
    columns=features,
    index=label_encoder.classes_
)

print("\nMODEL COEFFICIENTS")
print(coefficient_df)

model_results = pd.DataFrame({
    "Metric": [
        "Accuracy"
    ],
    "Value": [
        accuracy
    ]
})

model_results.to_csv(
    "data/logistic_regression_results.csv",
    index=False
)

print("\nMODEL RESULTS SAVED SUCCESSFULLY!")

joblib.dump(
    model,
    "data/logistic_regression_model.pkl"
)

joblib.dump(
    label_encoder,
    "data/efficiency_label_encoder.pkl"
)

print("\nTRAINED MODEL SAVED SUCCESSFULLY!")

print("\nDAY 10 LOGISTIC REGRESSION MODEL COMPLETED SUCCESSFULLY!")
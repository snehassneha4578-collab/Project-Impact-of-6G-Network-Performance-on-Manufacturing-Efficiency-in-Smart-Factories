import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
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
model = RandomForestClassifier(
n_estimators=100,
random_state=42
)
model.fit(
X_train,
y_train
)
print("\nRANDOM FOREST TRAINING COMPLETED!")
y_pred = model.predict(
X_test
)
print("\nRANDOM FOREST PREDICTIONS")
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
print("\nRANDOM FOREST ACCURACY")
print(accuracy)
print("\nRANDOM FOREST ACCURACY (%)")
print(accuracy * 100)
print("\nRANDOM FOREST CLASSIFICATION REPORT")
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
print("\nRANDOM FOREST CONFUSION MATRIX")
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
"Random Forest Confusion Matrix"
)
plt.tight_layout()
plt.show()
feature_importance = pd.DataFrame({
"Feature": features,
"Importance": model.feature_importances_
})
feature_importance = feature_importance.sort_values(
by="Importance",
ascending=False
)
print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(
feature_importance["Feature"],
feature_importance["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
"Random Forest Feature Importance"
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
logistic_results = pd.read_csv(
"data/logistic_regression_results.csv"
)
decision_tree_results = pd.read_csv(
"data/model_comparison_day11.csv"
)
logistic_accuracy = logistic_results[
"Value"
].iloc[0]
decision_tree_accuracy = decision_tree_results[
decision_tree_results["Model"] == "Decision Tree"
]["Accuracy"].iloc[0]
model_comparison = pd.DataFrame({
"Model": [
"Logistic Regression",
"Decision Tree",
"Random Forest"
],
"Accuracy": [
logistic_accuracy,
decision_tree_accuracy,
accuracy
]
})
print("\nTHREE-MODEL COMPARISON")
print(model_comparison)
model_comparison["Accuracy_Percentage"] = (
model_comparison["Accuracy"] * 100
)
print("\nMODEL ACCURACY COMPARISON (%)")
print(
model_comparison[
["Model", "Accuracy_Percentage"]
]
)
plt.figure(figsize=(9, 5))
plt.bar(
model_comparison["Model"],
model_comparison["Accuracy_Percentage"]
)
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title(
"Machine Learning Model Comparison"
)
plt.tight_layout()
plt.show()
best_model = model_comparison.loc[
model_comparison["Accuracy"].idxmax()
]
print("\nBEST MODEL BASED ON ACCURACY")
print(best_model)
model_comparison.to_csv(
"data/model_comparison_day12.csv",
index=False
)
print("\nMODEL COMPARISON SAVED SUCCESSFULLY!")
feature_importance.to_csv(
"data/random_forest_feature_importance.csv",
index=False
)
print("\nFEATURE IMPORTANCE SAVED SUCCESSFULLY!")
joblib.dump(
model,
"data/random_forest_model.pkl"
)
joblib.dump(
label_encoder,
"data/random_forest_label_encoder.pkl"
)
print("\nRANDOM FOREST MODEL SAVED SUCCESSFULLY!")

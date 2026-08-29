import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_smart_factory.csv")

print("CLEANED DATASET LOADED SUCCESSFULLY")

print("\nDATASET COLUMNS")
print(df.columns)

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

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())

print("\nFEATURE DATA TYPES")
print(X.dtypes)

print("\nTARGET VALUES")
print(y.value_counts())

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nENCODED TARGET")
print(y_encoded[:10])

print("\nTARGET CLASS MAPPING")

for index, class_name in enumerate(label_encoder.classes_):
    print(class_name, "=", index)

print("\nFEATURE DATASET SHAPE")
print(X.shape)

print("\nTARGET SHAPE")
print(y_encoded.shape)

print("\nMISSING VALUES IN FEATURES")
print(X.isnull().sum())

print("\nMISSING VALUES IN TARGET")
print(y.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

correlation = X.corr()

print("\nFEATURE CORRELATION")
print(correlation)

plt.figure(figsize=(10, 7))

plt.imshow(correlation, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTRAINING FEATURES")
print(X_train.shape)

print("\nTESTING FEATURES")
print(X_test.shape)

print("\nTRAINING TARGET")
print(y_train.shape)

print("\nTESTING TARGET")
print(y_test.shape)

print("\nTRAINING TARGET DISTRIBUTION")
print(pd.Series(y_train).value_counts())

print("\nTESTING TARGET DISTRIBUTION")
print(pd.Series(y_test).value_counts())

ml_dataset = X.copy()

ml_dataset["Efficiency_Status"] = y_encoded

ml_dataset.to_csv(
    "data/ml_ready_dataset.csv",
    index=False
)

print("\nML-READY DATASET SAVED SUCCESSFULLY!")

print("\nDAY 9 FEATURE ENGINEERING AND ML DATA PREPARATION COMPLETED SUCCESSFULLY!")
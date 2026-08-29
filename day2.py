"""
DAY 2 — PROJECT FOUNDATION + DATASET ANALYSIS

Project: Impact of 6G Network Performance on Manufacturing Efficiency
in Smart Factories: A Data-Driven Analysis
"""

import pandas as pd

print("=" * 70)
print("DAY 2 — PROJECT FOUNDATION")
print("=" * 70)

DATA_PATH = "data/smart_factory.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDATASET")
print("-" * 70)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nSTATISTICAL SUMMARY")
print("-" * 70)
print(df.describe())

print("\nMISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()
print(missing)
print("\nTotal missing values:", missing.sum())

print("\nDUPLICATE RECORDS")
print("-" * 70)

duplicates = df.duplicated().sum()
print("Duplicate records:", duplicates)

print("\nEFFICIENCY STATUS")
print("-" * 70)

if "Efficiency_Status" in df.columns:
    print(df["Efficiency_Status"].value_counts())

    print("\nEfficiency percentages:")
    print(
        df["Efficiency_Status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

print("\nNETWORK KPIs")
print("-" * 70)

network_features = [
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Error_Rate_%"
]

for feature in network_features:
    if feature in df.columns:
        print("\n", feature)
        print("Mean:", df[feature].mean())
        print("Minimum:", df[feature].min())
        print("Maximum:", df[feature].max())

print("\nMANUFACTURING KPIs")
print("-" * 70)

manufacturing_features = [
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Predictive_Maintenance_Score"
]

for feature in manufacturing_features:
    if feature in df.columns:
        print("\n", feature)
        print("Mean:", df[feature].mean())
        print("Minimum:", df[feature].min())
        print("Maximum:", df[feature].max())

print("\nDAY 2 SUMMARY")
print("-" * 70)
print("✓ Dataset statistical summary completed")
print("✓ Missing-value analysis completed")
print("✓ Duplicate check completed")
print("✓ Target distribution analyzed")
print("✓ Network KPIs identified")
print("✓ Manufacturing KPIs identified")
print("✓ Project foundation completed")

print("\nDAY 2 COMPLETE ✓")

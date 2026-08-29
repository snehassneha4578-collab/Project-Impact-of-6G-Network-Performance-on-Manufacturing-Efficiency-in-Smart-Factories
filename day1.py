"""
DAY 1 — PROJECT INTRODUCTION + DATASET UNDERSTANDING
Project: Impact of 6G Network Performance on Manufacturing Efficiency
in Smart Factories: A Data-Driven Analysis
"""

import pandas as pd

print("=" * 70)
print("DAY 1 — 6G SMART FACTORY NETWORK ANALYSIS")
print("=" * 70)

PROJECT_NAME = (
    "Impact of 6G Network Performance on Manufacturing "
    "Efficiency in Smart Factories: A Data-Driven Analysis"
)

print("\nPROJECT:")
print(PROJECT_NAME)

DATA_PATH = "data/smart_factory.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print("-" * 70)
print("Dataset path:", DATA_PATH)
print("Number of records:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFIRST 5 RECORDS")
print("-" * 70)
print(df.head())

print("\nCOLUMN NAMES")
print("-" * 70)

for column in df.columns:
    print(column)

print("\nDATA TYPES")
print("-" * 70)
print(df.dtypes)

print("\nTARGET VARIABLE")
print("-" * 70)

if "Efficiency_Status" in df.columns:
    print("Target variable: Efficiency_Status")
    print("\nTarget distribution:")
    print(df["Efficiency_Status"].value_counts())

print("\nDAY 1 SUMMARY")
print("-" * 70)
print("✓ Project introduced")
print("✓ Dataset loaded")
print("✓ Dataset dimensions inspected")
print("✓ Columns identified")
print("✓ Data types inspected")
print("✓ Target variable identified")

print("\nDAY 1 COMPLETE ✓")

import pandas as pd

df = pd.read_csv("data/smart_factory.csv")

print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nEFFICIENCY STATUS")
print(df["Efficiency_Status"].value_counts())

print("\nOPERATION MODE")
print(df["Operation_Mode"].value_counts())

print("\nNETWORK LATENCY")
print(df["Network_Latency_ms"].describe())

print("\nPACKET LOSS")
print(df["Packet_Loss_%"].describe())

print("\nPRODUCTION SPEED")
print(df["Production_Speed_units_per_hr"].describe())
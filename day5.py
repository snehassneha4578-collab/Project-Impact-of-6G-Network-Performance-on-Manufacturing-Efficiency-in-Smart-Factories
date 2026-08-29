import pandas as pd
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

df = pd.read_csv("data/cleaned_smart_factory.csv")

print("\nCLEANED DATASET LOADED SUCCESSFULLY")

print("\nDATASET SHAPE")
print(df.shape)

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nEFFICIENCY STATUS")
print(df["Efficiency_Status"].value_counts())

df["Efficiency_Status"].value_counts().plot(kind="bar")

plt.title("Efficiency Status Distribution")
plt.xlabel("Efficiency Status")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

print("\nNETWORK LATENCY")
print(df["Network_Latency_ms"].describe())

df["Network_Latency_ms"].plot(kind="hist", bins=20)

plt.title("Network Latency Distribution")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nPACKET LOSS")
print(df["Packet_Loss_%"].describe())

df["Packet_Loss_%"].plot(kind="hist", bins=20)

plt.title("Packet Loss Distribution")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nPRODUCTION SPEED")
print(df["Production_Speed_units_per_hr"].describe())

df["Production_Speed_units_per_hr"].plot(kind="hist", bins=20)

plt.title("Production Speed Distribution")
plt.xlabel("Production Speed (units/hr)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\nTEMPERATURE")
print(df["Temperature_C"].describe())

df["Temperature_C"].plot(kind="hist", bins=20)

plt.title("Machine Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

correlation = df.corr(numeric_only=True)

print("\nCORRELATION MATRIX")
print(correlation)

print("\nCORRELATION WITH PRODUCTION SPEED")
print(
    correlation["Production_Speed_units_per_hr"]
    .sort_values(ascending=False)
)

plt.scatter(
    df["Network_Latency_ms"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Network Latency vs Production Speed")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

plt.scatter(
    df["Packet_Loss_%"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Packet Loss vs Production Speed")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()
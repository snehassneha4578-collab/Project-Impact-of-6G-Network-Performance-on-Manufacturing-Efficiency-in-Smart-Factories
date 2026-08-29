import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_smart_factory.csv")

print("CLEANED DATASET LOADED SUCCESSFULLY")

# Network columns
print("\nNETWORK COLUMNS")
print(df[[
    "Network_Latency_ms",
    "Packet_Loss_%"
]])

# Network latency statistics
print("\nNETWORK LATENCY STATISTICS")
print(df["Network_Latency_ms"].describe())

# Packet loss statistics
print("\nPACKET LOSS STATISTICS")
print(df["Packet_Loss_%"].describe())

# Average network latency
average_latency = df["Network_Latency_ms"].mean()

print("\nAVERAGE NETWORK LATENCY")
print(average_latency)

# Average packet loss
average_packet_loss = df["Packet_Loss_%"].mean()

print("\nAVERAGE PACKET LOSS")
print(average_packet_loss)

# Minimum and maximum latency
print("\nMINIMUM LATENCY")
print(df["Network_Latency_ms"].min())

print("\nMAXIMUM LATENCY")
print(df["Network_Latency_ms"].max())

# Minimum and maximum packet loss
print("\nMINIMUM PACKET LOSS")
print(df["Packet_Loss_%"].min())

print("\nMAXIMUM PACKET LOSS")
print(df["Packet_Loss_%"].max())

# Average latency by efficiency
latency_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Network_Latency_ms"].mean()

print("\nAVERAGE LATENCY BY EFFICIENCY")
print(latency_by_efficiency)

# Average packet loss by efficiency
packet_loss_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Packet_Loss_%"].mean()

print("\nAVERAGE PACKET LOSS BY EFFICIENCY")
print(packet_loss_by_efficiency)

# Bar chart: latency by efficiency
latency_by_efficiency.plot(kind="bar")

plt.title("Average Network Latency by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Average Network Latency (ms)")
plt.tight_layout()
plt.show()

# Bar chart: packet loss by efficiency
packet_loss_by_efficiency.plot(kind="bar")

plt.title("Average Packet Loss by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Average Packet Loss (%)")
plt.tight_layout()
plt.show()

# Box plot: latency by efficiency
df.boxplot(
    column="Network_Latency_ms",
    by="Efficiency_Status"
)

plt.title("Network Latency by Efficiency Status")
plt.suptitle("")
plt.xlabel("Efficiency Status")
plt.ylabel("Network Latency (ms)")
plt.tight_layout()
plt.show()

# Box plot: packet loss by efficiency
df.boxplot(
    column="Packet_Loss_%",
    by="Efficiency_Status"
)

plt.title("Packet Loss by Efficiency Status")
plt.suptitle("")
plt.xlabel("Efficiency Status")
plt.ylabel("Packet Loss (%)")
plt.tight_layout()
plt.show()

# Network KPI summary
network_summary = pd.DataFrame({
    "KPI": [
        "Average Latency",
        "Average Packet Loss"
    ],
    "Value": [
        df["Network_Latency_ms"].mean(),
        df["Packet_Loss_%"].mean()
    ]
})

print("\nNETWORK KPI SUMMARY")
print(network_summary)

# Correlation with production speed
latency_correlation = df[
    "Network_Latency_ms"
].corr(
    df["Production_Speed_units_per_hr"]
)

packet_loss_correlation = df[
    "Packet_Loss_%"
].corr(
    df["Production_Speed_units_per_hr"]
)

print("\nLATENCY VS PRODUCTION SPEED CORRELATION")
print(latency_correlation)

print("\nPACKET LOSS VS PRODUCTION SPEED CORRELATION")
print(packet_loss_correlation)

# Scatter plot: latency vs production speed
plt.scatter(
    df["Network_Latency_ms"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Network Latency vs Production Speed")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

# Scatter plot: packet loss vs production speed
plt.scatter(
    df["Packet_Loss_%"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Packet Loss vs Production Speed")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

print("\nDAY 6 NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")


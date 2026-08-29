import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_smart_factory.csv")

print("CLEANED DATASET LOADED SUCCESSFULLY")

print("\nPROJECT ANALYSIS COLUMNS")

print(df[[
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Efficiency_Status"
]])

analysis_columns = [
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]

analysis_df = df[analysis_columns]

correlation_matrix = analysis_df.corr()

print("\nNETWORK + MANUFACTURING CORRELATION MATRIX")
print(correlation_matrix)

latency_production_corr = df["Network_Latency_ms"].corr(
    df["Production_Speed_units_per_hr"]
)

print("\nLATENCY VS PRODUCTION SPEED")
print(latency_production_corr)

plt.scatter(
    df["Network_Latency_ms"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Network Latency vs Production Speed")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

packet_production_corr = df["Packet_Loss_%"].corr(
    df["Production_Speed_units_per_hr"]
)

print("\nPACKET LOSS VS PRODUCTION SPEED")
print(packet_production_corr)

plt.scatter(
    df["Packet_Loss_%"],
    df["Production_Speed_units_per_hr"]
)

plt.title("Packet Loss vs Production Speed")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

latency_error_corr = df["Network_Latency_ms"].corr(
    df["Error_Rate_%"]
)

print("\nLATENCY VS ERROR RATE")
print(latency_error_corr)

plt.scatter(
    df["Network_Latency_ms"],
    df["Error_Rate_%"]
)

plt.title("Network Latency vs Error Rate")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Error Rate (%)")
plt.tight_layout()
plt.show()

packet_error_corr = df["Packet_Loss_%"].corr(
    df["Error_Rate_%"]
)

print("\nPACKET LOSS VS ERROR RATE")
print(packet_error_corr)

plt.scatter(
    df["Packet_Loss_%"],
    df["Error_Rate_%"]
)

plt.title("Packet Loss vs Error Rate")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Error Rate (%)")
plt.tight_layout()
plt.show()

latency_defect_corr = df["Network_Latency_ms"].corr(
    df["Quality_Control_Defect_Rate_%"]
)

print("\nLATENCY VS DEFECT RATE")
print(latency_defect_corr)

plt.scatter(
    df["Network_Latency_ms"],
    df["Quality_Control_Defect_Rate_%"]
)

plt.title("Network Latency vs Quality Defect Rate")
plt.xlabel("Network Latency (ms)")
plt.ylabel("Defect Rate (%)")
plt.tight_layout()
plt.show()

packet_defect_corr = df["Packet_Loss_%"].corr(
    df["Quality_Control_Defect_Rate_%"]
)

print("\nPACKET LOSS VS DEFECT RATE")
print(packet_defect_corr)

plt.scatter(
    df["Packet_Loss_%"],
    df["Quality_Control_Defect_Rate_%"]
)

plt.title("Packet Loss vs Quality Defect Rate")
plt.xlabel("Packet Loss (%)")
plt.ylabel("Defect Rate (%)")
plt.tight_layout()
plt.show()

latency_efficiency = df.groupby(
    "Efficiency_Status"
)["Network_Latency_ms"].mean()

print("\nAVERAGE LATENCY BY EFFICIENCY")
print(latency_efficiency)

packet_efficiency = df.groupby(
    "Efficiency_Status"
)["Packet_Loss_%"].mean()

print("\nAVERAGE PACKET LOSS BY EFFICIENCY")
print(packet_efficiency)

production_efficiency = df.groupby(
    "Efficiency_Status"
)["Production_Speed_units_per_hr"].mean()

defect_efficiency = df.groupby(
    "Efficiency_Status"
)["Quality_Control_Defect_Rate_%"].mean()

error_efficiency = df.groupby(
    "Efficiency_Status"
)["Error_Rate_%"].mean()

maintenance_efficiency = df.groupby(
    "Efficiency_Status"
)["Predictive_Maintenance_Score"].mean()

print("\nAVERAGE PRODUCTION SPEED BY EFFICIENCY")
print(production_efficiency)

print("\nAVERAGE DEFECT RATE BY EFFICIENCY")
print(defect_efficiency)

print("\nAVERAGE ERROR RATE BY EFFICIENCY")
print(error_efficiency)

print("\nAVERAGE MAINTENANCE SCORE BY EFFICIENCY")
print(maintenance_efficiency)

latency_efficiency.plot(kind="bar")

plt.title("Average Network Latency by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Average Latency (ms)")
plt.tight_layout()
plt.show()

packet_efficiency.plot(kind="bar")

plt.title("Average Packet Loss by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Average Packet Loss (%)")
plt.tight_layout()
plt.show()

production_efficiency.plot(kind="bar")

plt.title("Average Production Speed by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Production Speed (units/hr)")
plt.tight_layout()
plt.show()

error_efficiency.plot(kind="bar")

plt.title("Average Error Rate by Efficiency Status")
plt.xlabel("Efficiency Status")
plt.ylabel("Error Rate (%)")
plt.tight_layout()
plt.show()

correlation_summary = pd.DataFrame({
    "Relationship": [
        "Latency vs Production Speed",
        "Packet Loss vs Production Speed",
        "Latency vs Error Rate",
        "Packet Loss vs Error Rate",
        "Latency vs Defect Rate",
        "Packet Loss vs Defect Rate"
    ],
    "Correlation": [
        latency_production_corr,
        packet_production_corr,
        latency_error_corr,
        packet_error_corr,
        latency_defect_corr,
        packet_defect_corr
    ]
})

print("\nNETWORK-MANUFACTURING CORRELATION SUMMARY")
print(correlation_summary)

correlation_summary.to_csv(
    "data/network_manufacturing_correlations.csv",
    index=False
)

print("\nCORRELATION RESULTS SAVED SUCCESSFULLY!")

print("\nDAY 8 NETWORK + MANUFACTURING ANALYSIS COMPLETED SUCCESSFULLY!")
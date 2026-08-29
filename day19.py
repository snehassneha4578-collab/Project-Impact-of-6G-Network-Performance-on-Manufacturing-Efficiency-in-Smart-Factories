import pandas as pd
import matplotlib.pyplot as plt

print("LIBRARIES IMPORTED SUCCESSFULLY!")

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

print(
    "CLEANED DATASET LOADED SUCCESSFULLY!"
)

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET COLUMNS")
print(df.columns)

print("\nDATASET INFORMATION")
df.info()

efficiency_counts = (
    df["Efficiency_Status"]
    .value_counts()
)

print("\nEFFICIENCY DISTRIBUTION")
print(efficiency_counts)

efficiency_counts.plot(
    kind="bar"
)

plt.xlabel("Efficiency Status")
plt.ylabel("Number of Records")
plt.title("Smart Factory Efficiency Distribution")
plt.tight_layout()

plt.savefig(
    "data/efficiency_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

average_latency = (
    df["Network_Latency_ms"].mean()
)

print("\nAVERAGE NETWORK LATENCY")
print(average_latency)

average_packet_loss = (
    df["Packet_Loss_%"].mean()
)

print("\nAVERAGE PACKET LOSS")
print(average_packet_loss)

average_production_speed = (
    df["Production_Speed_units_per_hr"].mean()
)

print("\nAVERAGE PRODUCTION SPEED")
print(average_production_speed)

average_defect_rate = (
    df["Quality_Control_Defect_Rate_%"].mean()
)

print("\nAVERAGE DEFECT RATE")
print(average_defect_rate)

average_maintenance_score = (
    df["Predictive_Maintenance_Score"].mean()
)

print("\nAVERAGE PREDICTIVE MAINTENANCE SCORE")
print(average_maintenance_score)

average_error_rate = (
    df["Error_Rate_%"].mean()
)

print("\nAVERAGE ERROR RATE")
print(average_error_rate)

kpi_summary = pd.DataFrame({

    "KPI": [
        "Average Network Latency (ms)",
        "Average Packet Loss (%)",
        "Average Production Speed (units/hr)",
        "Average Defect Rate (%)",
        "Average Predictive Maintenance Score",
        "Average Error Rate (%)"
    ],

    "Value": [
        average_latency,
        average_packet_loss,
        average_production_speed,
        average_defect_rate,
        average_maintenance_score,
        average_error_rate
    ]
})

print("\nPROJECT KPI SUMMARY")
print(kpi_summary)

plt.figure(figsize=(10, 6))

plt.hist(
    df["Network_Latency_ms"],
    bins=20
)

plt.xlabel("Network Latency (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Network Latency")
plt.tight_layout()

plt.savefig(
    "data/latency_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.hist(
    df["Packet_Loss_%"],
    bins=20
)

plt.xlabel("Packet Loss (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Packet Loss")
plt.tight_layout()

plt.savefig(
    "data/packet_loss_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Network_Latency_ms"],
    df["Production_Speed_units_per_hr"]
)

plt.xlabel("Network Latency (ms)")
plt.ylabel("Production Speed (units/hr)")
plt.title("Network Latency vs Production Speed")
plt.tight_layout()

plt.savefig(
    "data/latency_vs_production_speed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Packet_Loss_%"],
    df["Production_Speed_units_per_hr"]
)

plt.xlabel("Packet Loss (%)")
plt.ylabel("Production Speed (units/hr)")
plt.title("Packet Loss vs Production Speed")
plt.tight_layout()

plt.savefig(
    "data/packet_loss_vs_production_speed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Network_Latency_ms"],
    df["Error_Rate_%"]
)

plt.xlabel("Network Latency (ms)")
plt.ylabel("Error Rate (%)")
plt.title("Network Latency vs Error Rate")
plt.tight_layout()

plt.savefig(
    "data/latency_vs_error_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Packet_Loss_%"],
    df["Error_Rate_%"]
)

plt.xlabel("Packet Loss (%)")
plt.ylabel("Error Rate (%)")
plt.title("Packet Loss vs Error Rate")
plt.tight_layout()

plt.savefig(
    "data/packet_loss_vs_error_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10, 6))

plt.hist(
    df["Production_Speed_units_per_hr"],
    bins=20
)

plt.xlabel("Production Speed (units/hr)")
plt.ylabel("Frequency")
plt.title("Production Speed Distribution")
plt.tight_layout()

plt.savefig(
    "data/production_speed_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

latency_efficiency = df.groupby(
    "Efficiency_Status"
)["Network_Latency_ms"].mean()

print("\nAVERAGE LATENCY BY EFFICIENCY")
print(latency_efficiency)

packet_loss_efficiency = df.groupby(
    "Efficiency_Status"
)["Packet_Loss_%"].mean()

print("\nAVERAGE PACKET LOSS BY EFFICIENCY")
print(packet_loss_efficiency)

production_efficiency = df.groupby(
    "Efficiency_Status"
)["Production_Speed_units_per_hr"].mean()

print("\nAVERAGE PRODUCTION SPEED BY EFFICIENCY")
print(production_efficiency)

efficiency_kpi = df.groupby(
    "Efficiency_Status"
)[
    [
        "Network_Latency_ms",
        "Packet_Loss_%",
        "Production_Speed_units_per_hr",
        "Quality_Control_Defect_Rate_%",
        "Predictive_Maintenance_Score",
        "Error_Rate_%"
    ]
].mean()

print("\nEFFICIENCY KPI COMPARISON")
print(efficiency_kpi)

efficiency_kpi[
    "Production_Speed_units_per_hr"
].plot(
    kind="bar"
)

plt.xlabel("Efficiency Status")
plt.ylabel("Production Speed (units/hr)")
plt.title("Production Speed by Efficiency Status")
plt.tight_layout()

plt.savefig(
    "data/production_speed_by_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

efficiency_kpi[
    "Quality_Control_Defect_Rate_%"
].plot(
    kind="bar"
)

plt.xlabel("Efficiency Status")
plt.ylabel("Defect Rate (%)")
plt.title("Defect Rate by Efficiency Status")
plt.tight_layout()

plt.savefig(
    "data/defect_rate_by_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

efficiency_kpi[
    "Error_Rate_%"
].plot(
    kind="bar"
)

plt.xlabel("Efficiency Status")
plt.ylabel("Error Rate (%)")
plt.title("Error Rate by Efficiency Status")
plt.tight_layout()

plt.savefig(
    "data/error_rate_by_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

network_kpi_summary = pd.DataFrame({

    "Metric": [
        "Average Latency (ms)",
        "Average Packet Loss (%)"
    ],

    "Value": [
        average_latency,
        average_packet_loss
    ]
})

print("\nNETWORK KPI SUMMARY")
print(network_kpi_summary)

manufacturing_kpi_summary = pd.DataFrame({

    "Metric": [
        "Average Production Speed (units/hr)",
        "Average Defect Rate (%)",
        "Average Predictive Maintenance Score",
        "Average Error Rate (%)"
    ],

    "Value": [
        average_production_speed,
        average_defect_rate,
        average_maintenance_score,
        average_error_rate
    ]
})

print("\nMANUFACTURING KPI SUMMARY")
print(manufacturing_kpi_summary)

dashboard_kpis = pd.DataFrame({

    "KPI": [
        "Average Network Latency",
        "Average Packet Loss",
        "Average Production Speed",
        "Average Defect Rate",
        "Average Predictive Maintenance Score",
        "Average Error Rate",
        "Total Records"
    ],

    "Value": [
        average_latency,
        average_packet_loss,
        average_production_speed,
        average_defect_rate,
        average_maintenance_score,
        average_error_rate,
        len(df)
    ]
})

print("\nDASHBOARD KPI DATA")
print(dashboard_kpis)

kpi_summary.to_csv(
    "data/project_kpi_summary.csv",
    index=False
)

dashboard_kpis.to_csv(
    "data/dashboard_kpis.csv",
    index=False
)

efficiency_kpi.to_csv(
    "data/efficiency_kpi_summary.csv"
)

print("\nKPI DATA SAVED SUCCESSFULLY!")

efficiency_counts.to_csv(
    "data/efficiency_distribution.csv"
)

latency_efficiency.to_csv(
    "data/latency_efficiency.csv"
)

packet_loss_efficiency.to_csv(
    "data/packet_loss_efficiency.csv"
)

production_efficiency.to_csv(
    "data/production_efficiency.csv"
)

print("\nEFFICIENCY ANALYSIS DATA SAVED SUCCESSFULLY!")

print("\nDAY 19 PROJECT KPI ANALYSIS COMPLETED SUCCESSFULLY!")
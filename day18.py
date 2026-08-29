import pandas as pd
import matplotlib.pyplot as plt

print("LIBRARIES IMPORTED SUCCESSFULLY!")

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

print(
    "CLEANED DATASET LOADED SUCCESSFULLY!"
)

important_columns = [
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Efficiency_Status"
]

print(
    "\nIMPORTANT PROJECT COLUMNS"
)

print(
    df[important_columns].head()
)

print(
    "\nNETWORK LATENCY STATISTICS"
)

print(
    df["Network_Latency_ms"].describe()
)

print(
    "\nPACKET LOSS STATISTICS"
)

print(
    df["Packet_Loss_%"].describe()
)

print(
    "\nEFFICIENCY STATUS DISTRIBUTION"
)

print(
    df["Efficiency_Status"].value_counts()
)

efficiency_network = df.groupby(
    "Efficiency_Status"
)[
    [
        "Network_Latency_ms",
        "Packet_Loss_%"
    ]
].mean()

print(
    "\nAVERAGE NETWORK PERFORMANCE BY EFFICIENCY"
)

print(
    efficiency_network
)

efficiency_manufacturing = df.groupby(
    "Efficiency_Status"
)[
    [
        "Production_Speed_units_per_hr",
        "Quality_Control_Defect_Rate_%",
        "Predictive_Maintenance_Score",
        "Error_Rate_%"
    ]
].mean()

print(
    "\nAVERAGE MANUFACTURING PERFORMANCE BY EFFICIENCY"
)

print(
    efficiency_manufacturing
)

efficiency_network = efficiency_network.reset_index()

print(
    "\nNETWORK PERFORMANCE TABLE"
)

print(
    efficiency_network
)

latency_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Network_Latency_ms"].mean()

print(
    "\nAVERAGE LATENCY BY EFFICIENCY"
)

print(
    latency_by_efficiency
)

packet_loss_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Packet_Loss_%"].mean()

print(
    "\nAVERAGE PACKET LOSS BY EFFICIENCY"
)

print(
    packet_loss_by_efficiency
)

latency_by_efficiency.plot(
    kind="bar"
)

plt.xlabel(
    "Efficiency Status"
)

plt.ylabel(
    "Average Network Latency (ms)"
)

plt.title(
    "Average Network Latency by Efficiency Status"
)

plt.tight_layout()

plt.savefig(
    "data/latency_vs_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nLATENCY VS EFFICIENCY GRAPH SAVED SUCCESSFULLY!"
)

packet_loss_by_efficiency.plot(
    kind="bar"
)

plt.xlabel(
    "Efficiency Status"
)

plt.ylabel(
    "Average Packet Loss (%)"
)

plt.title(
    "Average Packet Loss by Efficiency Status"
)

plt.tight_layout()

plt.savefig(
    "data/packet_loss_vs_efficiency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nPACKET LOSS VS EFFICIENCY GRAPH SAVED SUCCESSFULLY!"
)

numeric_columns = [
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

correlation_matrix = df[
    numeric_columns
].corr()

print(
    "\nCORRELATION MATRIX"
)

print(
    correlation_matrix
)

latency_production_corr = df[
    "Network_Latency_ms"
].corr(
    df[
        "Production_Speed_units_per_hr"
    ]
)

print(
    "\nLATENCY vs PRODUCTION SPEED CORRELATION"
)

print(
    latency_production_corr
)

packet_loss_production_corr = df[
    "Packet_Loss_%"
].corr(
    df[
        "Production_Speed_units_per_hr"
    ]
)

print(
    "\nPACKET LOSS vs PRODUCTION SPEED CORRELATION"
)

print(
    packet_loss_production_corr
)

latency_error_corr = df[
    "Network_Latency_ms"
].corr(
    df[
        "Error_Rate_%"
    ]
)

print(
    "\nLATENCY vs ERROR RATE CORRELATION"
)

print(
    latency_error_corr
)

packet_loss_error_corr = df[
    "Packet_Loss_%"
].corr(
    df[
        "Error_Rate_%"
    ]
)

print(
    "\nPACKET LOSS vs ERROR RATE CORRELATION"
)

print(
    packet_loss_error_corr
)

network_correlation_summary = pd.DataFrame({

    "Relationship": [
        "Latency vs Production Speed",
        "Packet Loss vs Production Speed",
        "Latency vs Error Rate",
        "Packet Loss vs Error Rate"
    ],

    "Correlation": [
        latency_production_corr,
        packet_loss_production_corr,
        latency_error_corr,
        packet_loss_error_corr
    ]
})

print(
    "\nNETWORK CORRELATION SUMMARY"
)

print(
    network_correlation_summary
)

low_efficiency_data = df[
    df["Efficiency_Status"] == "Low"
]

print(
    "\nLOW EFFICIENCY NETWORK CONDITIONS"
)

print(
    low_efficiency_data[
        [
            "Network_Latency_ms",
            "Packet_Loss_%"
        ]
    ].describe()
)

high_efficiency_data = df[
    df["Efficiency_Status"] == "High"
]

print(
    "\nHIGH EFFICIENCY NETWORK CONDITIONS"
)

print(
    high_efficiency_data[
        [
            "Network_Latency_ms",
            "Packet_Loss_%"
        ]
    ].describe()
)

high_low_comparison = pd.DataFrame({

    "Metric": [
        "Average Latency (ms)",
        "Average Packet Loss (%)"
    ],

    "High_Efficiency": [
        high_efficiency_data[
            "Network_Latency_ms"
        ].mean(),

        high_efficiency_data[
            "Packet_Loss_%"
        ].mean()
    ],

    "Low_Efficiency": [
        low_efficiency_data[
            "Network_Latency_ms"
        ].mean(),

        low_efficiency_data[
            "Packet_Loss_%"
        ].mean()
    ]
})

print(
    "\nHIGH VS LOW EFFICIENCY NETWORK COMPARISON"
)

print(
    high_low_comparison
)

latency_min = df[
    "Network_Latency_ms"
].min()

latency_max = df[
    "Network_Latency_ms"
].max()

df["Latency_Normalized"] = (
    (
        df["Network_Latency_ms"]
        - latency_min
    )
    /
    (
        latency_max
        - latency_min
    )
)

df["Latency_Performance"] = (
    1 - df["Latency_Normalized"]
)

packet_loss_min = df[
    "Packet_Loss_%"
].min()

packet_loss_max = df[
    "Packet_Loss_%"
].max()

df["Packet_Loss_Normalized"] = (
    (
        df["Packet_Loss_%"]
        - packet_loss_min
    )
    /
    (
        packet_loss_max
        - packet_loss_min
    )
)

df["Packet_Loss_Performance"] = (
    1 - df["Packet_Loss_Normalized"]
)

df["Network_Performance_Index"] = (
    (
        df["Latency_Performance"]
        +
        df["Packet_Loss_Performance"]
    )
    / 2
)

df["Network_Performance_Index_%"] = (
    df["Network_Performance_Index"]
    * 100
)

print(
    "\nNETWORK PERFORMANCE INDEX"
)

print(
    df[
        [
            "Network_Latency_ms",
            "Packet_Loss_%",
            "Network_Performance_Index_%"
        ]
    ].head()
)

network_index_by_efficiency = df.groupby(
    "Efficiency_Status"
)[
    "Network_Performance_Index_%"
].mean()

print(
    "\nAVERAGE NETWORK PERFORMANCE INDEX BY EFFICIENCY"
)

print(
    network_index_by_efficiency
)

network_index_by_efficiency.plot(
    kind="bar"
)

plt.xlabel(
    "Efficiency Status"
)

plt.ylabel(
    "Network Performance Index (%)"
)

plt.title(
    "Network Performance Index by Efficiency Status"
)

plt.tight_layout()

plt.savefig(
    "data/network_performance_index.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nNETWORK PERFORMANCE INDEX GRAPH SAVED SUCCESSFULLY!"
)

network_impact_summary = pd.DataFrame({

    "Efficiency_Status": (
        network_index_by_efficiency.index
    ),

    "Average_Latency_ms": (
        df.groupby(
            "Efficiency_Status"
        )[
            "Network_Latency_ms"
        ].mean().values
    ),

    "Average_Packet_Loss_%": (
        df.groupby(
            "Efficiency_Status"
        )[
            "Packet_Loss_%"
        ].mean().values
    ),

    "Average_Network_Performance_Index_%": (
        network_index_by_efficiency.values
    )
})

print(
    "\nFINAL NETWORK IMPACT SUMMARY"
)

print(
    network_impact_summary
)

network_impact_summary.to_csv(
    "data/network_impact_summary.csv",
    index=False
)

print(
    "\nNETWORK IMPACT SUMMARY SAVED SUCCESSFULLY!"
)

correlation_matrix.to_csv(
    "data/network_correlation_matrix.csv"
)

network_correlation_summary.to_csv(
    "data/network_correlation_summary.csv",
    index=False
)

print(
    "\nCORRELATION RESULTS SAVED SUCCESSFULLY!"
)

df.to_csv(
    "data/network_impact_analysis.csv",
    index=False
)

print(
    "\nNETWORK IMPACT ANALYSIS DATA SAVED SUCCESSFULLY!"
)

print(
    "\nDAY 18 NETWORK IMPACT ANALYSIS COMPLETED SUCCESSFULLY!"
)
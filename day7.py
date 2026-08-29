import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_smart_factory.csv")

print("CLEANED DATASET LOADED SUCCESSFULLY")

print("\nMANUFACTURING COLUMNS")

print(df[[
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Efficiency_Status"
]])

print("\nPRODUCTION SPEED STATISTICS")

print(df["Production_Speed_units_per_hr"].describe())

average_production_speed = df[
    "Production_Speed_units_per_hr"
].mean()

print("\nAVERAGE PRODUCTION SPEED")

print(average_production_speed)

print("\nQUALITY CONTROL DEFECT RATE")

print(df["Quality_Control_Defect_Rate_%"].describe())

average_defect_rate = df[
    "Quality_Control_Defect_Rate_%"
].mean()

print("\nAVERAGE DEFECT RATE")

print(average_defect_rate)

print("\nPREDICTIVE MAINTENANCE SCORE")

print(df["Predictive_Maintenance_Score"].describe())

average_maintenance_score = df[
    "Predictive_Maintenance_Score"
].mean()

print("\nAVERAGE PREDICTIVE MAINTENANCE SCORE")

print(average_maintenance_score)

print("\nERROR RATE")

print(df["Error_Rate_%"].describe())

average_error_rate = df[
    "Error_Rate_%"
].mean()

print("\nAVERAGE ERROR RATE")

print(average_error_rate)

efficiency_counts = df[
    "Efficiency_Status"
].value_counts()

print("\nEFFICIENCY STATUS DISTRIBUTION")

print(efficiency_counts)

production_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Production_Speed_units_per_hr"].mean()

print("\nAVERAGE PRODUCTION SPEED BY EFFICIENCY")

print(production_by_efficiency)

defect_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Quality_Control_Defect_Rate_%"].mean()

print("\nAVERAGE DEFECT RATE BY EFFICIENCY")

print(defect_by_efficiency)

error_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Error_Rate_%"].mean()

print("\nAVERAGE ERROR RATE BY EFFICIENCY")

print(error_by_efficiency)

maintenance_by_efficiency = df.groupby(
    "Efficiency_Status"
)["Predictive_Maintenance_Score"].mean()

print("\nAVERAGE MAINTENANCE SCORE BY EFFICIENCY")

print(maintenance_by_efficiency)

production_by_efficiency.plot(kind="bar")

plt.title("Average Production Speed by Efficiency Status")

plt.xlabel("Efficiency Status")

plt.ylabel("Production Speed (units/hr)")

plt.tight_layout()

plt.show()

defect_by_efficiency.plot(kind="bar")

plt.title("Average Defect Rate by Efficiency Status")

plt.xlabel("Efficiency Status")

plt.ylabel("Defect Rate (%)")

plt.tight_layout()

plt.show()

error_by_efficiency.plot(kind="bar")

plt.title("Average Error Rate by Efficiency Status")

plt.xlabel("Efficiency Status")

plt.ylabel("Error Rate (%)")

plt.tight_layout()

plt.show()

maintenance_by_efficiency.plot(kind="bar")

plt.title("Average Predictive Maintenance Score by Efficiency Status")

plt.xlabel("Efficiency Status")

plt.ylabel("Predictive Maintenance Score")

plt.tight_layout()

plt.show()

plt.scatter(
    df["Production_Speed_units_per_hr"],
    df["Error_Rate_%"]
)

plt.title("Production Speed vs Error Rate")

plt.xlabel("Production Speed (units/hr)")

plt.ylabel("Error Rate (%)")

plt.tight_layout()

plt.show()

plt.scatter(
    df["Production_Speed_units_per_hr"],
    df["Quality_Control_Defect_Rate_%"]
)

plt.title("Production Speed vs Defect Rate")

plt.xlabel("Production Speed (units/hr)")

plt.ylabel("Defect Rate (%)")

plt.tight_layout()

plt.show()

manufacturing_correlation = df[[
    "Production_Speed_units_per_hr",
    "Quality_Control_Defect_Rate_%",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]].corr()

print("\nMANUFACTURING CORRELATION MATRIX")

print(manufacturing_correlation)

print("\nCORRELATION WITH PRODUCTION SPEED")

print(
    manufacturing_correlation[
        "Production_Speed_units_per_hr"
    ].sort_values(ascending=False)
)

manufacturing_summary = pd.DataFrame({
    "KPI": [
        "Average Production Speed",
        "Average Defect Rate",
        "Average Maintenance Score",
        "Average Error Rate"
    ],
    "Value": [
        average_production_speed,
        average_defect_rate,
        average_maintenance_score,
        average_error_rate
    ]
})

print("\nMANUFACTURING KPI SUMMARY")

print(manufacturing_summary)

print("\nDAY 7 MANUFACTURING ANALYSIS COMPLETED SUCCESSFULLY!")
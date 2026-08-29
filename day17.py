import pandas as pd
import matplotlib.pyplot as plt
import joblib

print("LIBRARIES IMPORTED SUCCESSFULLY!")

model = joblib.load(
    "data/optimized_random_forest_model.pkl"
)

print(
    "OPTIMIZED RANDOM FOREST MODEL LOADED SUCCESSFULLY!"
)

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

print("\nNUMBER OF FEATURES")
print(len(features))

importance_values = model.feature_importances_

print("\nRAW FEATURE IMPORTANCE")
print(importance_values)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance_values
})

print("\nFEATURE IMPORTANCE")
print(feature_importance)

feature_importance["Importance_Percentage"] = (
    feature_importance["Importance"] * 100
)

print("\nFEATURE IMPORTANCE (%)")
print(
    feature_importance[
        [
            "Feature",
            "Importance_Percentage"
        ]
    ]
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance = feature_importance.reset_index(
    drop=True
)

feature_importance["Rank"] = (
    feature_importance.index + 1
)

feature_importance = feature_importance[
    [
        "Rank",
        "Feature",
        "Importance",
        "Importance_Percentage"
    ]
]

print("\nFINAL FEATURE RANKING")
print(feature_importance)

most_important_feature = feature_importance.iloc[0]

print("\nMOST IMPORTANT FEATURE")
print(
    most_important_feature["Feature"]
)

print(
    "Importance:",
    most_important_feature[
        "Importance_Percentage"
    ]
)

least_important_feature = feature_importance.iloc[-1]

print("\nLEAST IMPORTANT FEATURE")
print(
    least_important_feature["Feature"]
)

print(
    "Importance:",
    least_important_feature[
        "Importance_Percentage"
    ]
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "data/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFEATURE IMPORTANCE GRAPH SAVED SUCCESSFULLY!"
)

network_features = [
    "Network_Latency_ms",
    "Packet_Loss_%"
]

network_importance = feature_importance[
    feature_importance["Feature"].isin(
        network_features
    )
]

print(
    "\nNETWORK FEATURE IMPORTANCE"
)

print(
    network_importance
)

total_network_importance = (
    network_importance["Importance"].sum()
)

print(
    "\nTOTAL NETWORK FEATURE IMPORTANCE"
)

print(
    total_network_importance
)

print(
    "\nTOTAL NETWORK IMPORTANCE (%)"
)

print(
    total_network_importance * 100
)

manufacturing_features = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
]

manufacturing_importance = feature_importance[
    feature_importance["Feature"].isin(
        manufacturing_features
    )
]

print(
    "\nMANUFACTURING FEATURE IMPORTANCE"
)

print(
    manufacturing_importance
)

total_manufacturing_importance = (
    manufacturing_importance["Importance"].sum()
)

print(
    "\nTOTAL MANUFACTURING IMPORTANCE"
)

print(
    total_manufacturing_importance
)

print(
    "\nTOTAL MANUFACTURING IMPORTANCE (%)"
)

print(
    total_manufacturing_importance * 100
)

importance_comparison = pd.DataFrame({

    "Category": [
        "Network Features",
        "Manufacturing Features"
    ],

    "Total_Importance": [
        total_network_importance,
        total_manufacturing_importance
    ]

})

importance_comparison[
    "Importance_Percentage"
] = (
    importance_comparison[
        "Total_Importance"
    ] * 100
)

print(
    "\nNETWORK VS MANUFACTURING IMPORTANCE"
)

print(
    importance_comparison
)

plt.figure(figsize=(8, 5))

plt.bar(
    importance_comparison["Category"],
    importance_comparison[
        "Importance_Percentage"
    ]
)

plt.xlabel(
    "Feature Category"
)

plt.ylabel(
    "Total Importance (%)"
)

plt.title(
    "Network vs Manufacturing Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "data/network_vs_manufacturing_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nCATEGORY IMPORTANCE GRAPH SAVED SUCCESSFULLY!"
)

top_5_features = feature_importance.head(5)

print(
    "\nTOP 5 MOST IMPORTANT FEATURES"
)

print(
    top_5_features
)

top_network_feature = network_importance.sort_values(
    by="Importance",
    ascending=False
).iloc[0]

print(
    "\nMOST IMPORTANT NETWORK FEATURE"
)

print(
    top_network_feature["Feature"]
)

print(
    "Importance:",
    top_network_feature[
        "Importance_Percentage"
    ]
)

print(
    "\nMODEL INTERPRETATION"
)

print(
    "The Random Forest model ranks the project "
    "features according to their contribution "
    "to prediction."
)

print(
    "Network-related features such as latency "
    "and packet loss are analyzed to understand "
    "their contribution to manufacturing efficiency."
)

feature_importance.to_csv(
    "data/feature_importance_results.csv",
    index=False
)

print(
    "\nFEATURE IMPORTANCE RESULTS SAVED SUCCESSFULLY!"
)

network_importance.to_csv(
    "data/network_feature_importance.csv",
    index=False
)

print(
    "\nNETWORK FEATURE IMPORTANCE SAVED SUCCESSFULLY!"
)

importance_comparison.to_csv(
    "data/network_vs_manufacturing_importance.csv",
    index=False
)

print(
    "\nCATEGORY IMPORTANCE RESULTS SAVED SUCCESSFULLY!"
)

print(
    "\nDAY 17 MODEL INTERPRETATION AND FEATURE IMPORTANCE ANALYSIS COMPLETED SUCCESSFULLY!"
)
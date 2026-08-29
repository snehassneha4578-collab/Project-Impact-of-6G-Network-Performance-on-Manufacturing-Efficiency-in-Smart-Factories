import pandas as pd
import joblib

print("LIBRARIES IMPORTED SUCCESSFULLY!")

# --------------------------------------------------
# LOAD OPTIMIZED RANDOM FOREST MODEL
# --------------------------------------------------

model = joblib.load(
    "data/optimized_random_forest_model.pkl"
)

print(
    "OPTIMIZED RANDOM FOREST MODEL LOADED SUCCESSFULLY!"
)

# --------------------------------------------------
# LOAD LABEL ENCODER
# --------------------------------------------------

label_encoder = joblib.load(
    "data/optimized_random_forest_label_encoder.pkl"
)

print(
    "LABEL ENCODER LOADED SUCCESSFULLY!"
)

print("\nTARGET CLASSES")

print(label_encoder.classes_)

# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

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

print("\nFEATURES USED BY THE MODEL")

for feature in features:
    print(feature)

# --------------------------------------------------
# FIRST FACTORY CONDITION
# --------------------------------------------------

new_data = pd.DataFrame({
    "Network_Latency_ms": [20],
    "Packet_Loss_%": [1.5],
    "Temperature_C": [70],
    "Vibration_Hz": [35],
    "Power_Consumption_kW": [50],
    "Quality_Control_Defect_Rate_%": [2],
    "Production_Speed_units_per_hr": [100],
    "Predictive_Maintenance_Score": [85],
    "Error_Rate_%": [1]
})

print("\nNEW FACTORY DATA")

print(new_data)

new_data = new_data[features]

print("\nFEATURE ORDER VERIFIED")

print(new_data.columns)

# --------------------------------------------------
# FIRST PREDICTION
# --------------------------------------------------

prediction_encoded = model.predict(
    new_data
)

prediction = label_encoder.inverse_transform(
    prediction_encoded
)

print("\nPREDICTED EFFICIENCY STATUS")

print(prediction[0])

# --------------------------------------------------
# PREDICTION PROBABILITIES
# --------------------------------------------------

probabilities = model.predict_proba(
    new_data
)

print("\nPREDICTION PROBABILITIES")

print(probabilities)

probability_table = pd.DataFrame(
    probabilities,
    columns=label_encoder.classes_
)

print("\nEFFICIENCY PROBABILITY TABLE")

print(probability_table)

# --------------------------------------------------
# MOST PROBABLE CLASS
# --------------------------------------------------

predicted_class_index = probabilities.argmax(
    axis=1
)

predicted_class = label_encoder.inverse_transform(
    predicted_class_index
)

print("\nMOST PROBABLE EFFICIENCY STATUS")

print(predicted_class[0])

# --------------------------------------------------
# CONFIDENCE
# --------------------------------------------------

confidence = probabilities.max(
    axis=1
)

print("\nMODEL PREDICTION CONFIDENCE")

print(confidence[0])

print("\nMODEL CONFIDENCE (%)")

print(
    confidence[0] * 100
)

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_efficiency(input_data):

    input_df = pd.DataFrame(
        [input_data]
    )

    input_df = input_df[features]

    prediction_encoded = model.predict(
        input_df
    )

    prediction = label_encoder.inverse_transform(
        prediction_encoded
    )

    probabilities = model.predict_proba(
        input_df
    )

    confidence = probabilities.max(
        axis=1
    )

    return prediction[0], confidence[0]


# --------------------------------------------------
# FACTORY CONDITION 1
# --------------------------------------------------

factory_input = {

    "Network_Latency_ms": 20,
    "Packet_Loss_%": 1.5,
    "Temperature_C": 70,
    "Vibration_Hz": 35,
    "Power_Consumption_kW": 50,
    "Quality_Control_Defect_Rate_%": 2,
    "Production_Speed_units_per_hr": 100,
    "Predictive_Maintenance_Score": 85,
    "Error_Rate_%": 1
}

status, confidence = predict_efficiency(
    factory_input
)

print("\nFINAL FACTORY PREDICTION")

print(
    "Efficiency Status:",
    status
)

print(
    "Confidence:",
    confidence * 100
)

# --------------------------------------------------
# FACTORY CONDITION 2
# --------------------------------------------------

factory_input_2 = {

    "Network_Latency_ms": 80,
    "Packet_Loss_%": 8,
    "Temperature_C": 90,
    "Vibration_Hz": 70,
    "Power_Consumption_kW": 75,
    "Quality_Control_Defect_Rate_%": 8,
    "Production_Speed_units_per_hr": 60,
    "Predictive_Maintenance_Score": 45,
    "Error_Rate_%": 7
}

status_2, confidence_2 = predict_efficiency(
    factory_input_2
)

print("\nSECOND FACTORY PREDICTION")

print(
    "Efficiency Status:",
    status_2
)

print(
    "Confidence:",
    confidence_2 * 100
)

# --------------------------------------------------
# CONDITION COMPARISON
# --------------------------------------------------

prediction_comparison = pd.DataFrame({

    "Factory_Condition": [
        "Condition 1",
        "Condition 2"
    ],

    "Efficiency_Status": [
        status,
        status_2
    ],

    "Confidence": [
        confidence * 100,
        confidence_2 * 100
    ]

})

print("\nFACTORY CONDITION COMPARISON")

print(prediction_comparison)

# --------------------------------------------------
# BATCH PREDICTION FUNCTION
# --------------------------------------------------

def predict_batch(input_dataframe):

    input_dataframe = input_dataframe[
        features
    ]

    predictions_encoded = model.predict(
        input_dataframe
    )

    predictions = label_encoder.inverse_transform(
        predictions_encoded
    )

    probabilities = model.predict_proba(
        input_dataframe
    )

    confidence = probabilities.max(
        axis=1
    )

    result = input_dataframe.copy()

    result["Predicted_Efficiency_Status"] = (
        predictions
    )

    result["Prediction_Confidence"] = (
        confidence * 100
    )

    return result


# --------------------------------------------------
# BATCH FACTORY DATA
# --------------------------------------------------

batch_data = pd.DataFrame({

    "Network_Latency_ms": [
        20,
        40,
        80
    ],

    "Packet_Loss_%": [
        1.5,
        4,
        8
    ],

    "Temperature_C": [
        70,
        78,
        90
    ],

    "Vibration_Hz": [
        35,
        50,
        70
    ],

    "Power_Consumption_kW": [
        50,
        60,
        75
    ],

    "Quality_Control_Defect_Rate_%": [
        2,
        5,
        8
    ],

    "Production_Speed_units_per_hr": [
        100,
        85,
        60
    ],

    "Predictive_Maintenance_Score": [
        85,
        65,
        45
    ],

    "Error_Rate_%": [
        1,
        4,
        7
    ]

})

# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

batch_results = predict_batch(
    batch_data
)

print("\nBATCH PREDICTION RESULTS")

print(batch_results)

# --------------------------------------------------
# SAVE BATCH RESULTS
# --------------------------------------------------

batch_results.to_csv(
    "data/day16_prediction_results.csv",
    index=False
)

print(
    "\nPREDICTION RESULTS SAVED SUCCESSFULLY!"
)

# --------------------------------------------------
# SAVE MODEL FEATURE LIST
# --------------------------------------------------

feature_information = pd.DataFrame({

    "Feature": features

})

feature_information.to_csv(
    "data/model_features.csv",
    index=False
)

print(
    "\nMODEL FEATURE LIST SAVED SUCCESSFULLY!"
)

# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print(
    "\nDAY 16 MODEL PREDICTION AND FACTORY "
    "EFFICIENCY INFERENCE COMPLETED SUCCESSFULLY!"
)
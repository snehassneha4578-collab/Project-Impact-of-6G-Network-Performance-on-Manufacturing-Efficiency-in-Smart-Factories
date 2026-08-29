import pandas as pd

df = pd.read_csv("data/smart_factory.csv")

print("DATASET LOADED SUCCESSFULLY")

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES BY COLUMN")
print(df.isnull().sum())

print("\nTOTAL MISSING VALUES")
print(df.isnull().sum().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDUPLICATES AFTER CLEANING")
print(df.duplicated().sum())

print("\nDATA TYPES")
print(df.dtypes)

print("\nNEGATIVE TEMPERATURE VALUES")
print((df["Temperature_C"] < 0).sum())

print("\nNEGATIVE VIBRATION VALUES")
print((df["Vibration_Hz"] < 0).sum())

print("\nNEGATIVE POWER CONSUMPTION VALUES")
print((df["Power_Consumption_kW"] < 0).sum())

print("\nNEGATIVE NETWORK LATENCY VALUES")
print((df["Network_Latency_ms"] < 0).sum())

print("\nNEGATIVE PACKET LOSS VALUES")
print((df["Packet_Loss_%"] < 0).sum())

print("\nNEGATIVE PRODUCTION SPEED VALUES")
print((df["Production_Speed_units_per_hr"] < 0).sum())

print("\nEFFICIENCY STATUS")
print(df["Efficiency_Status"].value_counts())

df.to_csv("data/cleaned_smart_factory.csv", index=False)

print("\nFINAL DATASET SHAPE")
print(df.shape)

print("\nFINAL MISSING VALUES")
print(df.isnull().sum().sum())

print("\nFINAL DUPLICATES")
print(df.duplicated().sum())

print("\nCLEAN DATASET SAVED SUCCESSFULLY!")
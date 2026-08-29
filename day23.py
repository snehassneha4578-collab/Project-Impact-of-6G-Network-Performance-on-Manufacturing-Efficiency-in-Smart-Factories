import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="6G Smart Factory - Day 23",
    page_icon="📡",
    layout="wide"
)

st.title(
    "📡 6G Smart Factory Network Analysis"
)

df = pd.read_csv(
    "data/cleaned_smart_factory.csv"
)

# Use the complete dataset
filtered_df = df.copy()

st.success(
    "Cleaned Smart Factory dataset loaded successfully!"
)

print(
    "DAY 23 DATASET LOADED SUCCESSFULLY!"
)
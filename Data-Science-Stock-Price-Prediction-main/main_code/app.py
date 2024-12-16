import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from modelrunning import fetch_stock_data, process_stock_data, calculate_predictions

# Define the list of stock symbols to choose from
stock_symbols = ["AMZN", "AAPL", "NVDA", "MSFT", "GOOG", "META", "TSLA", "WMT", "JPM", "NFLX"]

# Sidebar: Stock selection
selected_stock = st.sidebar.selectbox("Select a Stock:", stock_symbols)

# Sidebar: Select number of days for trend visualization (5, 10, or 20)
days_option = st.sidebar.selectbox("Select number of days for trend visualization:", [5, 10, 20])

# Fetch stock data
st.write(f"Fetching data for {selected_stock}...")
stock_data = fetch_stock_data(selected_stock)

# Process the stock data
st.write(f"Processing data for {selected_stock}...")
processed_data = process_stock_data(stock_data)

# Display the processed data
st.write(f"Showing the last {days_option} rows of processed data for {selected_stock}:")
st.dataframe(processed_data.tail(days_option))

# Calculate the stock prediction
prediction = calculate_predictions(processed_data)

# Display the prediction result
if prediction == 1:
    st.write(f"The predicted trend for {selected_stock} is UP (1) for the next day.")
else:
    st.write(f"The predicted trend for {selected_stock} is DOWN (0) for the next day.")

# Trend visualization for selected number of days
trend_data = processed_data['trend'].tail(days_option)

# Plot the trend data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trend_data.index, trend_data.values, marker='o', linestyle='-', color='b', label='Trend')

# Add labels and title
ax.set_title(f"Trend Visualization for the Last {days_option} Days ({selected_stock})", fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Trend (1=UP, 0=DOWN)', fontsize=12)
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

# Background Image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('./images.jpg');
        background-size: cover;
        background-position: center;
        opacity: 0.5;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from modelrunning import fetch_stock_data, process_stock_data, calculate_predictions

# Define the list of stock symbols to choose from
stock_symbols = ["AMZN", "AAPL", "NVDA", "MSFT", "GOOG", "META", "TSLA", "WMT", "JPM", "NFLX"]

# Sidebar: Stock selection
selected_stock = st.sidebar.selectbox("Select a Stock:", stock_symbols)

# Fetch stock data
st.write(f"Fetching data for {selected_stock}...")
stock_data = fetch_stock_data(selected_stock)

# Process the stock data
st.write(f"Processing data for {selected_stock}...")
processed_data = process_stock_data(stock_data)

# Display the processed data
st.write(f"Showing the last few rows of processed data for {selected_stock}:")
st.dataframe(processed_data.tail())

# Calculate the stock prediction
prediction = calculate_predictions(processed_data)

# Display the prediction result
if prediction == 1:
    st.write(f"The predicted trend for {selected_stock} is UP (1) for the next day.")
else:
    st.write(f"The predicted trend for {selected_stock} is DOWN (0) for the next day.")

import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from modelrunning import fetch_stock_data, process_stock_data, calculate_predictions

# Define the list of stock symbols to choose from
stock_symbols = ["AMZN", "AAPL", "NVDA", "MSFT", "GOOG", "META", "TSLA", "WMT", "JPM", "NFLX"]

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)

# Display the app title and an image
st.title("Stock Trend Prediction App")
image_path = str(Path(__file__).parent / 'images.webp')
st.image(image_path, caption="Stock Prediction Dashboard", use_container_width=True)

# Sidebar: Stock selection
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.selectbox("Select a Stock:", stock_symbols)

# Sidebar: Time frame selection
time_frames = [5, 10, 20]
selected_time_frame = st.sidebar.selectbox("Select Time Frame (days):", time_frames)

# Fetch stock data
with st.spinner(f"Fetching data for {selected_stock}..."):
    stock_data = fetch_stock_data(selected_stock)

# Process stock data
with st.spinner(f"Processing data for {selected_stock}..."):
    processed_data = process_stock_data(stock_data)

# Calculate the stock predictions
with st.spinner(f"Calculating predictions for {selected_stock}..."):
    predictions, y_test = calculate_predictions(processed_data)

# Generate a date range for the last `selected_time_frame` days (relative to yesterday)
end_date = datetime.now() - timedelta(days=1)
date_range = [end_date - timedelta(days=i) for i in range(selected_time_frame)][::-1]

# Align predictions with the corresponding dates
prediction_data = pd.DataFrame({
    'Date': date_range,
    'Prediction': predictions[-selected_time_frame:]
})

# Display the prediction result for the next day
st.subheader("Prediction Result")
next_day_prediction = predictions[-1]

if next_day_prediction == 1:
    st.success(f"The predicted trend for {selected_stock} is **UP** (1) for the next day.")
else:
    st.error(f"The predicted trend for {selected_stock} is **DOWN** (0) for the next day.")

# Display Prediction Trends Line Chart
st.write(f"### Stock Trend Over the Last {selected_time_frame} Days:")

# Prediction Trend Chart
prediction_chart = alt.Chart(prediction_data).mark_line(size=2).encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Prediction:Q', title='Prediction Trend', scale=alt.Scale(domain=[0, 1])),
    color=alt.value('blue')
).properties(
    width=600,
    height=400
)
st.altair_chart(prediction_chart, use_container_width=True)
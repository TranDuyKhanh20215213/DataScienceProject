import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta
from modelrunning import fetch_stock_data, process_stock_data, calculate_predictions

# Define the list of stock symbols to choose from
stock_symbols = ["AMZN", "AAPL", "NVDA", "MSFT", "GOOG", "META", "TSLA", "WMT", "JPM", "NFLX"]

# Map named time frames to the corresponding number of days
time_frame_options = {
    "2 Weeks": 14,
    "1 Month": 30,
    "2 Months": 60
}

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
st.sidebar.header("Time Frame Selection")
selected_time_frame_label = st.sidebar.selectbox("Select Time Frame:", list(time_frame_options.keys()))
selected_time_frame_days = time_frame_options[selected_time_frame_label]

# Fetch stock data
with st.spinner(f"Fetching data for {selected_stock}..."):
    stock_data = fetch_stock_data(selected_stock)

# Process stock data
with st.spinner(f"Processing data for {selected_stock}..."):
    processed_data = process_stock_data(stock_data)

# Calculate the stock predictions
with st.spinner(f"Calculating predictions for {selected_stock}..."):
    predictions, y_test = calculate_predictions(processed_data)

# Generate a date range for the last `selected_time_frame_days` days
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=selected_time_frame_days)
date_range = pd.date_range(start=start_date, end=end_date)

# Align predictions with the corresponding dates
prediction_data = pd.DataFrame({
    'Date': date_range[-len(predictions[-len(date_range):]):],  # Align dates to match predictions
    'Prediction': predictions[-len(date_range):]
})

# Display the prediction result for the next day
st.subheader("Prediction Result")
next_day_prediction = predictions[-1]

if next_day_prediction == 1:
    st.success(f"The predicted trend for {selected_stock} is **UP** (1) for the next day.")
else:
    st.error(f"The predicted trend for {selected_stock} is **DOWN** (0) for the next day.")

# Display Prediction Trends Line Chart
st.write(f"### Stock Trend Over the Last {selected_time_frame_label}:")

# Prediction Trend Chart
prediction_chart = alt.Chart(prediction_data).mark_line(size=2).encode(
    x=alt.X(
        'Date:T',
        title='Date',
        axis=alt.Axis(format='%d %b %Y')  # Format date as "16 Dec 2024"
    ),
    y=alt.Y(
        'Prediction:Q',
        title='Prediction Trend',
        scale=alt.Scale(domain=[0, 1])
    ),
    color=alt.value('blue')
).properties(
    width=600,
    height=400
)
st.altair_chart(prediction_chart, use_container_width=True)

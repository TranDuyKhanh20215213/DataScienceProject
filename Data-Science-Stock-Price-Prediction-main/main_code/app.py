# Import necessary libraries
from datetime import date, timedelta
import pandas as pd
import streamlit as st
import joblib  # For loading .pkl models
from sklearn.preprocessing import StandardScaler
import os

# Set Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Stock Prediction",
    page_icon="📈"
)

# Add title
st.markdown("<h1 style='font-family: Arial; font-size: 25px; color: darkblue;'>Stock Prediction</h1>",
            unsafe_allow_html=True)

# Define available stocks
stock_names = ["AMZN", "AAPL", "NVDA", "MSFT", "GOOG", "META", "TSLA", "WMT", "JPM", "NFLX"]

# Add stock selection dropdown
selected_stock = st.selectbox('Select stock', stock_names)

# Load stock data from the corresponding CSV file
@st.cache_data
def load_data(stock_name):
    file_path = f"../clean_data/clean_data{stock_name}.csv"  # File path for the CSV
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])  # Ensure time column is in datetime format
        return data
    else:
        st.error(f"Data file for {stock_name} not found.")
        return None

# Load selected stock data
data = load_data(selected_stock)

# Check if data is successfully loaded
if data is not None:
    # Tab structure
    tab1, tab2 = st.tabs(["Real-time Data", "Prediction Data"])

    with tab1:
        st.subheader('Real-time Data')

        # Date range selection
        col1, col2 = st.columns([0.5, 0.5])
        max_allowed_date = date(2024, 12, 1)  # Set maximum allowed date to 1/12/2024
        with col1:
            start_date = st.date_input("Start Date",
                                       min_value=date(2016, 1, 2),
                                       max_value=max_allowed_date,
                                       value=data['Date'].min().date())
        with col2:
            end_date = st.date_input("End Date",
                                     min_value=start_date,
                                     max_value=max_allowed_date,
                                     value=min(data['Date'].max().date(), max_allowed_date))

        # Filter data by date range
        filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

        # Display filtered data
        st.write(f"Showing data for {selected_stock} from {start_date} to {end_date}")
        st.write(filtered_data)

    with tab2:
        st.subheader('Prediction Data')

        # Load trained models
        log_reg_model = joblib.load("logistic_regression_model.pkl")
        random_forest_model = joblib.load("random_forest_model.pkl")

        # Prediction settings
        n_months = st.number_input("Enter the number of months to predict:", value=1, step=1)
        period = n_months * 30

        # Prepare features for prediction
        df_train = data[['time', 'close']].copy()
        df_train['day_of_week'] = df_train['time'].dt.dayofweek
        df_train['day_of_month'] = df_train['time'].dt.day
        df_train['month'] = df_train['time'].dt.month
        df_train['year'] = df_train['time'].dt.year

        # Scale features
        scaler = StandardScaler()
        features = ['close', 'day_of_week', 'day_of_month', 'month', 'year']
        scaled_features = scaler.fit_transform(df_train[features])

        # Predict future values
        future_dates = pd.date_range(df_train['time'].iloc[-1] + timedelta(days=1), periods=period)
        log_reg_predictions = log_reg_model.predict(scaled_features[-period:])
        rf_predictions = random_forest_model.predict(scaled_features[-period:])

        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'time': future_dates,
            'logistic_regression': log_reg_predictions,
            'random_forest': rf_predictions
        })

        # Display prediction results
        st.write(f"Predicted values for {selected_stock}")
        st.line_chart(predictions_df.set_index('time'))

        with st.expander("View Prediction Data"):
            st.write(predictions_df)

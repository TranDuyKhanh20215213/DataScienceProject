import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Function to fetch stock data for a single ticker and return it as a DataFrame
def fetch_stock_data(ticker):
    # Get the current date as the end date for the stock data
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Start date for the stock data
    start_date = "2016-01-01"

    # Fetch historical data for the specified date range
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)

    # Return the DataFrame containing the stock data
    return stock_data

# Function to calculate RSI (Relative Strength Index)
def rsi(X, window=14):
  delta = X.diff(1)

  gains = delta.where(delta > 0, 0)
  losses = -delta.where(delta < 0, 0)

  avg_gains = gains.rolling(window=window, min_periods=1).mean()
  avg_losses = losses.rolling(window=window, min_periods=1).mean()

  rs = avg_gains / avg_losses
  rsi = 100 - (100 / (1 + rs))

  return rsi


def macd(X, short_window=12, long_window=29, signal=9):
  short_ema = X.ewm(span=short_window, adjust=False).mean()

  long_ema = X.ewm(span=long_window, adjust=False).mean()

  macd2 = short_ema - long_ema

  signal = macd2.ewm(span=signal, adjust=False).mean()

  return signal


def obv(X):
    obv = pd.Series(index=X.index)
    obv.iloc[0] = 0

    for i in range(1, len(X)):
        if X['Close'].iloc[i] > X['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + X['Volume'].iloc[i]
        elif X['Close'].iloc[i] < X['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - X['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv

def cmf(X, window=16):
  money_flow_multiplier = ((X['Close'] - X['Low']) - (X['High'] - X['Close']))/(X['High']-X['Low'])

  money_flow_volume = money_flow_multiplier * X['Volume']

  cmf = money_flow_volume.rolling(window=window).sum() / X['Volume'].rolling(window=window).sum()

  return cmf


def emv(X):
    emv = pd.Series(index=X.index)
    emv.iloc[0] = np.nan  # Set the first value to NaN as there is no previous data for comparison

    for i in range(1, len(X)):
        # Calculate the directional movement (dm)
        dm = 0.5 * ((X['High'].iloc[i] + X['Low'].iloc[i]) - (X['High'].iloc[i - 1] + X['Low'].iloc[i - 1]))

        # Calculate the buying pressure (br)
        br = X['Volume'].iloc[i] / (1000000 * (X['High'].iloc[i] - X['Low'].iloc[i]))

        # Calculate the EMV (Ease of Movement) value
        emv.iloc[i] = dm / br if br != 0 else 0  # Avoid division by zero

    return emv

def mfi(X, window=14):
  combine_price = (X['High'] + X['Low'] + X['Close']) / 3

  raw_money_flow = combine_price * X['Volume']

  flow_direction = (combine_price.diff() > 0).astype(int)

  positive_money_flow = flow_direction * raw_money_flow
  negative_money_flow = (1 - flow_direction) * raw_money_flow

  positive = positive_money_flow.rolling(window=window, min_periods=1).sum()
  negative = negative_money_flow.rolling(window=window, min_periods=1).sum()

  mf = positive / negative
  mfi = 100 - (100 / (1 + mf))

  return mfi
# Function to process stock data: apply technical indicators
def process_stock_data(stock_data):
    # Drop unnecessary columns ("Dividends", "Stock Splits") if they exist
    stock_data = stock_data.drop(columns=["Dividends", "Stock Splits"], errors='ignore')

    # Apply EWMA to the 'Close' column for the stock data
    stock_data['Close'] = stock_data['Close'].ewm(alpha=0.65).mean()

    # Add a 'today' column: percentage change of 'Close' from previous day
    stock_data['today'] = stock_data['Close'].pct_change() * 100

    # Add lag features for the previous 1 to 5 days of 'today'
    for j in range(1, 6):
        stock_data[f'previous{j}'] = stock_data['today'].shift(j)

    # Calculate various EMAs of the 'Close' price for different periods
    stock_data['ema50'] = stock_data['Close'] / stock_data['Close'].ewm(span=50).mean()
    stock_data['ema21'] = stock_data['Close'] / stock_data['Close'].ewm(span=21).mean()
    stock_data['ema14'] = stock_data['Close'] / stock_data['Close'].ewm(span=14).mean()
    stock_data['ema5'] = stock_data['Close'] / stock_data['Close'].ewm(span=5).mean()

    # Calculate the RSI (Relative Strength Index) for the 'Close' price
    stock_data['rsi'] = rsi(stock_data['Close'])

    # Calculate the MACD and Signal line
    stock_data['macd'] = macd(stock_data['Close'])

    window_roc = 6
    stock_data['roc'] = ((stock_data['Close'] - stock_data['Close'].shift(window_roc)) / stock_data['Close'].shift(
        window_roc)) * 100

    # Calculate True Range (TR) and Average True Range (ATR)
    stock_data['high-low'] = stock_data['High'] - stock_data['Low']
    stock_data['high-preclose'] = abs(stock_data['High'] - stock_data['Close'].shift(1))
    stock_data['low-preclose'] = abs(stock_data['Low'] - stock_data['Close'].shift(1))

    stock_data['tr'] = stock_data[['high-low', 'high-preclose', 'low-preclose']].max(axis=1)

    window_atr = 16
    stock_data['atr'] = stock_data['tr'].rolling(window=window_atr).mean()

    # Drop intermediate columns used for ATR calculation
    stock_data = stock_data.drop(['high-low', 'high-preclose', 'low-preclose', 'tr'], axis=1)

    stock_data['obv'] = obv(stock_data)

    # Calculate Chaikin Money Flow (CMF)
    stock_data['cmf'] = cmf(stock_data)

    stock_data['emv'] = emv(stock_data)

    stock_data['minimum_low'] = stock_data['Low'].rolling(window=16).min()
    stock_data['maximum_high'] = stock_data['High'].rolling(window=16).max()

    # Calculate the Stochastic Oscillator
    stock_data['stoch'] = ((stock_data['Close'] - stock_data['minimum_low']) /
                           (stock_data['maximum_high'] - stock_data['minimum_low'])) * 100

    # Drop the intermediate columns used for Stochastic Oscillator calculation
    stock_data = stock_data.drop(['minimum_low', 'maximum_high'], axis=1)

    stock_data['mfi'] = mfi(stock_data)

    stock_data['combine_price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3

    # Calculate the simple moving average of the combined price
    stock_data['sma_combine_price'] = stock_data['combine_price'].rolling(window=21).mean()

    # Calculate the mean deviation of the combined price
    # Apply custom function to calculate the mean absolute deviation (MAD) for each rolling window
    stock_data['mean_deviation'] = stock_data['combine_price'].rolling(window=21).apply(
        lambda x: (x - x.mean()).abs().mean(), raw=False)

    # Calculate the Commodity Channel Index (CCI)
    stock_data['cci'] = (stock_data['combine_price'] - stock_data['sma_combine_price']) / (
                0.015 * stock_data['mean_deviation'])

    stock_data = stock_data.drop(['combine_price', 'sma_combine_price', 'mean_deviation'], axis=1)

    # Normalize Volume using Exponential Moving Average (EMA)
    stock_data['Volume'] = stock_data['Volume'] / stock_data['Volume'].ewm(5).mean()

    # Replace 0 with NaN and drop NaN values
    stock_data.replace(0, np.nan, inplace=True)
    stock_data = stock_data.dropna()

    # Create a 'trend' column based on 'today' (positive change is 1, negative is 0)
    stock_data['trend'] = (stock_data['today'] > 0).astype(int)
    # Select the relevant columns for the final DataFrame
    df = stock_data[['today', 'previous1', 'previous2', 'previous3', 'previous4', 'previous5', 'Volume',
                             'ema50', 'ema21', 'ema14', 'ema5', 'rsi', 'macd', 'roc', 'atr', 'obv', 'cmf', 'emv',
                             'stoch', 'mfi', 'cci', 'trend']]

    # Return the processed stock data
    return df

def calculate_predictions(df):
    # Assuming that df contains all stock data with 'trend' as the target column
    # Select features (excluding 'trend')
    X = df.loc[:, df.columns != 'trend']
    y = df['trend']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Initialize the Logistic Regression model
    lr = LogisticRegression(penalty='l2', C=0.1, random_state=42)

    # Train the model
    lr.fit(X_train_scaled, y_train.values)

    # Make predictions
    predictions = lr.predict(X_test_scaled)

    # Return the predictions and the y_test set
    return predictions, y_test.values







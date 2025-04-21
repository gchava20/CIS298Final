import yfinance as yf
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



def calculate_EMA(prices: np.ndarray, window: int) -> np.ndarray:

    """
    Calculates the Exponential Moving Average (EMA) for a given price array and window size.
    The EMA gives more weight to recent prices, making it more responsive to new information.
    Used to identify buy/sell signals and trends.
    """

    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema

def calculate_bollinger_bands(prices: np.ndarray, window: int = 20) -> tuple:

    """
    Calculates Bollinger Bands for a given price array and window size.

    Bollinger Bands consist of a simple moving average (SMA) and two standard deviation bands.
    Bollinger Bands are used to identify overbought or oversold conditions in a market and thus act as a volatility indicator.

    Returns:
        - sma: Simple moving average
        - upper_band: SMA + 2 * std
        - lower_band: SMA - 2 * std
    """
    sma = np.convolve(prices, np.ones(window)/window, mode='valid')

    # Rolling standard deviation
    rolling_std = np.array([
        np.std(prices[i-window:i]) if i >= window else np.nan
        for i in range(len(prices))
    ])

    # Align std with sma length
    std_aligned = rolling_std[window - 1:]

    upper_band = sma + 2 * std_aligned
    lower_band = sma - 2 * std_aligned

    return sma, upper_band, lower_band

def calculate_RSI(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) for a given price array.
    RSI helps identify overbought or oversold conditions in a market.

    Returns:
        An array of RSI values (NaN for the first `window` entries).

    RSI is calculated after the first `window` days of data.
    Window is typically set to 14 days to ensure a good balance between both financial sensitivity and reliability.
    Contrary to popular belief, RSI is not a leading indicator, but rather a lagging one.
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[window] = np.mean(gains[:window])
    avg_loss[window] = np.mean(losses[:window])

    for i in range(window + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i-1]) / window

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[:window] = np.nan  # pad with NaNs to align with original price array

    return rsi


def calculate_SMA(ticker: yf.Ticker, time_period: str, interval: str) -> DataFrame:
    if not time_period or not interval:
        raise ValueError("Time period and interval must be specified.")

    data = ticker.history(period=time_period, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for the given ticker symbol and time period: {time_period}, {interval}")

    close_prices = data["Close"].to_numpy()

    if len(close_prices) == 0:
        raise ValueError("No price data available for the given ticker symbol and time period.")

    # Add SMA, EMA, Bollinger Bands, and RSI calculations
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_20"] = calculate_EMA(close_prices, 20)
    sma_bb, upper_bb, lower_bb = calculate_bollinger_bands(close_prices, 20)
    data["BB_Middle"] = np.concatenate((np.full(19, np.nan), sma_bb))
    data["BB_Upper"] = np.concatenate((np.full(19, np.nan), upper_bb))
    data["BB_Lower"] = np.concatenate((np.full(19, np.nan), lower_bb))
    data["RSI"] = calculate_RSI(close_prices)

    return data

# Function to train the Random Forest model
def train_random_forest(data, future_days=5):
    features = ['RSI', 'SMA_20', 'SMA_50', 'EMA_20', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    target = 'Close'
    data = data.dropna(subset=features + [target])
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict future prices
    future_features = X.iloc[-future_days:]  # Use the last `future_days` rows for prediction
    future_predictions = model.predict(future_features)

    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'Future Predictions for {future_days} days: {future_predictions}')

    return model, X_test, y_test, y_pred

def plot_SMA(data, symbol, y_test, y_pred):
    plt.figure(figsize=(12,6))

    # Prices, EMA, and Bollinger Bands Subplot (Top subplot)
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
    plt.plot(data.index, data["Close"], label="Close Price")
    plt.plot(data.index, data["SMA_20"], label="20-Day SMA")
    plt.plot(data.index, data["SMA_50"], label="50-Day SMA")
    plt.plot(data.index, data["EMA_20"], label="20-Day EMA", linestyle='--')
    plt.plot(data.index, data["BB_Upper"], label="Bollinger Upper", color='grey', linestyle=':')
    plt.plot(data.index, data["BB_Lower"], label="Bollinger Lower", color='grey', linestyle=':')
    plt.fill_between(data.index, data["BB_Upper"], data["BB_Lower"], color='grey', alpha=0.1)
    plt.title(f"{symbol} Price, Moving Averages & Bollinger Bands")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # RSI Subplot (Bottom subplot)
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
    plt.plot(data.index, data["RSI"], label="RSI", color='purple')
    plt.axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold (30)')
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.legend()
    plt.grid(True)

    # Create a new figure for Actual vs Predicted (Separate Plot)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], y_test, label="Actual Close Price", color='blue')
    plt.plot(data.index[-len(y_pred):], y_pred, label="Predicted Close Price", color='orange')
    plt.title(f"Actual vs Predicted Close Price for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to get user input for stock and interval
def get_user_input():
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL, MSFT): ").strip().upper()
    interval = input("Enter the interval (e.g., 1d, 1wk, 1mo): ").strip()
    time_period = input("Enter the time period (e.g., 1y, 6mo, 5d): ").strip()
    future_days = int(input("Enter the number of days into the future to predict: ").strip())
    ticker = yf.Ticker(ticker_symbol)
    data = calculate_SMA(ticker, time_period, interval)
    return data, ticker_symbol, interval, future_days

# Main function to run everything
def main():
    # Get user input for stock symbol, interval, and time period
    data, ticker_symbol, interval, future_days = get_user_input()

    # Train the Random Forest model
    model, X_test, y_test, y_pred = train_random_forest(data, future_days)

    # Plot the results
    plot_SMA(data, ticker_symbol, y_test, y_pred)

if __name__ == "__main__":
    main()

def export_to_csv(df: DataFrame) -> None:

    with open("data.csv", "w") as csvfile:
        df.to_csv(csvfile)
        print("Data Exported to 'data.csv' File!")
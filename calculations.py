import yfinance as yf
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np



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

    if time_period == "" or interval == "":
        return -1

    data = ticker.history(period=time_period, interval=interval)
    close_prices = data["Close"].to_numpy()

    # SMA_20 means the Simple Moving Average over 20 days
    # We need to "roll" over the 20 days and find the mean
    # Then compare that to mean of the 50 days

    # SMA_20 and SMA_50 added as new columns to the ticker dataframe
    data = ticker.history(period=time_period, interval=interval)
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()


    # EMA_20 added as a new column to the ticker dataframe
    ema_20 = calculate_EMA(close_prices, 20)
    data["EMA_20"] = ema_20

    # Bollinger Bands added as new columns to the ticker dataframe
    sma_bb, upper_bb, lower_bb = calculate_bollinger_bands(close_prices, 20)
    data["BB_Middle"] = np.concatenate((np.full(19, np.nan), sma_bb))
    data["BB_Upper"] = np.concatenate((np.full(19, np.nan), upper_bb))
    data["BB_Lower"] = np.concatenate((np.full(19, np.nan), lower_bb))

     # RSI added as a new column to the ticker dataframe
    rsi = calculate_RSI(close_prices)
    data["RSI"] = rsi

    return data

def plot_SMA(data, symbol):
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

    plt.tight_layout()
    plt.show()

def export_to_csv(df: DataFrame) -> None:

    with open("data.csv", "w") as csvfile:
        df.to_csv(csvfile)
        print("Data Exported to 'data.csv' File!")
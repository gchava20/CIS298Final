import yfinance as yf
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


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


    # EMA
    ema_20 = calculate_EMA(close_prices, 20)
    data["EMA_20"] = ema_20

    # Bollinger Bands
    sma_bb, upper_bb, lower_bb = calculate_bollinger_bands(close_prices, 20)
    data["BB_Middle"] = np.concatenate((np.full(19, np.nan), sma_bb))
    data["BB_Upper"] = np.concatenate((np.full(19, np.nan), upper_bb))
    data["BB_Lower"] = np.concatenate((np.full(19, np.nan), lower_bb))

    return data

def plot_SMA(data, symbol):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data["Close"], label="Close Price")
    plt.plot(data.index, data["SMA_20"], label="20-Day SMA")
    plt.plot(data.index, data["SMA_50"], label="50-Day SMA")
    plt.plot(data.index, data["EMA_20"], label="20-Day EMA", linestyle='--')
    plt.plot(data.index, data["BB_Upper"], label="Bollinger Upper", color='grey', linestyle=':')
    plt.plot(data.index, data["BB_Lower"], label="Bollinger Lower", color='grey', linestyle=':')
    plt.fill_between(data.index, data["BB_Upper"], data["BB_Lower"], color='grey', alpha=0.1)
    plt.title(f"{symbol} Price & SMAs")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def export_to_csv(df: DataFrame) -> None:

    with open("data.csv", "w") as csvfile:
        df.to_csv(csvfile)
        print("Data Exported to 'data.csv' File!")
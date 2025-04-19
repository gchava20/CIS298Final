import yfinance as yf
import matplotlib.pyplot as plt

def calculate_SMA(ticker: yf.Ticker, time_period: str, interval: str):

    if time_period == "" or interval == "":
        return -1

    data = ticker.history(period=time_period, interval=interval)

    # SMA_20 means the Simple Moving Average over 20 days
    # We need to "roll" over the 20 days and find the mean
    # Then compare that to mean of the 50 days

    # SMA_20 and SMA_50 added as new columns to the ticker dataframe
    data = ticker.history(period=time_period, interval=interval)
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    return data

def plot_SMA(data, symbol):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data["Close"], label="Close Price")
    plt.plot(data.index, data["SMA_20"], label="20-Day SMA")
    plt.plot(data.index, data["SMA_50"], label="50-Day SMA")
    plt.title(f"{symbol} Price & SMAs")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()
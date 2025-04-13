import yfinance as yf

def calculate_SMA(ticker: yf.Ticker, time_period: str, interval: str) -> None:

    if time_period == "" or interval == "":
        return -1

    data = ticker.history(period=time_period, interval=interval)

    # SMA_20 means the Simple Moving Average over 20 days
    # We need to "roll" over the 20 days and find the mean
    # Then compare that to mean of the 50 days

    # SMA_20 and SMA_50 added as new columns to the ticker dataframe
    print(data["Close"].rolling(window=20).mean())
    print(data["Close"].rolling(window=50).mean())
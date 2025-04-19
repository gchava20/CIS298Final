import yfinance as yf
import pandas as pd
from calculations import calculate_SMA, plot_SMA

"""def view_symbol(symbol: str) -> None:

    # User didn't select a symbol
    if not symbol:
        raise ValueError("symbol not selected")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y", interval="1d")

    calculate_SMA(ticker, "6mo", "1d")"""


def view_symbol(symbol: str) -> None:
    if not symbol:
        raise ValueError("symbol not selected")
    
    ticker = yf.Ticker(symbol)
    df = calculate_SMA(ticker, "6mo", "1d")
    if df is not None:
        plot_SMA(df, symbol)
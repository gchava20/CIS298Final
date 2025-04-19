import yfinance as yf
from pandas import DataFrame
from calculations import calculate_SMA, plot_SMA

def view_symbol(symbol: str, ret=False) -> DataFrame:
    if not symbol:
        raise ValueError("symbol not selected")
    
    ticker = yf.Ticker(symbol)
    df = calculate_SMA(ticker, "6mo", "1d")
    if df is not None:
        plot_SMA(df, symbol)
    
    if ret:
        return df
    

def view_sector(symbol: str) -> None:
    if not symbol:
        raise ValueError("symbol not selected")
    print(yf.Ticker(symbol).info.get("sector"))


def view_industry(symbol: str) -> None:
    if not symbol:
        raise ValueError("symbol not selected")
    print(yf.Ticker(symbol).info.get("industry"))
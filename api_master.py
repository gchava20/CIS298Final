import yfinance as yf
from pandas import DataFrame
from calculations import calculate_SMA, plot_SMA, train_random_forest

def view_symbol(symbol: str, ret=False) -> DataFrame:
    if not symbol:
        raise ValueError("symbol not selected")
    
    ticker = yf.Ticker(symbol)
    df = calculate_SMA(ticker, "6mo", "1d")
    
    if df is not None:
        # Train the Random Forest model
        model, X_test, y_test, y_pred = train_random_forest(df)
        
        # Plot the results
        plot_SMA(df, symbol, y_test, y_pred)
    
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
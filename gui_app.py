import tkinter as tk
from tkinter import messagebox
import yfinance as yf
from calculations import calculate_SMA, train_random_forest, plot_SMA

def run_analysis():
    symbol = symbol_entry.get().strip().upper()
    interval = interval_entry.get().strip()
    time_period = period_entry.get().strip()
    future_days = future_days_entry.get().strip()

    if not symbol or not interval or not time_period or not future_days:
        messagebox.showerror("Missing Info", "Please fill in all fields.")
        return

    try:
        future_days = int(future_days)
    except ValueError:
        messagebox.showerror("Invalid Input", "Future days must be an integer.")
        return

    try:
        ticker = yf.Ticker(symbol)
        data = calculate_SMA(ticker, time_period, interval)
        model, X_test, y_test, y_pred = train_random_forest(data, future_days)
        plot_SMA(data, symbol, y_test, y_pred)
        result_label.config(text=f"Analysis complete for {symbol}.")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# --- GUI Layout ---
root = tk.Tk()
root.title("Stock Analyzer & Predictor")
root.geometry("450x350")

tk.Label(root, text="Stock Symbol:", font=("Arial", 12)).pack(pady=5)
symbol_entry = tk.Entry(root, font=("Arial", 12))
symbol_entry.pack(pady=5)

tk.Label(root, text="Time Period (e.g. 6mo, 1y):", font=("Arial", 12)).pack(pady=5)
period_entry = tk.Entry(root, font=("Arial", 12))
period_entry.pack(pady=5)

tk.Label(root, text="Interval (e.g. 1d, 1wk):", font=("Arial", 12)).pack(pady=5)
interval_entry = tk.Entry(root, font=("Arial", 12))
interval_entry.pack(pady=5)

tk.Label(root, text="Days to Predict:", font=("Arial", 12)).pack(pady=5)
future_days_entry = tk.Entry(root, font=("Arial", 12))
future_days_entry.pack(pady=5)

tk.Button(root, text="Run Analysis", font=("Arial", 12), command=run_analysis).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 11), fg="blue")
result_label.pack(pady=10)

root.mainloop()

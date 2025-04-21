from api_master import view_symbol, view_sector, view_industry
from calculations import export_to_csv, calculate_SMA, train_random_forest, plot_SMA
from stack import Stack
import yfinance as yf

def main():
    history = Stack()
    while True:
        company = input("Which company would you like to view? If you want to quit the program, type exit: ").strip().upper()
        if company.lower() == 'exit':
            break

        history.append(f"Selected : {company}")
        selection = input("Select which action you'd like to perform:\n(1) VIEW SYMBOL\n(2) VIEW SECTOR\n(3) VIEW INDUSTRY\n(4) ANALYZE & PREDICT STOCK\n(0) VIEW USER HISTORY\n(-1) EXIT\n").strip()
        
        if selection == "-1":
            break
        elif selection == "0":
            history.append("View History")
            print("---USER HISTORY---")
            print(history)
        elif selection == "1":
            try:
                df = view_symbol(company, ret=True)
                history.append(f"View Symbol : {company}")
                if export := input("Would you like to export to CSV? (Y/N): ") == "Y":
                    export_to_csv(df)
                    history.append(f"Export CSV : {company}")
                else:
                    continue

            except Exception as e:
                print(f"Error fetching data for {company}: {e}")

        elif selection == "2":
            view_sector(company)
            history.append(f"View Sector : {company}")

        elif selection == "3":
            view_industry(company)
            history.append(f"View Industry : {company}")

        elif selection == "4":
            try:
                interval = input("Enter the interval (e.g., 1d, 1wk, 1mo): ").strip()
                time_period = input("Enter the time period (e.g., 1y, 6mo, 5d): ").strip()
                future_days = int(input("Enter the number of days into the future to predict: ").strip())
                
                ticker = yf.Ticker(company)
                data = calculate_SMA(ticker, time_period, interval)
                model, X_test, y_test, y_pred = train_random_forest(data, future_days)
                plot_SMA(data, company, y_test, y_pred)
                
                history.append(f"Ran Prediction : {company} ({time_period}, {interval})")

                if input("Would you like to export this data to CSV? (Y/N): ").strip().upper() == "Y":
                    export_to_csv(data)
                    history.append(f"Export CSV : {company}")
            except Exception as e:
                print(f"Error analyzing {company}: {e}")

        else:
            print("Invalid selection. Try again.")

if __name__ == "__main__":
    main()

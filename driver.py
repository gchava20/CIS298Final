from api_master import view_symbol, view_sector, view_industry
from calculations import export_to_csv

def main():
    while True:
        company = input("Which company would you like to view? If you want to quit the program, type exit: ").strip().upper()
        if company.lower() == 'exit':
            break

        selection = input("Select which action you'd like to perform:\n(1) VIEW SYMBOL\n(2) VIEW SECTOR\n(3) VIEW INDUSTRY\n(-1) EXIT\n").strip()
        
        if selection == "-1":
            break
        elif selection == "1":
            try:
                df = view_symbol(company, ret=True)
                if export := input("Would you like to export to CSV? (Y/N): ") == "Y":
                    export_to_csv(df)
                else:
                    continue

            except Exception as e:
                print(f"Error fetching data for {company}: {e}")

        elif selection == "2":
            view_sector(company)

        elif selection == "3":
            view_industry(company)

        else:
            print("Invalid selection. Try again.")

if __name__ == "__main__":
    main()

from api_master import view_symbol

def main():
    while True:
        company = input("Which company would you like to view? If you want to quit the program, type exit").strip().upper()
        if company.lower() == 'exit':
            break

        selection = input("Select which action you'd like to perform: (1) VIEW SYMBOL, (-1) EXIT : ").strip()
        
        if selection == "-1":
            break
        elif selection == "1":
            try:
                view_symbol(company)
            except Exception as e:
                print(f"Error fetching data for {company}: {e}")
        else:
            print("Invalid selection. Try again.")

if __name__ == "__main__":
    main()

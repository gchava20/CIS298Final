from api_master import view_symbol

def options(company: str) -> None:
    while True:
        selection = 0

        if selection == 0:
            selection = int(input("Select which action you'd like to perform: (1) VIEW SYMBOL, (-1) EXIT : ").strip())
        if selection == -1:
            break

        if selection == 1:
            view_symbol(company)


def main():
    company = input("Which company would you like to view? ").strip().upper()
    options(company)


if __name__ == "__main__":
    main()
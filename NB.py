import re


def main():
    with open('typhoon.txt', 'r') as file:
        pattern = r'\d{2}/[A-Za-z]{3}/\d{4}'
        lst = []
        for line in file:
            print(line)
            lst = lst.append(re.findall(pattern=pattern, string=line))
            print(lst)
            break
    return 0


if __name__ == "__main__":
    main()

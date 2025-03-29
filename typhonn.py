import pandas as pd


def change_abbrev(x):
    x = x.lower()
    month_abbrev_to_num = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    return month_abbrev_to_num.get(x)


def change_year(x):
    x = str(x)
    if x.startswith('9'):
        return int('19'+x)
    return int('20'+x)


def change_intensity(x):
    intensity_to_num = {
        "Tropical Depression": 1,
        "Tropical Depression/Tropical Depression": 1,
        "Tropical Storm": 2,
        "Severe Tropical Storm": 3,
        "Typhoon": 4,
        "Typhoon (Severe Typhoon)": 5,
        "Severe Typhoon": 5,
        "Typhoon (Super Typhoon)": 6,
        "Super Typhoon": 6
    }
    return intensity_to_num.get(x)


def change_signal(x):
    x = str(x)
    signal_to_num = {
        "1": 1,
        "3": 2,
        "8 NE": 3,
        "8 SE": 3,
        "8 NW": 3,
        "8 SW": 3,
        "9": 4,
        "10": 5

    }
    return signal_to_num.get(x)


def change_dur(x):
    hr = int(x.split(" ")[0])
    mins = int(x.split(" ")[1])
    return hr * 60 + mins


def change_time_zone(x):
    x = int(x.split(':')[0])
    if x >= 0 and x < 6:
        return 1
    if x >= 6 and x < 12:
        return 2
    if x >= 12 and x < 18:
        return 3
    return 4


def main():
    df = pd.read_csv('Typhoon.csv', names=[
                     'Intensity', 'Name', 'Signal', 'Start_time', 'Start_date', 'End_time', 'End_date', 'Duration'])
    df["Year"] = df["Start_date"].map(lambda x: x.split('-')[2])
    df["Month"] = df["Start_date"].map(lambda x: x.split('-')[1])
    df["Day"] = df["Start_date"].map(lambda x: x.split('-')[0])

    df["Month"] = df["Month"].apply(change_abbrev)
    df["Year"] = df["Year"].apply(change_year)
    df['Intensity'] = df['Intensity'].apply(change_intensity)
    df['Signal'] = df['Signal'].apply(change_signal)
    df['Duration'] = df['Duration'].apply(change_dur)
    df['Start_time'] = df['Start_time'].apply(change_time_zone)
    df['End_time'] = df['End_time'].apply(change_time_zone)

    # print(df.Signal.unique())
    print(df[['Year', 'Month', 'Day', 'Intensity',
          'Signal', "Duration", "Start_time", "End_time"]])


if __name__ == '__main__':
    main()

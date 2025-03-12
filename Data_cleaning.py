import pandas as pd
import numpy as np
import os


def read_csv(path):
    return pd.read_csv(path)


def sub_nan(df):
    return df.map(lambda x: np.nan if x == "***" else x)


def sub_mean_temp(df):
    df["Mean Temperature"] = df["Mean Temperature"].map(lambda x: float(x))
    return df["Mean Temperature"].fillna((df["Max Temperature"] + df["Min Temperature"]) / 2)


def sub_value(df):
    df = df.map(lambda x: float(x))
    return df.interpolate(method='linear')


def data_cleaning(df):
    df = df.drop("Data Completeness", axis=1)
    df["Rainfall"] = df["Rainfall"].map(lambda x: 0.05 if x == "Trace" else x)
    df = df.dropna(subset=["Rainfall"])
    df["Year"] = df["Date"].map(lambda x: int(x[0:4]))
    df["Month"] = df["Date"].map(lambda x: int(x[5:8].split(".")[0]))
    df["Day"] = df["Date"].map(lambda x: int(
        x[8:].split(".")[0].split(" ")[1]))
    df = df.sort_values(by=['Year', 'Month', 'Day'], ascending=True)
    df = df.loc[df["Year"] >= 1990]
    column_names = [
        "Dew Point Temp.",
        "Mean Temperature",
        "Max Temperature",
        "Min Temperature",
        "Mean Cloud",
        "Mean Pressure",
        "Rainfall",
        "Relative Humidity",
        "Wet Bulb Temp.",
        "Evaporation",
        "Global Solar Radiation",
        "Max UV",
        "Mean UV",
        "Prevailing Wind Direction",
        "Total Sunlight",
        "Wind Speed"
    ]
    for col in column_names:
        df[col] = sub_nan(df[col])
        df[col] = sub_value(df[col])
    return df


def main():
    path = r"unclean output.csv"
    input_file = read_csv(path)
    input_file = data_cleaning(input_file).set_index(
        ["Year", "Month", "Day"]).drop("Date", axis=1).sort_index()

    # os.remove(r"output.csv")
    input_file.to_csv(r"output.csv")


if __name__ == "__main__":
    main()

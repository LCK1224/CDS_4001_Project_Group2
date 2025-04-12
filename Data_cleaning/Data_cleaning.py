import pandas as pd
import numpy as np
import os
import Typhoon
import air_pollutant


def read_csv(path):
    df = pd.read_csv(path, low_memory=False).astype(str)
    return df


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
    # df = df.dropna(subset=["Rainfall"])
    df["Year"] = df["Date"].map(lambda x: int(x[0:4]))
    df["Month"] = df["Date"].map(lambda x: int(x[5:8].split(".")[0]))
    df["Day"] = df["Date"].map(lambda x: int(
        x[8:].split(".")[0].split(" ")[1]))

    df = df.sort_values(by=['Year', 'Month', 'Day'], ascending=True)
    df = df.loc[df["Year"] >= 1990]

    column_names = df.columns.values
    skip_lst = ["Date", "Year", "Month", "Day"]
    for col in column_names:
        if col in skip_lst:
            pass
        else:
            df[col] = sub_nan(df[col])
            df[col] = sub_value(df[col])
    df["Mean Temperature"] = sub_mean_temp(df)
    return df


def main():
    path = r"Data_cleaning/unclean_output.csv"
    input_file = read_csv(path)
    df = data_cleaning(input_file)
    df[["Year", "Month", "Day"]] = df[["Year", "Month", "Day"]].astype(int)
    df = df.set_index(["Year", "Month", "Day"]).drop(
        ["Date"], axis=1).sort_index()
    typhoon_df = Typhoon.main()
    df = pd.merge(df, typhoon_df, how='left',
                  left_index=True, right_index=True).fillna(0)

    air_df = air_pollutant.main()
    df = pd.merge(df, air_df, how='right',
                  left_index=True, right_index=True).fillna(0)

    df = df.drop(["Dew Point Temp.", "Max Temperature",
                 "Min Temperature", "Wet Bulb Temp."], axis=1)
    df = df.reset_index()
    print(df)
    df.to_csv(r"Data_cleaning/merge_air.csv")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def read_csv(path):
    return pd.read_csv(path)


def split_date(df):
    df["Year"] = df["DATE"].map(lambda x: x.split(
        "-")[2]).astype(str)  # Ensure type is str
    df["Month"] = df["DATE"].map(
        lambda x: str(int(x.split("-")[1]))).astype(str)
    df["Day"] = df["DATE"].map(lambda x: str(int(x.split("-")[0]))).astype(str)
    return df


def sub_nan(df):
    return df.map(lambda x: np.nan if x == "N.A." else x)


def main():
    air_df = read_csv(r"weather_forecast/air pollutant 1994 2024.csv")
    air_df = split_date(air_df)
    air_df = sub_nan(air_df)
    air_df = air_df.set_index(["Year", "Month", "Day"])

    weather_df = read_csv(r"weather_forecast/output2.csv")

    # Ensure Year, Month, Day columns are strings in weather_df as well
    weather_df["Year"] = weather_df["Year"].astype(str)
    weather_df["Month"] = weather_df["Month"].astype(str)
    weather_df["Day"] = weather_df["Day"].astype(str)
    weather_df = weather_df.set_index(["Year", "Month", "Day"])

    # Merge the two DataFrames
    merge_df = pd.merge(weather_df, air_df, on=["Year", "Month", "Day"]).drop(
        ["DATE", "STATION"], axis=1)
    merge_df = merge_df.drop("Unnamed: 0", axis=1)

    merge_df.to_csv("merge.csv")


if __name__ == "__main__":
    main()

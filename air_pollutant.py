import pandas as pd
import numpy as np


def read_csv(path):
    return pd.read_csv(path, low_memory=False)


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
    air_path = "air pollutant 1994 2024.csv"
    air_df = read_csv(air_path)
    air_df = split_date(air_df)
    air_df = sub_nan(air_df)
    air_df = air_df.set_index(["Year", "Month", "Day"])

    weather_path = "output2.csv"
    weather_df = read_csv(weather_path)

    # Ensure Year, Month, Day columns are strings in weather_df as well
    col_lst = ["Year", "Month", "Day"]
    for col in col_lst:
        weather_df[col] = weather_df[col].astype(str)
    weather_df = weather_df.set_index(col_lst)

    # Merge the two DataFrames
    merge_df = pd.merge(weather_df, air_df, on=col_lst).drop(
        ["DATE", "STATION"], axis=1)
    merge_df = merge_df.drop(["Unnamed: 0"], axis=1)
    merge_df.to_csv("merge.csv")


if __name__ == "__main__":
    main()

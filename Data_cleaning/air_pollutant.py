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
    air_path = "air_pollutant_1994_2024.csv"
    air_df = read_csv(air_path)
    air_df = split_date(air_df)
    air_df = sub_nan(air_df)
    air_df[["Year", "Month", "Day"]] = air_df[[
        "Year", "Month", "Day"]].astype(int)
    air_df = air_df.set_index(["Year", "Month", "Day"])
    air_df = air_df.drop(["DATE", "STATION"], axis=1)
    return air_df


if __name__ == "__main__":
    main()

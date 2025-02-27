import pandas as pd
import numpy as np
import os


def read_csv(path):
    return pd.read_csv(path)


def create_nan(df):
    return df.map(lambda x: np.nan if x == "***" else x)


def sub_temp(df):
    return df["Mean Temperature"].fillna((df["Max Temperature"].astype('float') + df["Min Temperature"].astype('float')) / 2)


def sub_value(df):
    return df.interpolate()

# 152 - 182


def data_cleaning(df):
    df = df.drop("Data Completeness", axis=1)
    df["Rainfall"] = df["Rainfall"].map(lambda x: 0.05 if x == "Trace" else x)
    df = df.dropna(subset=["Rainfall"])
    df["Mean Temperature"] = create_nan(df["Mean Temperature"])
    df["Mean Temperature"] = sub_temp(df)
    print(df["Mean Temperature"].iloc[94:125])

    return df


def main():
    path = "unclean output.csv"
    input_file = read_csv(path)
    input_file = data_cleaning(input_file).reset_index()
    print(input_file.head())
    os.remove("output.csv")
    input_file.to_csv("output.csv")


if __name__ == "__main__":
    main()

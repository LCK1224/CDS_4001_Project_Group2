import pandas as pd
import numpy as np


def tracker(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        # print(f"{func.__name__} is processing")
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"function:{func.__name__}\n{" ":13} time used:{end - start}")
        return ret
    return wrapper


class DataProcessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = self.df.reset_index()
        self.df["Day of Year"] = self.df.apply(self.day_of_year, axis=1)

    def day_of_year(self, row):
        from datetime import datetime

        date = datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]))
        day_of_year = (date - datetime(int(row["Year"]), 1, 1)).days + 1
        return day_of_year

    @tracker
    def load_na_row(self, col):
        return self.df[self.df[col].isna()].index.tolist()

    @tracker
    def linear_regression(self, data, label, non_zero=False):
        from sklearn.linear_model import LinearRegression as lr
        reg = lr(positive=non_zero).fit(data, label)
        return reg

    @tracker
    def fill_missing_values(self, col_name, col_range, round_result=False, non_zero=False, round_digit=0):
        nan_index = self.load_na_row(col_name)
        self.train_df = self.df.copy().dropna()

        # using log1p to force coeficient to be positive
        model = self.linear_regression(
            self.train_df.iloc[:, col_range], self.train_df[col_name], non_zero)

        predictions = model.predict(
            self.df.iloc[nan_index, col_range])
        col_to_idx = {name: idx for idx, name in enumerate(self.df.columns)}
        column_index = col_to_idx[col_name]
        self.df.iloc[nan_index, column_index] = pd.Series(
            [round(i, round_digit) if round_result else i for i in predictions])

    @tracker
    def save_to_csv(self, output_path):
        self.df.to_csv(output_path)


def main():
    processor = DataProcessor(r"weather_forecast/output.csv")

    processor.fill_missing_values(
        "Max UV", list(range(4, 10)) + list(range(11, 15)) + [18] + [20])
    processor.fill_missing_values(
        "Mean UV", list(range(4, 10)) + list(range(11, 16)) + [18] + [20])
    processor.fill_missing_values(
        "Wind Speed", list(range(4, 10)) + list(range(11, 17)) + [18] + [20])
    processor.fill_missing_values("Prevailing Wind Direction", list(range(
        4, 10)) + list(range(11, 17)) + [18, 19, 20], round_result=True, round_digit=-1)

    processor.save_to_csv("output2s.csv")


if __name__ == "__main__":
    main()

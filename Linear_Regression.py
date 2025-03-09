import pandas as pd
import numpy as np


def tracker(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        print(f"{func.__name__} is processing")
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"time used:{end - start}")
        return ret
    return wrapper


class DataProcessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = self.df.reset_index()
        self.df["Day of Year"] = self.df.apply(self.day_of_year, axis=1)
        self.train_df = self.df.iloc[3500:].reset_index().copy()

    def day_of_year(self, row):
        from datetime import datetime

        date = datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]))
        day_of_year = (date - datetime(int(row["Year"]), 1, 1)).days + 1
        return day_of_year

    @tracker
    def load_na_row(self, col):
        return self.df[self.df[col].isna()].index.tolist()

    @tracker
    def linear_regression(self, data, label):
        from sklearn.linear_model import LinearRegression as lr
        reg = lr().fit(data, label)
        return reg

    @tracker
    def fill_missing_values(self, col_name, col_range, label, round_result=False):
        nan_index = self.load_na_row(col_name)
        model = self.linear_regression(
            self.train_df.iloc[:, col_range], self.train_df[label])
        predictions = model.predict(
            self.train_df.loc[nan_index].iloc[:, col_range])
        self.df.loc[nan_index, col_name] = pd.Series(
            [round(i, -1) if round_result else i for i in predictions])

    @tracker
    def save_to_csv(self, output_path):
        self.df.to_csv(output_path)


def main():
    processor = DataProcessor("output.csv")

    processor.fill_missing_values(
        "Max UV", list(range(4, 15)) + [18] + [20], "Max UV")
    processor.fill_missing_values(
        "Mean UV", list(range(4, 16)) + [18] + [20], "Mean UV")
    processor.fill_missing_values(
        "Wind Speed", list(range(4, 17)) + [18] + [20], "Wind Speed")
    processor.fill_missing_values("Prevailing Wind Direction", list(
        range(4, 17)) + [18, 19, 20], "Prevailing Wind Direction", round_result=True)

    processor.save_to_csv("output2.csv")


if __name__ == "__main__":
    main()

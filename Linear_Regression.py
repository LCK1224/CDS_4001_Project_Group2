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
        self.train_df = self.df.iloc[3500:]

    @tracker
    def load_na_row(self, col):
        return self.df[self.df[col].isna()].index.tolist()

    @tracker
    def linear_regression(self, data, label):
        from sklearn.linear_model import LinearRegression as lr
        from sklearn.model_selection import train_test_split as tts

        X_train, X_test, y_train, y_test = tts(
            data, label, test_size=0.3, random_state=1224)
        reg = lr().fit(X_train, y_train)
        return reg, X_test, y_test

    @tracker
    def fill_missing_values(self, col_name, col_range, label):
        nan_index = self.load_na_row(col_name)
        model, _, _ = self.linear_regression(
            self.train_df.iloc[:, col_range], self.train_df[label])
        predictions = model.predict(self.df.loc[nan_index].iloc[:, col_range])
        self.df.loc[nan_index, col_name] = pd.Series(predictions)

    @tracker
    def save_to_csv(self, output_path):
        self.df.to_csv(output_path)


def main():
    processor = DataProcessor("output.csv")

    processor.fill_missing_values(
        "Max UV", list(range(1, 15)) + [18], "Max UV")
    processor.fill_missing_values(
        "Mean UV", list(range(1, 16)) + [18], "Mean UV")
    processor.fill_missing_values(
        "Wind Speed", list(range(1, 17)) + [18], "Wind Speed")
    processor.fill_missing_values("Prevailing Wind Direction", list(
        range(1, 17)) + [18, 19], "Prevailing Wind Direction")

    processor.save_to_csv("output2.csv")


if __name__ == "__main__":
    main()

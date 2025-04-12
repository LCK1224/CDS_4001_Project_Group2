import pandas as pd
import numpy as np
import air_pollutant
import Data_cleaning
import logging


class DataProcessor:
    def __init__(self, filepath=None, df=None):
        if filepath is not None:
            self.df = pd.read_csv(filepath)
        if df is not None:
            self.df = df
        if "Day of Year" not in self.df.columns:
            self.df["Day of Year"] = self.df.apply(self.day_of_year, axis=1)
        logging.basicConfig(level=logging.INFO,  # Set the logging level to INFO or lower
                            format='%(asctime)s - %(levelname)s - %(message)s'
                            )

    def getter(self):
        '''return dataframe store in dataprocessor'''
        return self.df

    @staticmethod
    def tracker(func):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            logging.info(
                f"function: {func.__name__} | time used: {end - start:.4f}s")
            return ret
        return wrapper

    def day_of_year(self, row):
        '''Convert Year, Month and Date to Day of year (range (1-366))'''
        from datetime import datetime

        date = datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]))
        day_of_year = (date - datetime(int(row["Year"]), 1, 1)).days + 1
        return day_of_year

    @tracker
    def load_na_row(self, col):
        '''find all nan value rows and save in a list'''
        return self.df[self.df[col].isna()].index.tolist()

    @tracker
    def linear_regression(self, data, label, non_zero=False):
        '''linear regression for subsitute missing value'''
        from sklearn.linear_model import LinearRegression as lr
        # using log1p to force coeficient to be positive if non_zero == True
        reg = lr(positive=non_zero).fit(data, label)
        return reg

    @tracker
    def fill_missing_values(self, col_name, col_range, round_result=False, non_zero=False, round_digit=0):
        '''subsitute all missing value by linear regression'''

        nan_index = self.load_na_row(col_name)
        self.train_df = self.df.copy().dropna()

        X = self.train_df.iloc[:, col_range]
        y = self.train_df[col_name]

        model = self.linear_regression(X, y, non_zero)

        predictions = model.predict(
            self.df.iloc[nan_index, col_range])
        column_index = self.df.columns.get_loc(col_name)
        self.df.iloc[nan_index, column_index] = pd.Series(
            [round(i, round_digit) if round_result else i for i in predictions])
        return self

    @tracker
    def save_to_csv(self, output_path):
        self.df.to_csv(output_path)
        return self


class Weather_DataProcessor(DataProcessor):
    def __init__(self, filepath=None, df=None, weather_lst=[], label=""):
        super().__init__(filepath, df)
        self.weather_lst = weather_lst
        self.label = label
        self.col_idx = []

    def weather_get_feature_data(self):
        column_names = self.df.columns.values
        skip_lst = ["Year", "Month", "Day",
                    "Mean Temperature"] + self.weather_lst

        for col in column_names:
            if col in skip_lst:
                pass
            else:
                self.col_idx.append(self.df.columns.get_loc(col))
        return self.col_idx

    def weather_fill_missing_values(self, round_result=False, non_zero=False, round_digit=0, save_csv=None):
        self.weather_get_feature_data()
        # print(f"lable = {self.label}")
        # print(f"col_idx = {self.df.columns[self.col_idx]}")
        super().fill_missing_values(self.label,
                                    self.col_idx,
                                    round_result=round_result,
                                    non_zero=non_zero,
                                    round_digit=round_digit)
        if save_csv is not None:
            return super().save_to_csv(str(save_csv))

        return self


class Air_DataProcessor(DataProcessor):
    def __init__(self, filepath, pollutant_lst):
        super().__init__(filepath)
        self.pollutant_lst = pollutant_lst
        self.col_idx = []

    def air_get_feature_data(self):
        column_names = self.df.columns.values
        skip_lst = ["Year", "Month", "Day",
                    "Rainfall", "index"] + self.pollutant_lst

        for col in column_names:
            if col in skip_lst:
                pass
            else:
                self.col_idx.append(self.df.columns.get_loc(col))
        return self.col_idx

    def air_fill_missing_values(self, round_result=False, non_zero=False, round_digit=0, save_csv=None):
        self.air_get_feature_data()
        for p in self.pollutant_lst:
            super().fill_missing_values(p,
                                        self.col_idx,
                                        round_result=round_result,
                                        non_zero=non_zero,
                                        round_digit=round_digit)
        if save_csv is not None:
            return super().save_to_csv(str(save_csv) + ".csv")

        return self


def main():

    path = "Data/unclean_output.csv"
    maxuv_processor = Weather_DataProcessor(
        filepath=path, weather_lst=["Max UV", "Mean UV", "Prevailing Wind Direction", "Wind Speed"], label="Max UV")
    maxuv_processor.weather_fill_missing_values()
    cache = maxuv_processor.getter()

    meanuv_processor = Weather_DataProcessor(df=cache, weather_lst=[
                                             "Mean UV", "Prevailing Wind Direction", "Wind Speed"], label="Mean UV")
    meanuv_processor.weather_fill_missing_values()
    cache = meanuv_processor.getter()

    winspeed_processor = Weather_DataProcessor(
        df=cache, weather_lst=["Prevailing Wind Direction", "Wind Speed"], label="Wind Speed")
    winspeed_processor.weather_fill_missing_values()
    cache = meanuv_processor.getter()

    windir_processor = Weather_DataProcessor(
        df=cache, weather_lst=["Prevailing Wind Direction"], label="Prevailing Wind Direction")
    windir_processor.weather_fill_missing_values(
        round_result=True, round_digit=-1, save_csv="output2")

    air_pollutant.main()

    air_processor = Air_DataProcessor(
        "merge.csv", ["SO2", "NOX", "NO2", "CO", "RSP", "O3", "FSP"])
    air_processor.air_fill_missing_values(
        non_zero=True, round_result=True, round_digit=0, save_csv="cleaned_dataset")


if __name__ == "__main__":
    main()

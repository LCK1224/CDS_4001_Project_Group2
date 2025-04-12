import pandas as pd
import numpy as np
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
    def linear_regression(self, data, label):
        '''linear regression for subsitute missing value'''
        from sklearn.linear_model import LinearRegression as lr
        # using log1p to force coeficient to be positive if non_zero == True
        reg = lr(n_jobs=5).fit(data, label)
        return reg

    @tracker
    def fill_missing_values(self, col_name, col_range, round_result=False, positive=False, round_digit=0):
        '''subsitute all missing value by linear regression'''

        nan_index = self.load_na_row(col_name)
        self.train_df = self.df.copy().dropna()

        X = self.train_df.iloc[:, col_range]
        y = self.train_df[col_name]

        if positive:
            y = np.log(y + 1)

        model = self.linear_regression(X, y)

        predictions = model.predict(
            self.df.iloc[nan_index, col_range])
        if positive:
            predictions = np.exp(predictions) - 1

        column_index = self.df.columns.get_loc(col_name)
        self.df.iloc[nan_index, column_index] = pd.Series(
            [round(i, round_digit) if round_result else i for i in predictions])
        return self

    @tracker
    def save_to_csv(self, output_path):
        self.df.to_csv(output_path, index=False)
        return self


class Weather_DataProcessor(DataProcessor):
    def __init__(self, filepath=None, df=None, label="", skip_lst=[]):
        super().__init__(filepath, df)
        self.label = label
        self.col_idx = []
        self.skip_lst = skip_lst

    def weather_get_feature_data(self):
        column_names = self.df.columns.values

        for col in column_names:
            if col in self.skip_lst:
                pass
            else:
                self.col_idx.append(self.df.columns.get_loc(col))
        return self.col_idx

    def weather_fill_missing_values(self, round_result=False, positive=False, round_digit=0, save_csv=None):
        self.weather_get_feature_data()
        # print(f"lable = {self.label}")
        # print(f"col_idx = {self.df.columns[self.col_idx]}")
        super().fill_missing_values(self.label,
                                    self.col_idx,
                                    round_result=round_result,
                                    positive=positive,
                                    round_digit=round_digit)
        if save_csv is not None:
            return super().save_to_csv(str(save_csv))

        return self


class Air_DataProcessor(DataProcessor):
    def __init__(self, filepath=None, df=None, pollutant_lst=[], skip_lst=[]):
        super().__init__(filepath, df)
        self.pollutant_lst = pollutant_lst
        self.col_idx = []
        self.skip_lst = skip_lst

    def air_get_feature_data(self):
        column_names = self.df.columns.values

        for col in column_names:
            if col in self.skip_lst:
                pass
            else:
                self.col_idx.append(self.df.columns.get_loc(col))
        return self.col_idx

    def air_fill_missing_values(self, round_result=False, positive=False, round_digit=0, save_csv=None):
        self.air_get_feature_data()
        for p in self.pollutant_lst:
            super().fill_missing_values(p,
                                        self.col_idx,
                                        round_result=round_result,
                                        positive=positive,
                                        round_digit=round_digit)
        if save_csv is not None:
            return super().save_to_csv(str(save_csv))

        return self


def main():

    path = "Data/clean_output.csv"
    df = pd.read_csv(path)

    skip_lst = df.columns[df.isna().any()].tolist()

    maxuv_processor = Weather_DataProcessor(
        df=df, label="Max UV", skip_lst=skip_lst)
    maxuv_processor.weather_fill_missing_values(positive=True)
    cache = maxuv_processor.getter()

    meanuv_processor = Weather_DataProcessor(
        df=cache, label="Mean UV", skip_lst=skip_lst)
    meanuv_processor.weather_fill_missing_values(positive=True)
    cache = meanuv_processor.getter()

    winspeed_processor = Weather_DataProcessor(
        df=cache, label="Wind Speed", skip_lst=skip_lst)
    winspeed_processor.weather_fill_missing_values(positive=True)
    cache = meanuv_processor.getter()

    windir_processor = Weather_DataProcessor(
        df=cache, label="Prevailing Wind Direction", skip_lst=skip_lst)
    windir_processor.weather_fill_missing_values(
        round_result=True, round_digit=-1, positive=True)
    cache = windir_processor.getter()
    eva_processor = Weather_DataProcessor(
        df=cache, label="Evaporation", skip_lst=skip_lst)
    eva_processor.weather_fill_missing_values(
        round_result=True, round_digit=-1, positive=True)
    cache = eva_processor.getter()

    air_processor = Air_DataProcessor(
        df=cache, pollutant_lst=["SO2", "NOX", "NO2", "CO", "RSP", "O3", "FSP"], skip_lst=skip_lst)
    air_processor.air_fill_missing_values(
        positive=True, round_result=True, round_digit=0)
    df = air_processor.getter()
    df[["Intensity", "Signal", "Duration(hr min)"]] = df[[
        "Intensity", "Signal", "Duration(hr min)"]].fillna(0)

    df["day_sin"] = df["Day of Year"].map(
        lambda x: np.sin(x / 365 * 2 * np.pi))
    df["day_cos"] = df["Day of Year"].map(
        lambda x: np.cos(x / 365 * 2 * np.pi))
    df["wind_sin"] = df["Prevailing Wind Direction"].map(
        lambda x: np.sin(x / 360 * 2 * np.pi))
    df["wind_cos"] = df["Prevailing Wind Direction"].map(
        lambda x: np.cos(x / 360 * 2 * np.pi))

    # df["tmr_temp"] = df["Mean Temperature"].shift(-1)
    df = df.drop(["Day of Year", "Year", "Month",
                 "Day", "Mean Temperature", "Prevailing Wind Direction"], axis=1)
    df = df.dropna()
    df.to_csv(r"Data/cleaned_dataset.csv", index=False)


if __name__ == "__main__":
    main()

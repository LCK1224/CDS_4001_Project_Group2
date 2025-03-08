import pandas as pd
import numpy as np


def tracker(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        print(f"{func.__name__} is processing")
        ret = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} time used:{end - start}")
        return ret
    return wrapper


@tracker
def linear_regression(data, label):
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.model_selection import train_test_split as tts

    X_train, X_test, y_train, y_test = tts(
        data, label, test_size=0.3, random_state=1224)
    reg = lr().fit(X_train, y_train)
    return reg, X_test, y_test


@tracker
def load_na_row(df, col):
    nan_rows = df[df[col].isna()].index.tolist()
    return nan_rows


@tracker
def linear_reg_sub(df, train_df, nan_index, col_range, label):
    model, _, _ = linear_regression(
        train_df.iloc[:, col_range], train_df[label])
    prediction = model.predict(df.loc[nan_index].iloc[:, col_range])

    return pd.Series(prediction)


@tracker
def load_data():
    df = pd.read_csv(r"output.csv")
    df = df.reset_index()
    train_df = df.iloc[3500:]
    nan_index_maxuv = load_na_row(df, "Max UV")
    nan_index_meanuv = load_na_row(df, "Mean UV")
    nan_index_windir = load_na_row(df, "Prevailing Wind Direction")
    nan_index_windspeed = load_na_row(df, "Prevailing Wind Direction")
    # breakpoint()

    return df, train_df, nan_index_maxuv, nan_index_meanuv, nan_index_windir, nan_index_windspeed


def main():
    df, train_df, nan_index_maxuv, nan_index_meanuv, nan_index_windir, nan_index_windspeed = load_data()

    # subsitute max uv
    col_range_maxuv = list(range(1, 15)) + [18]
    df.loc[nan_index_maxuv, "Max UV"] = linear_reg_sub(
        df, train_df, nan_index_maxuv, col_range_maxuv, "Max UV")

    # subsitute mean uv
    col_range_meanuv = list(range(1, 16)) + [18]
    df.loc[nan_index_meanuv, "Mean UV"] = linear_reg_sub(
        df, train_df, nan_index_meanuv, col_range_meanuv, "Mean UV")

    # subsitute Wind Speed
    col_range_windspeed = list(range(1, 17)) + [18]
    df.loc[nan_index_windspeed, "Wind Speed"] = linear_reg_sub(
        df, train_df, nan_index_windspeed, col_range_windspeed, "Wind Speed")

    # subsitute Prevailing Wind Direction
    col_range_windir = list(range(1, 17)) + [18] + [19]
    df.loc[nan_index_windir, "Prevailing Wind Direction"] = linear_reg_sub(
        df, train_df, nan_index_windir, col_range_windir, "Prevailing Wind Direction")

    df.to_csv(r"output2.csv")


if __name__ == "__main__":
    main()

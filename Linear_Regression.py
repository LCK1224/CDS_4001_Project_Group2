import pandas as pd
import numpy as np


def linear_regression(data, label):
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.model_selection import train_test_split as tts

    X_train, X_test, y_train, y_test = tts(
        data, label, test_size=0.3, random_state=1224)
    reg = lr().fit(X_train, y_train)
    return reg, X_test, y_test


def load_data():
    df = pd.read_csv(r"weather_forecast/output.csv")
    df = df.reset_index()
    df_temp = df.iloc[3500:]
    dfy1 = df_temp["Max UV"]
    nan_index_maxuv = df[df['Max UV'].isna()].index.tolist()
    nan_index_meanuv = df[df['Mean UV'].isna()].index.tolist()
    nan_index_pressure = df[df['Mean Pressure'].isna()].index.tolist()
    breakpoint()
    dfy2 = df_temp["Mean UV"]
    dfy3 = df_temp["Prevailing Wind Direction"]
    dfy4 = df_temp["Wind Speed"]

    dfx1 = df_temp.iloc[:, list(range(1, 15)) + [18]]
    breakpoint()

    return dfx1, nan_index_maxuv, dfy1, dfy2, dfy3, dfy4, df


def main():
    data1, nan_index_maxuv, label1, label2, label3, label4, df = load_data()
    model, X_test, _ = linear_regression(data1, label1)
    prediction = model.predict(
        df.loc[nan_index_maxuv].iloc[:, list(range(1, 15)) + [18]]).tolist()
    print(prediction)
    df["Prediction"] = pd.Series(prediction)
    df.loc[nan_index_maxuv, "Max UV"] = df["Prediction"]
    print(df["Max UV"])
    df.to_csv(r"weather_forecast/output2.csv")


if __name__ == "__main__":
    main()

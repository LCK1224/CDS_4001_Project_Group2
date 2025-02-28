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
    df = df.reset_index().iloc[3500:]
    df1 = df.iloc[3500:]
    dfy1 = df1["Max UV"]
    print(dfy1)
    dfy2 = df1["Mean UV"]
    dfy3 = df1["Prevailing Wind Direction"]
    dfy4 = df1["Wind Speed"]

    dfx1 = df1.iloc[:, list(range(1, 15)) + [18]]
    print(dfx1)

    return dfx1, dfy1, dfy2, dfy3, dfy4, df


def main():
    data1, label1, label2, label3, label4, df = load_data()
    model, X_test, _ = linear_regression(data1, label1)
    prediction = model.predict(
        df.iloc[0:3500].iloc[:, list(range(1, 15)) + [18]])
    print(prediction)
    df["Prediction"] = pd.Series(prediction)
    print(df["Prediction"])
    # df.to_csv(r"weather_forecast/output2.csv")


if __name__ == "__main__":
    main()

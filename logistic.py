import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


def main():
    df = pd.read_csv(r"merge_with_typhoon.csv")
    # df = df.drop(["yesterday rainfall", "Rainfall label"], axis=1)
    df["tmr rainfall"] = df["Rainfall"].shift(1)
    for i in range(1, 4):
        df[f"previous {i}th day rainfall"] = df["Rainfall"].shift(i)
    df = df.dropna()
    X = df.loc[:, df.columns != "tmr rainfall"]
    y = df["tmr rainfall"].astype('string')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=1234)

    clf = LogisticRegression(
        max_iter=100, solver='sag', n_jobs=5).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)


if __name__ == "__main__":
    main()

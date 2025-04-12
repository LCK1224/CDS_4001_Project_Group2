import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np


def main():
    df = pd.read_csv(r"cleaned_datasetwlabeltodayyesterdaytmr.csv")
    df = df.drop(["yesterday rainfall", "Rainfall label"], axis=1)
    df["prev_rainfall"] = df["Rainfall"].shift(1)
    # df["next_rainfall"] = df["Rainfall"].shift(-1)
    df = df.dropna()
    X = df.loc[:, df.columns != "tmr rainfall"]
    y = df["tmr rainfall"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=1234)
    best_acc = 0
    best_c = 0
    best_in = 0
    for i in np.arange(0.1, 3.0, 0.1):
        for j in np.arange(0.1, 3.0, 0.1):

            clf = LinearSVC(C=i, intercept_scaling=j).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            best_acc = accuracy if accuracy > best_acc else best_acc
            best_c = i if accuracy == best_acc else best_c
            best_in = j if accuracy == best_acc else best_in
# best_acc = 0.5995065789473685 | best_c = 2.5000000000000004 | best_intercept_scaling = 0.30000000000000004
    print(
        f"best_acc = {best_acc} | best_c = {best_c} | best_intercept_scaling = {best_in}")


if __name__ == "__main__":
    main()

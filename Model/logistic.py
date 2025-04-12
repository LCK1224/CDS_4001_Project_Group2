import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def tempintensity(x):
    '''
    Convert rainfall to oridinal data
    '''
    if x <= 10.0:
        return 0
    if x <= 15.0:
        return 1
    if x <= 20.0:
        return 2
    if x <= 25.0:
        return 3
    if x <= 30.0:
        return 4
    return 5


def main():
    df = pd.read_csv(
        r"C:\Users\leech\Desktop\weather_forecast\Data\cleaned_dataset.csv")
    df["Mean Temperature"] = df["Mean Temperature"].shift(-1)
    df["Mean Temperature"] = df["Mean Temperature"].map(
        lambda x: tempintensity(x))
    df["Mean Temperature"] = df["Mean Temperature"].astype('string')
    # df["next_rainfall"] = df["Rainfall"].shift(-1)
    df = df.dropna()
    X = df.drop('Mean Temperature', axis=1).values
    y = df['Mean Temperature'].values
    print(df['Mean Temperature'].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=1234)

    clf = LogisticRegression(
        max_iter=10000, solver='sag', n_jobs=5).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {acc * 100:.2f}%')
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")


if __name__ == "__main__":
    main()

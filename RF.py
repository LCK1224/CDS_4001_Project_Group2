import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    df = pd.read_csv(r"merge_with_typhoon.csv")
    # df = df.drop(["yesterday rainfall", "Rainfall label"], axis=1)
    # for i in range(1, 4):
    #     df[f"previous {i}th day rainfall"] = df["Rainfall"].shift(i)
    df["tmr rainfall"] = df["Rainfall"].shift(1)
    df["tmr rainfall"] = df["tmr rainfall"].astype('string')
    # df["next_rainfall"] = df["Rainfall"].shift(-1)
    df = df.dropna()
    X = df.loc[:, df.columns != "tmr rainfall"]
    y = df["tmr rainfall"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=1234)
    best_acc = 0
    best_depth = 0

    for i in range(1, 20):
        clf = RandomForestClassifier(
            max_depth=i, random_state=0, max_features='log2', n_jobs=5, n_estimators=115).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        best_acc = accuracy if accuracy > best_acc else best_acc
        best_depth = i if accuracy == best_acc else best_depth
        print(accuracy)
    print(f"best_acc = {best_acc} | best_depth = {best_depth}")

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

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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

# def rainintensity(x):
#     '''
#     Convert rainfall to ordinal data
#     '''
#     if x <= 9.9:
#         return 0
#     if x <= 24.9:
#         return 1
#     if x <= 49.9:
#         return 2
#     if x <= 99.9:
#         return 3
#     if x <= 249.9:
#         return 4
#     return 5


def main():
    df = pd.read_csv(r"merge_with_typhoon.csv")
    df["label"] = df["Mean Temperature"].shift(1)
    df["label"] = df["label"].map(
        lambda x: tempintensity(x))
    df["label"] = df["label"].astype('string')
    df = df.drop(["Max Temperature", "Mean Temperature",
                 "Min Temperature"], axis=1)
    # df["next_rainfall"] = df["Rainfall"].shift(-1)
    df = df.dropna()
    X = df.drop('label', axis=1).values
    y = df['label'].values
    print(df['label'].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=1234)
    best_acc = 0
    best_depth = 0

    for i in range(10, 30):
        clf = RandomForestClassifier(
            max_depth=i, random_state=1234, max_features=None, n_jobs=5, n_estimators=104).fit(X_train, y_train)
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
    mat = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4, 5])
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    cm_display.plot()
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def tempintensity(x):
    '''
    Convert rainfall to oridinal data
    '''
    if x <= 12.0:
        return 0
    if x <= 17.0:
        return 1
    if x <= 22.0:
        return 2
    if x <= 27.0:
        return 3
    return 4


def main():
    df = pd.read_csv(r"Data/cleaned_dataset.csv")
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
    best_acc = 0
    best_c = 0
    best_in = 0
    # for i in np.arange(0.1, 3.0, 0.1):
    #     for j in np.arange(0.1, 3.0, 0.1):

    clf = LinearSVC(random_state=1234).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # best_acc = accuracy if accuracy > best_acc else best_acc
    # best_c = i if accuracy == best_acc else best_c
    # best_in = j if accuracy == best_acc else best_in
# best_acc = 0.5995065789473685 | best_c = 2.5000000000000004 | best_intercept_scaling = 0.30000000000000004
    print(
        f"best_acc = {best_acc} | best_c = {best_c} | best_intercept_scaling = {best_in}")
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    mat = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=mat, display_labels=[0, 1, 2, 3, 4])
    print(f'Accuracy: {acc * 100:.2f}%')
    print(f"F1-score: {f1 * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    cm_display.plot()
    plt.show()


if __name__ == "__main__":
    main()
